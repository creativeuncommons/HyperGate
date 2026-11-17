use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

use crate::classifier::SystemOneClassifier;
use crate::ledger::BudgetLedger;
use crate::pricing::PricingConfig;

#[derive(Clone)]
pub struct AppState {
    pub pricing: Arc<PricingConfig>,
    pub ledger: BudgetLedger,
    pub classifier: SystemOneClassifier,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<ChatMessage>>,
    pub trace_id: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
}

pub async fn health_check() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "service": "hypergate-rs",
        "version": "0.1.0"
    }))
}

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let models_data: Vec<Value> = state
        .pricing
        .models
        .iter()
        .map(|(id, info)| {
            json!({
                "id": id,
                "object": "model",
                "owned_by": info.provider.as_deref().unwrap_or("hypergate"),
                "tier": info.tier.as_deref().unwrap_or("TIER_2_BALANCED"),
                "pricing": {
                    "prompt_per_1m": info.prompt_per_1m,
                    "completion_per_1m": info.completion_per_1m
                }
            })
        })
        .collect();

    Json(json!({
        "object": "list",
        "data": models_data
    }))
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let t0 = Instant::now();

    // 1. Resolve & validate Trace ID (prevent header injection / slice panics)
    let raw_trace_id = headers
        .get("x-hypergate-trace-id")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
        .or(payload.trace_id);
    let trace_id = match raw_trace_id {
        Some(ref tid) if !tid.is_empty() && tid.len() <= 64 && tid.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') => {
            tid.clone()
        }
        _ => format!("trc_{}", &Uuid::new_v4().simple().to_string()[..12]),
    };

    let custom_limit = headers
        .get("x-hypergate-budget-limit")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok());

    let requested_model = payload.model.unwrap_or_else(|| "gpt-4o-mini".to_string());
    let messages = payload.messages.unwrap_or_default();

    let full_prompt: String = messages
        .iter()
        .filter_map(|m| m.content.as_deref())
        .collect::<Vec<_>>()
        .join("\n");

    // 2. System-One Fast-Path Classification
    let (routing_tier, _rationale) = state.classifier.classify(&full_prompt);

    // 3. Pre-flight Budget & Circuit Breaker Check
    let prompt_tokens = state.pricing.estimate_tokens(&full_prompt);
    let est_cost = state.pricing.calculate_cost(&requested_model, prompt_tokens, prompt_tokens);

    if let Err((msg, rolling_cost, limit_usd)) = state.ledger.check_preflight(&trace_id, est_cost, custom_limit) {
        let overhead_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let err_body = json!({
            "error": {
                "code": "budget_exceeded",
                "message": msg,
                "type": "circuit_breaker_error",
                "trace_id": trace_id,
                "rolling_cost_usd": (rolling_cost * 1_000_000.0).round() / 1_000_000.0,
                "limit_usd": limit_usd
            }
        });

        let mut resp = Response::builder()
            .status(StatusCode::TOO_MANY_REQUESTS)
            .header("Content-Type", "application/json")
            .header("x-hypergate-trace-id", &trace_id)
            .header("x-hypergate-routing-tier", routing_tier)
            .header("x-hypergate-latency-overhead-ms", format!("{:.3}", overhead_ms))
            .header("x-hypergate-cost-usd", "0.000000")
            .header("x-hypergate-rolling-cost-usd", format!("{:.6}", rolling_cost))
            .header("x-hypergate-circuit-tripped", "true")
            .body(Body::from(serde_json::to_string(&err_body).unwrap()))
            .unwrap();
        return resp;
    }

    // 4. Execution / Completion Generation
    let short_tid: String = trace_id.chars().take(8).collect();
    let (resp_text, completion_tokens) = if routing_tier == "TIER_0_CACHE" {
        (format!("[HyperGate Cache Hit] Response for trace {}", short_tid), 0)
    } else if routing_tier == "TIER_1_FAST" && full_prompt.len() < 80 {
        (format!("[HyperGate FastPath] ack: {}", full_prompt.trim()), 5)
    } else {
        (
            format!("[HyperGate Routed ({})] Synthetic completion for model {}.", routing_tier, requested_model),
            state.pricing.estimate_tokens("Synthetic completion response.")
        )
    };

    let total_tokens = prompt_tokens + completion_tokens;
    let cost_usd = state.pricing.calculate_cost(&requested_model, prompt_tokens, completion_tokens);
    let (rolling_cost, tripped) = state.ledger.record_cost(&trace_id, cost_usd);
    let total_overhead_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let resp_payload = json!({
        "id": format!("chatcmpl-{}", &Uuid::new_v4().simple().to_string()[..12]),
        "object": "chat.completion",
        "created": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": resp_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .header("x-hypergate-trace-id", &trace_id)
        .header("x-hypergate-routing-tier", routing_tier)
        .header("x-hypergate-latency-overhead-ms", format!("{:.3}", total_overhead_ms))
        .header("x-hypergate-cost-usd", format!("{:.6}", cost_usd))
        .header("x-hypergate-rolling-cost-usd", format!("{:.6}", rolling_cost))
        .header("x-hypergate-circuit-tripped", if tripped { "true" } else { "false" })
        .body(Body::from(serde_json::to_string(&resp_payload).unwrap()))
        .unwrap()
}
