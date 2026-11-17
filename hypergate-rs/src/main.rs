use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

mod classifier;
mod handlers;
mod ledger;
mod pricing;

use classifier::SystemOneClassifier;
use handlers::{chat_completions, health_check, list_models, AppState};
use ledger::BudgetLedger;
use pricing::PricingConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = env::var("HYPERGATE_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("HYPERGATE_PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse::<u16>()
        .unwrap_or(8080);

    let config_path = env::var("HYPERGATE_CONFIG").unwrap_or_else(|_| "config/model_pricing.json".to_string());

    println!("⚡ Initializing HyperGate Core Gateway (Rust Axum/Tokio)...");
    println!("   Config path: {}", config_path);

    let pricing = Arc::new(PricingConfig::load_from_file(&config_path));
    let ledger = BudgetLedger::new(pricing.default_budget_limit_usd, pricing.default_max_turns);
    let classifier = SystemOneClassifier::new();

    let state = AppState {
        pricing,
        ledger,
        classifier,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .layer(cors)
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    println!("🚀 HyperGate Core listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
