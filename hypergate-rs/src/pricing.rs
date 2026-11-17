use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPriceInfo {
    pub provider: Option<String>,
    pub tier: Option<String>,
    pub prompt_per_1m: f64,
    pub completion_per_1m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingConfig {
    pub version: Option<String>,
    #[serde(default = "default_char_ratio")]
    pub char_to_token_ratio: f64,
    #[serde(default = "default_budget_limit")]
    pub default_budget_limit_usd: f64,
    #[serde(default = "default_max_turns")]
    pub default_max_turns: u32,
    #[serde(default)]
    pub models: HashMap<String, ModelPriceInfo>,
}

fn default_char_ratio() -> f64 {
    4.0
}
fn default_budget_limit() -> f64 {
    0.50
}
fn default_max_turns() -> u32 {
    30
}

impl Default for PricingConfig {
    fn default() -> Self {
        let mut models = HashMap::new();
        models.insert(
            "cache".to_string(),
            ModelPriceInfo {
                provider: Some("local".to_string()),
                tier: Some("TIER_0_CACHE".to_string()),
                prompt_per_1m: 0.0,
                completion_per_1m: 0.0,
            },
        );
        models.insert(
            "gpt-4o-mini".to_string(),
            ModelPriceInfo {
                provider: Some("openai".to_string()),
                tier: Some("TIER_1_FAST".to_string()),
                prompt_per_1m: 0.15,
                completion_per_1m: 0.60,
            },
        );
        models.insert(
            "gpt-4o".to_string(),
            ModelPriceInfo {
                provider: Some("openai".to_string()),
                tier: Some("TIER_2_BALANCED".to_string()),
                prompt_per_1m: 2.50,
                completion_per_1m: 10.00,
            },
        );
        models.insert(
            "claude-3-5-sonnet".to_string(),
            ModelPriceInfo {
                provider: Some("anthropic".to_string()),
                tier: Some("TIER_2_BALANCED".to_string()),
                prompt_per_1m: 3.00,
                completion_per_1m: 15.00,
            },
        );
        models.insert(
            "o3-mini".to_string(),
            ModelPriceInfo {
                provider: Some("openai".to_string()),
                tier: Some("TIER_3_REASONING".to_string()),
                prompt_per_1m: 1.10,
                completion_per_1m: 4.40,
            },
        );

        Self {
            version: Some("1.0.0".to_string()),
            char_to_token_ratio: 4.0,
            default_budget_limit_usd: 0.50,
            default_max_turns: 30,
            models,
        }
    }
}

impl PricingConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Self {
        if let Ok(content) = fs::read_to_string(path) {
            if let Ok(cfg) = serde_json::from_str::<PricingConfig>(&content) {
                return cfg;
            }
        }
        Self::default()
    }

    pub fn estimate_tokens(&self, text: &str) -> u32 {
        if text.is_empty() {
            return 0;
        }
        let est = (text.len() as f64 / self.char_to_token_ratio).ceil() as u32;
        est.max(1)
    }

    pub fn calculate_cost(&self, model: &str, prompt_tokens: u32, completion_tokens: u32) -> f64 {
        let (p_rate, c_rate) = match self.models.get(model) {
            Some(info) => (info.prompt_per_1m, info.completion_per_1m),
            None => (2.50, 10.00), // Default balanced rate
        };

        let p_cost = (prompt_tokens as f64 / 1_000_000.0) * p_rate;
        let c_cost = (completion_tokens as f64 / 1_000_000.0) * c_rate;
        p_cost + c_cost
    }

    pub fn get_model_tier(&self, model: &str) -> String {
        self.models
            .get(model)
            .and_then(|info| info.tier.clone())
            .unwrap_or_else(|| "TIER_2_BALANCED".to_string())
    }
}
