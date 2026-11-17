use dashmap::DashMap;
use std::sync::Arc;

pub const TIER_0_CACHE: &str = "TIER_0_CACHE";
pub const TIER_1_FAST: &str = "TIER_1_FAST";
pub const TIER_2_BALANCED: &str = "TIER_2_BALANCED";
pub const TIER_3_REASONING: &str = "TIER_3_REASONING";

#[derive(Clone)]
pub struct SystemOneClassifier {
    cache: Arc<DashMap<String, String>>,
}

impl Default for SystemOneClassifier {
    fn default() -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
        }
    }
}

impl SystemOneClassifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn classify(&self, prompt: &str) -> (&'static str, &'static str) {
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            return (TIER_1_FAST, "empty_prompt");
        }

        // Tier 0: Semantic Cache Hit
        if self.cache.contains_key(trimmed) {
            return (TIER_0_CACHE, "semantic_cache_hit");
        }

        let lower = trimmed.to_lowercase();

        // Tier 1: Deterministic Heuristic Fast Path
        if trimmed.len() < 80 {
            for p in &["ping", "health", "echo", "status", "version", "true or false", "summarize in 3 words"] {
                if lower.contains(p) {
                    return (TIER_1_FAST, "heuristic_match");
                }
            }
        }

        if trimmed.len() < 120 && trimmed.chars().any(|c| matches!(c, '+' | '-' | '*' | '/' | '%' | '^')) {
            return (TIER_1_FAST, "arithmetic_fast_path");
        }

        // Tier 3: Escalation to Frontier Reasoning
        for trig in &[
            "formal proof",
            "derive the mathematical",
            "multi-step invariant",
            "adversarial attack vector",
            "prove correctness",
            "concurrency race condition",
            "simd vectorization",
        ] {
            if lower.contains(trig) {
                return (TIER_3_REASONING, "frontier_deep_reasoning");
            }
        }

        if trimmed.len() > 4000 {
            return (TIER_3_REASONING, "large_context_reasoning");
        }

        // Tier 2: Balanced Default
        (TIER_2_BALANCED, "balanced_system_two")
    }

    pub fn put_cache(&self, key: String, value: String) {
        if self.cache.len() > 20000 {
            self.cache.clear();
        }
        self.cache.insert(key, value);
    }
}
