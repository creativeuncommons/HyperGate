use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct SessionState {
    pub trace_id: String,
    pub budget_limit_usd: f64,
    pub max_turns: u32,
    pub current_turns: u32,
    pub rolling_cost_usd: f64,
    pub circuit_tripped: bool,
    pub created_at: Instant,
}

#[derive(Clone)]
pub struct BudgetLedger {
    sessions: Arc<DashMap<String, SessionState>>,
    default_limit_usd: f64,
    default_max_turns: u32,
}

impl BudgetLedger {
    pub fn new(default_limit_usd: f64, default_max_turns: u32) -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            default_limit_usd,
            default_max_turns,
        }
    }

    pub fn get_or_create(&self, trace_id: &str, custom_limit: Option<f64>) -> SessionState {
        self.sessions
            .entry(trace_id.to_string())
            .or_insert_with(|| SessionState {
                trace_id: trace_id.to_string(),
                budget_limit_usd: custom_limit.unwrap_or(self.default_limit_usd),
                max_turns: self.default_max_turns,
                current_turns: 0,
                rolling_cost_usd: 0.0,
                circuit_tripped: false,
                created_at: Instant::now(),
            })
            .clone()
    }

    pub fn check_preflight(&self, trace_id: &str, estimated_cost: f64, custom_limit: Option<f64>) -> Result<(), (String, f64, f64)> {
        let mut entry = self.sessions.entry(trace_id.to_string()).or_insert_with(|| SessionState {
            trace_id: trace_id.to_string(),
            budget_limit_usd: custom_limit.unwrap_or(self.default_limit_usd),
            max_turns: self.default_max_turns,
            current_turns: 0,
            rolling_cost_usd: 0.0,
            circuit_tripped: false,
            created_at: Instant::now(),
        });

        if entry.circuit_tripped {
            return Err((
                format!(
                    "HyperGate Circuit Breaker Tripped: session budget ceiling (${:.2}) previously exceeded.",
                    entry.budget_limit_usd
                ),
                entry.rolling_cost_usd,
                entry.budget_limit_usd,
            ));
        }

        if (entry.rolling_cost_usd + estimated_cost) > entry.budget_limit_usd {
            entry.circuit_tripped = true;
            return Err((
                format!(
                    "HyperGate Circuit Breaker Tripped: session budget ceiling (${:.2}) exceeded. Runaway loop terminated.",
                    entry.budget_limit_usd
                ),
                entry.rolling_cost_usd,
                entry.budget_limit_usd,
            ));
        }

        if entry.current_turns >= entry.max_turns {
            entry.circuit_tripped = true;
            return Err((
                format!(
                    "HyperGate Circuit Breaker Tripped: max turn limit ({}) exceeded.",
                    entry.max_turns
                ),
                entry.rolling_cost_usd,
                entry.budget_limit_usd,
            ));
        }

        Ok(())
    }

    pub fn record_cost(&self, trace_id: &str, cost_usd: f64) -> (f64, bool) {
        if let Some(mut entry) = self.sessions.get_mut(trace_id) {
            entry.current_turns += 1;
            entry.rolling_cost_usd += cost_usd;
            if entry.rolling_cost_usd >= entry.budget_limit_usd || entry.current_turns > entry.max_turns {
                entry.circuit_tripped = true;
            }
            (entry.rolling_cost_usd, entry.circuit_tripped)
        } else {
            (cost_usd, false)
        }
    }

    pub fn get_session(&self, trace_id: &str) -> Option<SessionState> {
        self.sessions.get(trace_id).map(|s| s.clone())
    }
}
