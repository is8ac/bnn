use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct PerfResults {
    pub machine_type: String,
    pub cpu_arch: String,
    pub n_cores: usize,
    pub n_physical: usize,
    pub price: f64,
    pub spot_price: f64,
    pub tests: Vec<PerfTest>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PerfTest {
    pub algorithm: String,
    pub n_threads: usize,
    pub chunk_size: usize,
    pub n_examples: usize,
    pub unit_nanos: u128,
    pub exp_nanos: u128,
    pub apply_nanos: u128,
    pub total_nanos: u128,
    pub unit_hash: u64,
    pub exp_hash: u64,
}
