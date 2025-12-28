use dcsr::{Dcsr, DcsrView};
use ulib::{Device, UVec, UniversalCopy, AsUPtr};
use std::time::Instant;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};

// ----------------------------------------------------------------------------
// SECTION 0: Configuration (JSON-based)
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkConfig {
    pub num_nodes: u32,
    pub avg_degree: usize,
    pub super_node_percent: f32,
    pub super_node_degree: usize,
    pub update_rounds: usize,
    pub batch_size: usize,
    pub iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_nodes: 1_000_000,
            avg_degree: 10,
            super_node_percent: 0.05,
            super_node_degree: 200,
            update_rounds: 10,
            batch_size: 2000,
            iterations: 3,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkSuite {
    pub configs: Vec<BenchmarkConfig>,
}

impl BenchmarkSuite {
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let suite: BenchmarkSuite = serde_json::from_str(&content)?;
        Ok(suite)
    }

    pub fn from_default() -> Self {
        Self {
            configs: vec![BenchmarkConfig::default()],
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    pub config: BenchmarkConfig,
    pub dcsr_avg_time_ms: f64,
    pub naive_avg_time_ms: f64,
    pub speedup: f64,
}

impl BenchmarkResult {
    pub fn print_table_row(&self) {
        println!(
            "{:>10} | {:>8} | {:>8} | {:>6.1}% | {:>6} | {:>6} | {:>10.2} | {:>10.2} | {:>6.2}x",
            self.config.num_nodes,
            self.config.batch_size,
            self.config.update_rounds,
            self.config.super_node_percent * 100.0,
            self.config.avg_degree,
            self.config.super_node_degree,
            self.dcsr_avg_time_ms,
            self.naive_avg_time_ms,
            self.speedup,
        );
    }

    pub fn print_table_header() {
        println!(
            "{:>10} | {:>8} | {:>8} | {:>6} | {:>6} | {:>6} | {:>10} | {:>10} | {:>6}",
            "Nodes", "Batch", "Rounds", "Super%", "AvgDeg", "SupDeg",
            "DCSR(ms)", "Naive(ms)", "Speedup"
        );
        println!("{}", "-".repeat(85));
    }
}

// ----------------------------------------------------------------------------
// SECTION 1: The Baseline - Naive CSR implemented with UVec
// ----------------------------------------------------------------------------
// This represents the "brute-force" approach: every update triggers a full
// re-allocation and copy on the CPU, then pushes back to the GPU.
// If your Dcsr is slower than this, throw your design in the trash.

struct NaiveCsr<T: UniversalCopy> {
    // --- CPU-side "Logical View" ---
    adj: Vec<Vec<T>>, // The data structure users interact with

    // --- GPU-side "Physical View" (Flat CSR) ---
    pub indptr: UVec<u32>,
    pub data: UVec<T>,

    device: Device,
}

impl<T: UniversalCopy + Clone> NaiveCsr<T> {
    pub fn new(num_nodes: u32, device: Device) -> Self {
        let _ctx = device.get_context();

        // Initialize CPU-side structure
        let adj = vec![Vec::new(); num_nodes as usize];

        // Initialize GPU-side structures (initially empty but allocated)
        let indptr = UVec::from(vec![0u32; (num_nodes + 1) as usize]);
        Self {
            adj,
            indptr,
            data: UVec::with_capacity(0, device),
            device,
        }
    }

    // --- Staging APIs (operate on CPU, very fast) ---

    pub fn append(&mut self, node_id: u32, new_neighbors: &[T]) {
        self.adj[node_id as usize].extend_from_slice(new_neighbors);
    }

    // --- Commit API (The heavy lifting) ---
    pub fn commit(&mut self) {
        let _ctx = self.device.get_context();

        // 1. Rebuild flat CSR structure on CPU from the `adj` list
        let num_nodes = self.adj.len();
        let mut indptr_vec = Vec::with_capacity(num_nodes + 1);
        let mut data_vec = Vec::new();

        indptr_vec.push(0);
        let mut current_offset = 0;

        for neighbors in &self.adj {
            data_vec.extend_from_slice(neighbors);
            current_offset += neighbors.len();
            indptr_vec.push(current_offset as u32);
        }

        // 2. Push the newly built structures to the GPU
        //    UVec::from will handle the H2D copy.
        self.indptr = UVec::from(indptr_vec);
        self.data = UVec::from(data_vec);

        let _ = self.indptr.as_uptr(self.device);
        let _ = self.data.as_uptr(self.device);
    }
}

// ----------------------------------------------------------------------------
// SECTION 2: FFI Verification Helpers
// ----------------------------------------------------------------------------

extern "C" {
    fn dcsr_test_verify_sum(view: DcsrView, results: *mut i32, count: u32);
    fn naive_csr_test_verify_sum(
        d_indptr: *const u32,
        d_data: *const i32,
        results: *mut i32,
        count: u32,
    );
}

// Rust wrappers for the FFI calls
fn verify_dcsr(dcsr: &Dcsr<i32>, count: u32) -> Vec<i32> {
    let mut results = vec![0i32; count as usize];
    let view = dcsr.view();
    unsafe {
        dcsr_test_verify_sum(view, results.as_mut_ptr(), count);
    }
    results
}

fn verify_naive_csr(csr: &NaiveCsr<i32>, count: u32) -> Vec<i32> {
    let mut results = vec![0i32; count as usize];
    let _ctx = csr.device.get_context();
    let d_indptr = csr.indptr.as_uptr(csr.device);
    let d_data = csr.data.as_uptr(csr.device);
    unsafe {
        naive_csr_test_verify_sum(d_indptr, d_data, results.as_mut_ptr(), count);
    }
    results
}

// ----------------------------------------------------------------------------
// SECTION 3: Single Benchmark Run
// ----------------------------------------------------------------------------

fn run_single_benchmark(config: &BenchmarkConfig) -> BenchmarkResult {
    println!("\n>>> [BENCHMARK] DCSR vs Naive CSR Update Performance");
    println!("Configuration: {:?}", config);

    let device = Device::CUDA(0);

    // ==========================================
    // 1. Data Prep
    // ==========================================
    println!("    Generating synthetic graph data...");
    use rand::prelude::*;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(42);

    let num_super_nodes = (config.num_nodes as f32 * config.super_node_percent) as u32;
    let mut super_nodes: std::collections::HashSet<u32> = std::collections::HashSet::new();
    while super_nodes.len() < num_super_nodes as usize {
        super_nodes.insert(rng.gen_range(0..config.num_nodes));
    }

    let normal_data = vec![1i32; config.avg_degree];
    let super_data = vec![1i32; config.super_node_degree];

    // Template Data
    let mut template_adj: Vec<Vec<i32>> = Vec::with_capacity(config.num_nodes as usize);
    for i in 0..config.num_nodes {
        if super_nodes.contains(&i) {
            template_adj.push(super_data.clone());
        } else {
            template_adj.push(normal_data.clone());
        }
    }
    println!("    Data generation complete.");

    // ==========================================
    // 2. Benchmarking Loop
    // ==========================================
    let mut dcsr_total_time = std::time::Duration::new(0, 0);
    let mut naive_total_time = std::time::Duration::new(0, 0);

    let update_data = vec![5i32; 1];  // 每次只增加一个元素
    let super_update_data = vec![10i32; 1];  // 每次只增加一个元素

    for iter in 0..config.iterations {
        println!("    Iteration {}/{} (Resetting State...)", iter + 1, config.iterations);

        // --- Context Reset ---
        let mut dcsr_graph = Dcsr::<i32>::new_from_data(template_adj.clone(), device).unwrap();

        let mut naive_graph = NaiveCsr::<i32>::new(config.num_nodes, device);
        for (i, neighbors) in template_adj.iter().enumerate() {
            naive_graph.append(i as u32, neighbors);
        }
        naive_graph.commit(); // Initial commit

        if iter == 0 {
            let dcsr_res = verify_dcsr(&dcsr_graph, config.num_nodes);
            let naive_res = verify_naive_csr(&naive_graph, config.num_nodes);
            assert_eq!(dcsr_res, naive_res, "Mismatch after setup");
        }

        // --- Round Loop ---
        for r in 0..config.update_rounds {
            let nodes_to_update: Vec<u32> = (0..config.batch_size)
                .map(|_| rng.gen_range(0..config.num_nodes))
                .collect();

            // >>>>>> Benchmark Dcsr <<<<<<
            let start = Instant::now();
            for &node_id in &nodes_to_update {
                if super_nodes.contains(&node_id) {
                    dcsr_graph.append(node_id, &super_update_data);
                } else {
                    dcsr_graph.append(node_id, &update_data);
                }
            }
            dcsr_graph.commit().unwrap();
            dcsr_total_time += start.elapsed();

            // >>>>>> Benchmark Naive CSR <<<<<<
            let start = Instant::now();
            for &node_id in &nodes_to_update {
                if super_nodes.contains(&node_id) {
                    naive_graph.append(node_id, &super_update_data);
                } else {
                    naive_graph.append(node_id, &update_data);
                }
            }
            naive_graph.commit();
            naive_total_time += start.elapsed();

            // Intermittent Verification
            if iter == 0 && (r + 1) % (config.update_rounds / 5).max(1) == 0 {
                let dcsr_res = verify_dcsr(&dcsr_graph, config.num_nodes);
                let naive_res = verify_naive_csr(&naive_graph, config.num_nodes);
                assert_eq!(dcsr_res, naive_res, "Mismatch at round {}", r + 1);
            }
        } // End Rounds
    } // End Iterations

    // ==========================================
    // 3. Statistics & Reporting
    // ==========================================

    let total_commits = (config.iterations * config.update_rounds) as f64;

    let dcsr_avg_time_ms = dcsr_total_time.as_secs_f64() * 1000.0 / total_commits;
    let naive_avg_time_ms = naive_total_time.as_secs_f64() * 1000.0 / total_commits;
    let speedup = naive_avg_time_ms / dcsr_avg_time_ms;

    println!("\n----- BENCHMARK RESULTS -----");
    println!("Total Commits Measured: {}", total_commits);
    println!("DCSR (Log-Structured) Avg Latency: {:>10.4} ms / batch", dcsr_avg_time_ms);
    println!("Naive CSR (Realloc)   Avg Latency: {:>10.4} ms / batch", naive_avg_time_ms);
    println!("Speedup: {:.2}x", speedup);
    println!("-----------------------------\n");

    assert!(
        dcsr_total_time < naive_total_time,
        "FAILURE: Your DCSR is slower than the brute-force baseline. Go back to the drawing board."
    );
    println!("[SUCCESS] Your DCSR is faster and correct.");

    BenchmarkResult {
        config: config.clone(),
        dcsr_avg_time_ms,
        naive_avg_time_ms,
        speedup,
    }
}

// ----------------------------------------------------------------------------
// SECTION 4: Main Entry Point (for test mode)
// ----------------------------------------------------------------------------

#[cfg(test)]
fn main() {
    // Hardcoded config path for test mode
    let config_path = "tests/benchmark_config.json";

    let suite = if Path::new(config_path).exists() {
        println!("Loading benchmark configurations from: {}", config_path);
        BenchmarkSuite::from_json(config_path).expect("Failed to load benchmark config")
    } else {
        // Use default configuration
        println!("Config file not found, using default configuration.");
        BenchmarkSuite::from_default()
    };

    println!("\n========================================");
    println!("  DCSR Benchmark Suite");
    println!("  Configurations: {}", suite.configs.len());
    println!("========================================\n");

    let mut results = Vec::new();

    for (i, config) in suite.configs.iter().enumerate() {
        println!("\n[Config {}/{}]", i + 1, suite.configs.len());
        let result = run_single_benchmark(config);
        results.push(result);
    }

    // Print summary table
    println!("\n========================================");
    println!("  Summary Table");
    println!("========================================\n");
    BenchmarkResult::print_table_header();
    for result in &results {
        result.print_table_row();
    }
    println!("\n========================================");

    // Save results to JSON
    let output_path = "tests/benchmark_results.json";
    if let Err(e) = fs::write(output_path, serde_json::to_string_pretty(&results).unwrap()) {
        eprintln!("Failed to save results: {}", e);
    } else {
        println!("Results saved to: {}", output_path);
    }
}

// ----------------------------------------------------------------------------
// SECTION 5: Unit Test (Backward Compatibility)
// ----------------------------------------------------------------------------

#[test]
fn benchmark_and_verify_update_performance() {
    let config = BenchmarkConfig::default();
    let result = run_single_benchmark(&config);

    assert!(result.speedup > 1.0, "DCSR should be faster than naive CSR");
}

// Test to run the benchmark suite
#[test]
fn run_benchmark() {
    main();
}
