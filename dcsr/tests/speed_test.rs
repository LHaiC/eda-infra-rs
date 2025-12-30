use dcsr::{DynamicCSR, MemPolicy};
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
            num_nodes: 100_000,
            avg_degree: 10,
            super_node_percent: 0.05,
            super_node_degree: 200,
            update_rounds: 5,
            batch_size: 500,
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
    pub dcsr_commit_avg_ms: f64,
    pub dcsr_get_ptr_avg_ms: f64,
    pub dcsr_total_avg_ms: f64,
    pub naive_commit_avg_ms: f64,
    pub naive_get_ptr_avg_ms: f64,
    pub naive_total_avg_ms: f64,
    pub speedup_commit: f64,
    pub speedup_get_ptr: f64,
    pub speedup_total: f64,
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
            self.dcsr_total_avg_ms,
            self.naive_total_avg_ms,
            self.speedup_total,
        );
    }

    pub fn print_table_header() {
        println!(
            "{:>10} | {:>8} | {:>8} | {:>6} | {:>6} | {:>6} | {:>10} | {:>10} | {:>6}",
            "Nodes", "Batch", "Rounds", " Super%", "AvgDeg", "SupDeg",
            "DCSR(ms)", "Naive(ms)", "Speedup"
        );
        println!("{}", "-".repeat(97));
    }
}

// ----------------------------------------------------------------------------
// SECTION 1: FFI Verification Helpers
// ----------------------------------------------------------------------------

extern "C" {
    fn dcsr_test_verify_sum(
        num_nodes: u32,
        d_data: *const i32,
        d_start: *const usize,
        d_size: *const usize,
        results: *mut i32,
        count: u32
    );
    fn naive_csr_test_verify_sum(
        d_indptr: *const u32,
        d_data: *const i32,
        results: *mut i32,
        count: u32,
    );
}

// Rust wrappers for the FFI calls
fn verify_dcsr(dcsr: &mut DynamicCSR<i32>, count: u32) -> Vec<i32> {
    let mut results = vec![0i32; count as usize];
    let device = Device::CUDA(0);
    let num_nodes = dcsr.policy().num_nodes() as u32;
    let d_data = dcsr.data_ptr(device);
    let (d_start, d_size) = dcsr.topology_ptrs(device);
    unsafe {
        dcsr_test_verify_sum(num_nodes, d_data, d_start, d_size, results.as_mut_ptr(), count);
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
// SECTION 2: The Baseline - Naive CSR implemented with UVec
// ----------------------------------------------------------------------------

struct NaiveCsr<T: UniversalCopy> {
    // --- CPU-side "Logical View" ---
    adj: Vec<Vec<T>>,

    // --- GPU-side "Physical View" (Flat CSR) ---
    pub indptr: UVec<u32>,
    pub data: UVec<T>,

    device: Device,
}

impl<T: UniversalCopy + Clone> NaiveCsr<T> {
    pub fn new(num_nodes: u32, device: Device) -> Self {
        let _ctx = device.get_context();

        let adj = vec![Vec::new(); num_nodes as usize];
        let indptr = UVec::from(vec![0u32; (num_nodes + 1) as usize]);

        Self {
            adj,
            indptr,
            data: UVec::with_capacity(0, device),
            device,
        }
    }

    pub fn append(&mut self, node_id: usize, val: T) {
        self.adj[node_id].push(val);
    }

    pub fn commit(&mut self) {
        let _ctx = self.device.get_context();

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

        self.indptr = UVec::from(indptr_vec);
        self.data = UVec::from(data_vec);
    }

    pub fn get_pointers(&self) -> (*const u32, *const T) {
        let indptr_ptr = self.indptr.as_uptr(self.device);
        let data_ptr = self.data.as_uptr(self.device);
        (indptr_ptr, data_ptr)
    }
}

// ----------------------------------------------------------------------------
// SECTION 2: Single Benchmark Run
// ----------------------------------------------------------------------------

fn run_single_benchmark(config: &BenchmarkConfig) -> BenchmarkResult {
    let device = Device::CUDA(0);

    // Data Prep
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

    let mut template_adj: Vec<Vec<i32>> = Vec::with_capacity(config.num_nodes as usize);
    for i in 0..config.num_nodes {
        if super_nodes.contains(&i) {
            template_adj.push(super_data.clone());
        } else {
            template_adj.push(normal_data.clone());
        }
    }

    // Benchmarking Loop
    let mut dcsr_commit_total = std::time::Duration::new(0, 0);
    let mut dcsr_get_ptr_total = std::time::Duration::new(0, 0);
    let mut naive_commit_total = std::time::Duration::new(0, 0);
    let mut naive_get_ptr_total = std::time::Duration::new(0, 0);

    for iter in 0..config.iterations {

            // Context Reset - DCSR

            let mut dcsr_graph = DynamicCSR::<i32>::new();

    

            for (i, neighbors) in template_adj.iter().enumerate() {

                for &val in neighbors {

                    dcsr_graph.append(i, val);

                }

            }

            dcsr_graph.commit();

        // Context Reset - Naive
        let mut naive_graph = NaiveCsr::<i32>::new(config.num_nodes, device);
        for (i, neighbors) in template_adj.iter().enumerate() {
            for &val in neighbors {
                naive_graph.append(i, val);
            }
        }
        naive_graph.commit();

        // Initial verification
        if iter == 0 {
            let dcsr_res = verify_dcsr(&mut dcsr_graph, config.num_nodes);
            let naive_res = verify_naive_csr(&naive_graph, config.num_nodes);
            assert_eq!(dcsr_res, naive_res, "Mismatch after setup");
        }

        // Round Loop
        for r in 0..config.update_rounds {
            let nodes_to_update: Vec<usize> = (0..config.batch_size)
                .map(|_| rng.gen_range(0..config.num_nodes as usize))
                .collect();

            // Benchmark Dcsr - Commit
            let start = Instant::now();
            for &node_id in &nodes_to_update {
                dcsr_graph.append(node_id, 1i32);
            }
            dcsr_graph.commit();
            dcsr_commit_total += start.elapsed();

            // Benchmark Dcsr - Get Pointer
            let start = Instant::now();
            let _data_ptr = dcsr_graph.data_ptr(device);
            let (_start_ptr, _size_ptr) = dcsr_graph.topology_ptrs(device);
            dcsr_get_ptr_total += start.elapsed();

            // Benchmark Naive CSR - Commit
            let start = Instant::now();
            for &node_id in &nodes_to_update {
                naive_graph.append(node_id, 1i32);
            }
            naive_graph.commit();
            naive_commit_total += start.elapsed();

            // Benchmark Naive CSR - Get Pointer
            let start = Instant::now();
            let _ = naive_graph.get_pointers();
            naive_get_ptr_total += start.elapsed();

            // Intermittent verification (disabled for now)
            if iter == 0 && (r + 1) % (config.update_rounds / 5).max(1) == 0 {
                let dcsr_res = verify_dcsr(&mut dcsr_graph, config.num_nodes);
                let naive_res = verify_naive_csr(&naive_graph, config.num_nodes);
                assert_eq!(dcsr_res, naive_res, "Mismatch at round {}", r + 1);
            }
        }
    }

    // Statistics
    let total_ops = (config.iterations * config.update_rounds) as f64;

    let dcsr_commit_avg_ms = dcsr_commit_total.as_secs_f64() * 1000.0 / total_ops;
    let dcsr_get_ptr_avg_ms = dcsr_get_ptr_total.as_secs_f64() * 1000.0 / total_ops;
    let dcsr_total_avg_ms = dcsr_commit_avg_ms + dcsr_get_ptr_avg_ms;

    let naive_commit_avg_ms = naive_commit_total.as_secs_f64() * 1000.0 / total_ops;
    let naive_get_ptr_avg_ms = naive_get_ptr_total.as_secs_f64() * 1000.0 / total_ops;
    let naive_total_avg_ms = naive_commit_avg_ms + naive_get_ptr_avg_ms;

    let speedup_commit = naive_commit_avg_ms / dcsr_commit_avg_ms;
    let speedup_get_ptr = naive_get_ptr_avg_ms / dcsr_get_ptr_avg_ms;
    let speedup_total = naive_total_avg_ms / dcsr_total_avg_ms;

    BenchmarkResult {
        config: config.clone(),
        dcsr_commit_avg_ms,
        dcsr_get_ptr_avg_ms,
        dcsr_total_avg_ms,
        naive_commit_avg_ms,
        naive_get_ptr_avg_ms,
        naive_total_avg_ms,
        speedup_commit,
        speedup_get_ptr,
        speedup_total,
    }
}

// ----------------------------------------------------------------------------
// SECTION 3: Main Entry Point
// ----------------------------------------------------------------------------

#[cfg(test)]
fn main() {
    let config_path = "tests/benchmark_config_small.json";

    let suite = if Path::new(config_path).exists() {
        BenchmarkSuite::from_json(config_path).expect("Failed to load benchmark config")
    } else {
        BenchmarkSuite::from_default()
    };

    let mut results = Vec::new();

    for config in suite.configs.iter() {
        println!("\nRunning benchmark with config: {:?}", config);
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
    let output_path = "tests/speed_test_results.json";
    if let Err(e) = fs::write(output_path, serde_json::to_string_pretty(&results).unwrap()) {
        eprintln!("Failed to save results: {}", e);
    } else {
        println!("Results saved to: {}", output_path);
    }
}

// ----------------------------------------------------------------------------
// SECTION 4: Unit Tests
// ----------------------------------------------------------------------------

#[test]
fn run_benchmark_suite() {
    main();
}