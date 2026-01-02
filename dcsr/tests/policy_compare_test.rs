use dcsr::{DynamicCSR, MemPolicy, VanillaLogPolicy, PowerOfTwoSlabPolicy};
use ulib::Device;
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
pub struct PolicyCompareResult {
    pub config: BenchmarkConfig,
    
    // VanillaLogPolicy metrics
    pub vanilla_commit_avg_ms: f64,
    pub vanilla_get_ptr_avg_ms: f64,
    pub vanilla_total_avg_ms: f64,
    pub vanilla_mem_usage_bytes: usize,
    
    // PowerOfTwoSlabPolicy metrics
    pub slab_commit_avg_ms: f64,
    pub slab_get_ptr_avg_ms: f64,
    pub slab_total_avg_ms: f64,
    pub slab_mem_usage_bytes: usize,
    
    // Speedup (VanillaLogPolicy / PowerOfTwoSlabPolicy)
    pub speedup_commit: f64,
    pub speedup_get_ptr: f64,
    pub speedup_total: f64,
    
    // Memory reduction (VanillaLogPolicy / PowerOfTwoSlabPolicy)
    pub mem_reduction_ratio: f64,
}

impl PolicyCompareResult {
    pub fn print_table_row(&self) {
        let vanilla_mem_mb = self.vanilla_mem_usage_bytes as f64 / 1024.0 / 1024.0;
        let slab_mem_mb = self.slab_mem_usage_bytes as f64 / 1024.0 / 1024.0;
        
        println!(
            "{:>10} | {:>8} | {:>8} | {:>6.1}% | {:>6} | {:>6} | {:>10.2} | {:>10.2} | {:>6.2}x | {:>8.2} | {:>8.2} | {:>6.2}x",
            self.config.num_nodes,
            self.config.batch_size,
            self.config.update_rounds,
            self.config.super_node_percent * 100.0,
            self.config.avg_degree,
            self.config.super_node_degree,
            self.vanilla_total_avg_ms,
            self.slab_total_avg_ms,
            self.speedup_total,
            vanilla_mem_mb,
            slab_mem_mb,
            self.mem_reduction_ratio,
        );
    }

    pub fn print_table_header() {
        println!(
            "{:>10} | {:>8} | {:>8} | {:>6} | {:>6} | {:>6} | {:>10} | {:>10} | {:>6} | {:>8} | {:>8} | {:>6}",
            "Nodes", "Batch", "Rounds", " Super%", "AvgDeg", "SupDeg",
            "Vanilla(ms)", "Slab(ms)", "Speedup", "Vanilla(MB)", "Slab(MB)", "MemRed"
        );
        println!("{}", "-".repeat(118));
    }
}

#[cfg(feature = "cuda")]
extern "C" {
    fn dcsr_test_verify_sum(
        num_nodes: u32,
        d_data: *const i32,
        d_start: *const usize,
        d_size: *const usize,
        results: *mut i32,
        count: u32
    );
}

// Rust wrappers for the FFI calls
#[cfg(feature = "cuda")]
fn verify_dcsr<P: MemPolicy>(dcsr: &mut DynamicCSR<i32, P>, count: u32) -> Vec<i32> {
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

// ----------------------------------------------------------------------------
// SECTION 1: Single Benchmark Run
// ----------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn run_single_benchmark<P1: MemPolicy, P2: MemPolicy>(
    config: &BenchmarkConfig,
) -> PolicyCompareResult
where
    P1: Send + Sync + 'static,
    P2: Send + Sync + 'static,
{
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

    let mut template_adj: Vec<Vec<i32>> = Vec::with_capacity(config.num_nodes as usize);
    for i in 0..config.num_nodes {
        if super_nodes.contains(&i) {
            let degree = rng.gen_range(config.super_node_degree / 2..=config.super_node_degree);
            template_adj.push((0..degree).map(|_| rng.gen::<i32>()).collect());
        } else {
            let degree = rng.gen_range(1..=config.avg_degree * 2);
            template_adj.push((0..degree).map(|_| rng.gen::<i32>()).collect());
        }
    }

    // Benchmarking Loop
    let mut vanilla_commit_total = std::time::Duration::new(0, 0);
    let mut vanilla_get_ptr_total = std::time::Duration::new(0, 0);
    let mut slab_commit_total = std::time::Duration::new(0, 0);
    let mut slab_get_ptr_total = std::time::Duration::new(0, 0);

    let mut vanilla_final_mem_usage = 0usize;
    let mut slab_final_mem_usage = 0usize;

    for iter in 0..config.iterations {
        // Context Reset - VanillaLogPolicy
        let mut vanilla_graph = DynamicCSR::<i32, P1>::new();
        vanilla_graph.init(&template_adj);

        // Context Reset - PowerOfTwoSlabPolicy
        let mut slab_graph = DynamicCSR::<i32, P2>::new();
        slab_graph.init(&template_adj);

        // Initial verification
        if iter == 0 {
            let vanilla_res = verify_dcsr(&mut vanilla_graph, config.num_nodes);
            let slab_res = verify_dcsr(&mut slab_graph, config.num_nodes);
            assert_eq!(vanilla_res, slab_res, "Mismatch after setup");
        }

        // Round Loop
        for r in 0..config.update_rounds {
            let mut operations: Vec<(usize, bool, Option<usize>, i32)> = Vec::new(); // (node_id, is_replace, replace_index, value)
            for _ in 0..config.batch_size {
                let node_id = rng.gen_range(0..config.num_nodes as usize);
                // let is_replace = rng.gen::<bool>();
                let is_replace = true;
                let val = rng.gen::<i32>();
                let replace_index = if is_replace && template_adj[node_id].len() > 0 {
                    Some(rng.gen_range(0..template_adj[node_id].len()))
                } else {
                    None
                };
                operations.push((node_id, is_replace, replace_index, val));
            }

            // Benchmark VanillaLogPolicy - Commit
            let start = Instant::now();
            for &(node_id, is_replace, replace_index, val) in &operations {
                if is_replace {
                    if let Some(idx) = replace_index {
                        vanilla_graph.replace(node_id, idx, val);
                    }
                } else {
                    vanilla_graph.append(node_id, val);
                }
            }
            vanilla_graph.commit();
            vanilla_commit_total += start.elapsed();

            // Benchmark VanillaLogPolicy - Get Pointer
            let start = Instant::now();
            let _data_ptr = vanilla_graph.data_ptr(device);
            let (_start_ptr, _size_ptr) = vanilla_graph.topology_ptrs(device);
            vanilla_get_ptr_total += start.elapsed();

            // Benchmark PowerOfTwoSlabPolicy - Commit
            let start = Instant::now();
            for &(node_id, is_replace, replace_index, val) in &operations {
                if is_replace {
                    if let Some(idx) = replace_index {
                        slab_graph.replace(node_id, idx, val);
                    }
                } else {
                    slab_graph.append(node_id, val);
                }
            }
            slab_graph.commit();
            slab_commit_total += start.elapsed();

            // Benchmark PowerOfTwoSlabPolicy - Get Pointer
            let start = Instant::now();
            let _data_ptr = slab_graph.data_ptr(device);
            let (_start_ptr, _size_ptr) = slab_graph.topology_ptrs(device);
            slab_get_ptr_total += start.elapsed();

            // Intermittent verification
            if iter == 0 && (r + 1) % (config.update_rounds / 5).max(1) == 0 {
                let vanilla_res = verify_dcsr(&mut vanilla_graph, config.num_nodes);
                let slab_res = verify_dcsr(&mut slab_graph, config.num_nodes);
                assert_eq!(vanilla_res, slab_res, "Mismatch at round {}", r + 1);
            }
        }

        // Record memory usage at the end
        vanilla_final_mem_usage = vanilla_graph.mem_usage();
        slab_final_mem_usage = slab_graph.mem_usage();
    }

    // Statistics
    let total_ops = (config.iterations * config.update_rounds) as f64;

    let vanilla_commit_avg_ms = vanilla_commit_total.as_secs_f64() * 1000.0 / total_ops;
    let vanilla_get_ptr_avg_ms = vanilla_get_ptr_total.as_secs_f64() * 1000.0 / total_ops;
    let vanilla_total_avg_ms = vanilla_commit_avg_ms + vanilla_get_ptr_avg_ms;

    let slab_commit_avg_ms = slab_commit_total.as_secs_f64() * 1000.0 / total_ops;
    let slab_get_ptr_avg_ms = slab_get_ptr_total.as_secs_f64() * 1000.0 / total_ops;
    let slab_total_avg_ms = slab_commit_avg_ms + slab_get_ptr_avg_ms;

    let speedup_commit = vanilla_commit_avg_ms / slab_commit_avg_ms;
    let speedup_get_ptr = vanilla_get_ptr_avg_ms / slab_get_ptr_avg_ms;
    let speedup_total = vanilla_total_avg_ms / slab_total_avg_ms;

    let mem_reduction_ratio = vanilla_final_mem_usage as f64 / slab_final_mem_usage as f64;

    PolicyCompareResult {
        config: config.clone(),
        vanilla_commit_avg_ms,
        vanilla_get_ptr_avg_ms,
        vanilla_total_avg_ms,
        vanilla_mem_usage_bytes: vanilla_final_mem_usage,
        slab_commit_avg_ms,
        slab_get_ptr_avg_ms,
        slab_total_avg_ms,
        slab_mem_usage_bytes: slab_final_mem_usage,
        speedup_commit,
        speedup_get_ptr,
        speedup_total,
        mem_reduction_ratio,
    }
}

// ----------------------------------------------------------------------------
// SECTION 2: Main Entry Point
// ----------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
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
        let result = run_single_benchmark::<VanillaLogPolicy, PowerOfTwoSlabPolicy>(config);
        results.push(result);
    }

    // Print summary table
    println!("\n========================================");
    println!("  Policy Comparison Summary Table");
    println!("========================================\n");
    PolicyCompareResult::print_table_header();
    for result in &results {
        result.print_table_row();
    }
    println!("\n========================================");

    // Save results to JSON
    let output_path = "tests/policy_compare_results.json";
    if let Err(e) = fs::write(output_path, serde_json::to_string_pretty(&results).unwrap()) {
        eprintln!("Failed to save results: {}", e);
    } else {
        println!("Results saved to: {}", output_path);
    }
}

// ----------------------------------------------------------------------------
// SECTION 3: Unit Tests
// ----------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[test]
fn run_policy_compare_benchmark() {
    main();
}
