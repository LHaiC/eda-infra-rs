use dcsr::{Dcsr, DcsrView};
use ulib::{Device, UVec, UniversalCopy, AsUPtr};
use std::time::Instant;

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
// SECTION 3: The Benchmark Test
// ----------------------------------------------------------------------------

#[test]
fn benchmark_and_verify_update_performance() {
    println!("\n>>> [BENCHMARK] DCSR vs Naive CSR Update Performance");

    let device = Device::CUDA(0);

    const NUM_NODES: u32 = 3999174;
    const AVG_DEGREE: usize = 10;
    const SUPER_NODE_PERCENT: f32 = 0.05;
    const SUPER_NODE_DEGREE: usize = 200;
    const UPDATE_ROUNDS: usize = 10;
    const BATCH_SIZE: usize = 2000;

    // 1. Setup
    let mut naive_graph = NaiveCsr::<i32>::new(NUM_NODES, device);

    use rand::prelude::*;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(42);

    let num_super_nodes = (NUM_NODES as f32 * SUPER_NODE_PERCENT) as u32;

    // Generate a list of super nodes
    let mut super_nodes: std::collections::HashSet<u32> = std::collections::HashSet::new();
    while super_nodes.len() < num_super_nodes as usize {
        super_nodes.insert(rng.gen_range(0..NUM_NODES));
    }

    let normal_data = vec![1i32; AVG_DEGREE];
    let super_data = vec![1i32; SUPER_NODE_DEGREE];

    // Build initial data for Dcsr
    let mut initial_data: Vec<Vec<i32>> = Vec::with_capacity(NUM_NODES as usize);
    for i in 0..NUM_NODES {
        if super_nodes.contains(&i) {
            initial_data.push(super_data.clone());
            naive_graph.append(i, &super_data);
        } else {
            initial_data.push(normal_data.clone());
            naive_graph.append(i, &normal_data);
        }
    }

    println!("    Creating Dcsr from initial data...");
    let start = Instant::now();
    let mut dcsr_graph = Dcsr::<i32>::new_from_data(initial_data, device).unwrap();
    println!("    DCSR creation took: {:?}", start.elapsed());

    println!("    Creating Naive CSR from initial data...");
    let start = Instant::now();
    naive_graph.commit();
    println!("    Naive CSR creation took: {:?}", start.elapsed());
    
    println!("    Verifying setup...");
    let dcsr_res = verify_dcsr(&dcsr_graph, NUM_NODES);
    let naive_res = verify_naive_csr(&naive_graph, NUM_NODES);
    assert_eq!(dcsr_res, naive_res, "Mismatch after setup");
    println!("    Setup complete and verified.");

    // 2. Benchmarking Loop
    let mut dcsr_total_time = std::time::Duration::new(0, 0);
    let mut naive_total_time = std::time::Duration::new(0, 0);

    let update_data = vec![5i32; AVG_DEGREE];
    let super_update_data = vec![10i32; SUPER_NODE_DEGREE];
    for r in 0..UPDATE_ROUNDS {
        let nodes_to_update: Vec<u32> = (0..BATCH_SIZE)
            .map(|_| rng.gen_range(0..NUM_NODES))
            .collect();

        // --- Benchmark Dcsr (Batch Update) ---
        let start = Instant::now();
        for &node_id in &nodes_to_update {
            if super_nodes.contains(&node_id) {
                dcsr_graph.append(node_id, &super_update_data).unwrap();
            } else {
                dcsr_graph.append(node_id, &update_data).unwrap();
            }
        }
        dcsr_graph.commit().unwrap();
        dcsr_total_time += start.elapsed();

        // --- Benchmark Naive CSR (One by one, then commit) ---
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

        // --- Intermittent Verification ---
        if (r + 1) % (UPDATE_ROUNDS / 5) == 0 {
            let dcsr_res = verify_dcsr(&dcsr_graph, NUM_NODES);
            let naive_res = verify_naive_csr(&naive_graph, NUM_NODES);
            assert_eq!(dcsr_res, naive_res, "Mismatch at round {}", r + 1);
            println!("    Verification pass at round {}/{}", r + 1, UPDATE_ROUNDS);
        }
    }
    
    // 3. Final Verification and Reporting
    println!("    Final verification...");
    let dcsr_res = verify_dcsr(&dcsr_graph, NUM_NODES);
    let naive_res = verify_naive_csr(&naive_graph, NUM_NODES);
    assert_eq!(dcsr_res, naive_res, "Final results mismatched!");

    println!("\n----- BENCHMARK RESULTS -----");
    println!("Total updates: {}", UPDATE_ROUNDS);
    println!("DCSR (Log-Structured) Total Time: {:>10.2?}", dcsr_total_time);
    println!("Naive CSR (Realloc)   Total Time: {:>10.2?}", naive_total_time);
    println!("-----------------------------\n");
    
    assert!(
        dcsr_total_time < naive_total_time, 
        "FAILURE: Your DCSR is slower than the brute-force baseline. Go back to the drawing board."
    );
    println!("[SUCCESS] Your DCSR is faster and correct.");
}