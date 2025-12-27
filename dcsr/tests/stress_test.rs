use dcsr::{Dcsr, DcsrView};
use ulib::Device;

extern "C" {
    fn dcsr_test_verify_sum(view: DcsrView, results: *mut i32, count: u32);
}

fn verify_sum(dcsr: &Dcsr<i32>, count: u32) -> Vec<i32> {
    let mut results = vec![0i32; count as usize];
    let view = dcsr.view();
    unsafe {
        dcsr_test_verify_sum(view, results.as_mut_ptr(), count);
    }
    results
}

#[test]
fn integrated_stress_test() {
    println!(">>> [TEST START] Integrated DCSR Reliability & Stress Test (Rust Edition)");

    let device = Device::CUDA(0);

    // ==========================================
    // Phase 1: Basic Build
    // ==========================================
    println!(">>> Phase 1: Basic Batch Insert...");
    {
        let initial_data = vec![
            vec![1, 2],
            vec![10, 20, 30],
            vec![],
            vec![100],
            vec![],
        ];
        let mut graph = Dcsr::<i32>::new_from_data(initial_data, device).unwrap();

        graph.commit().unwrap();

        let res = verify_sum(&graph, 5);
        assert_eq!(res[0], 3, "Node 0 mismatch");
        assert_eq!(res[1], 60, "Node 1 mismatch");
        assert_eq!(res[2], 0, "Node 2 mismatch");
        assert_eq!(res[3], 100, "Node 3 mismatch");
        println!("    [PASS] Functional correctness verified.");
    }

    // ==========================================
    // Phase 2: Dynamic Updates
    // ==========================================
    println!(">>> Phase 2: Dynamic Updates...");
    {
        let initial_data = vec![
            vec![1, 2],
            vec![10, 20, 30],
            vec![],
            vec![100],
            vec![],
        ];
        let mut graph = Dcsr::<i32>::new_from_data(initial_data, device).unwrap();

        graph.append(0, &[3]).unwrap();
        graph.overwrite(1, &[5]).unwrap();
        graph.append(2, &[50, 50]).unwrap();

        graph.commit().unwrap();

        let res = verify_sum(&graph, 5);
        if res[0] != 6 { panic!("P2 Fail: Node 0 exp 6, got {}", res[0]); }
        if res[1] != 5 { panic!("P2 Fail: Node 1 exp 5, got {}", res[1]); }
        if res[2] != 100 { panic!("P2 Fail: Node 2 exp 100, got {}", res[2]); }
        println!("    [PASS] Update logic verified.");
    }

    // ==========================================
    // Phase 3: The Chunk Crosser (20MB Data)
    // ==========================================
    println!(">>> Phase 3: The Chunk Crosser...");
    const HUGE_COUNT: usize = 5_000_000;
    {
        let initial_data = vec![
            vec![1, 2],
            vec![10, 20, 30],
            vec![],
            vec![100],
            vec![],
        ];
        let mut graph = Dcsr::<i32>::new_from_data(initial_data, device).unwrap();

        let huge_data = vec![1i32; HUGE_COUNT];
        graph.overwrite(0, &huge_data).unwrap();

        graph.commit().unwrap();

        let res = verify_sum(&graph, 1);
        if res[0] != HUGE_COUNT as i32 {
            panic!("FATAL: Chunk Crossing Failed! Exp {}, got {}", HUGE_COUNT, res[0]);
        }
        println!("    [PASS] 20MB Data across chunk boundary verified.");
    }

    // ==========================================
    // Phase 4: The Fragmentation Bomb
    // ==========================================
    println!(">>> Phase 4: Fragmentation Bomb (1024 nodes, 10 rounds)...");

    const FRAG_START_NODE: u32 = 100;
    const NUM_FRAG_NODES: u32 = 1024;
    const ROUNDS: usize = 10;

    let initial_data = vec![
        vec![1, 2],
        vec![10, 20, 30],
        vec![],
        vec![100],
        vec![],
    ];
    let mut graph = Dcsr::<i32>::new_from_data(initial_data, device).unwrap();

    for r in 0..ROUNDS {
        for i in 0..NUM_FRAG_NODES {
            let nid = FRAG_START_NODE + i;
            graph.append(nid, &[1]).unwrap();
        }

        graph.commit().unwrap();

        if r == 0 || r == ROUNDS - 1 {
            println!("    ... Round {}/{} committed.", r + 1, ROUNDS);
        }
    }

    let total_nodes = FRAG_START_NODE + NUM_FRAG_NODES;
    let res = verify_sum(&graph, total_nodes);

    for i in 0..NUM_FRAG_NODES {
        let nid = (FRAG_START_NODE + i) as usize;
        if res[nid] != ROUNDS as i32 {
            panic!("FATAL: Frag Bomb Failed at Node {}! Exp {}, got {}", nid, ROUNDS, res[nid]);
        }
    }

    if res[0] != 3 {
        panic!("FATAL: Memory Corruption! Node 0 overwritten.");
    }

    println!("    [PASS] High frequency allocator stress verified.");
    println!(">>> [SUCCESS] All Systems Operational.");
}