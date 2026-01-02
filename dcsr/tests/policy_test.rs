use dcsr::{MemPolicy, VanillaLogPolicy};

#[test]
fn test_vanilla_log_policy_new() {
    let policy = VanillaLogPolicy::new();
    assert_eq!(policy.num_nodes(), 0);
    assert_eq!(policy.total_size(), 0);
    assert_eq!(policy.total_capacity(), 0);
}

#[test]
fn test_vanilla_log_policy_with_size() {
    let policy = VanillaLogPolicy::with_size(5);
    assert_eq!(policy.num_nodes(), 5);
    assert_eq!(policy.total_size(), 0);
    assert_eq!(policy.total_capacity(), 0);
}

#[test]
fn test_vanilla_log_policy_init() {
    let mut policy = VanillaLogPolicy::new();
    let sizes = vec![2, 3, 0, 4];

    policy.init(&sizes);

    assert_eq!(policy.num_nodes(), 4);
    assert_eq!(policy.total_size(), 9);
    assert_eq!(policy.total_capacity(), 9);

    assert_eq!(policy.get_node_offset(0), Some(0));
    assert_eq!(policy.get_node_size(0), Some(2));

    assert_eq!(policy.get_node_offset(1), Some(2));
    assert_eq!(policy.get_node_size(1), Some(3));

    assert_eq!(policy.get_node_offset(2), Some(5));
    assert_eq!(policy.get_node_size(2), Some(0));

    assert_eq!(policy.get_node_offset(3), Some(5));
    assert_eq!(policy.get_node_size(3), Some(4));
}

#[test]
fn test_vanilla_log_policy_get_node_offset_out_of_bounds() {
    let mut policy = VanillaLogPolicy::new();
    policy.init(&[2, 3, 4]);

    assert_eq!(policy.get_node_offset(0), Some(0));
    assert_eq!(policy.get_node_offset(2), Some(5));
    assert_eq!(policy.get_node_offset(3), None);
    assert_eq!(policy.get_node_offset(10), None);
}

#[test]
fn test_vanilla_log_policy_get_node_size_out_of_bounds() {
    let mut policy = VanillaLogPolicy::new();
    policy.init(&[2, 3, 4]);

    assert_eq!(policy.get_node_size(0), Some(2));
    assert_eq!(policy.get_node_size(2), Some(4));
    assert_eq!(policy.get_node_size(3), None);
    assert_eq!(policy.get_node_size(10), None);
}

#[test]
fn test_vanilla_log_policy_realloc_same_nodes() {
    let mut policy = VanillaLogPolicy::new();
    policy.init(&[2, 3, 4]);

    let updates = vec![(0, 3), (1, 5)];
    policy.realloc(3, &updates);

    assert_eq!(policy.num_nodes(), 3);
    assert_eq!(policy.total_size(), 12);

    assert_eq!(policy.get_node_offset(0), Some(9));
    assert_eq!(policy.get_node_size(0), Some(3));
    assert_eq!(policy.get_node_offset(1), Some(12));
    assert_eq!(policy.get_node_size(1), Some(5));
}

#[test]
fn test_vanilla_log_policy_realloc_add_nodes() {
    let mut policy = VanillaLogPolicy::new();
    policy.init(&[2, 3]);

    let updates = vec![(0, 4), (2, 5), (3, 2)];
    policy.realloc(4, &updates);

    assert_eq!(policy.num_nodes(), 4);

    assert_eq!(policy.total_size(), 14);

    assert_eq!(policy.get_node_offset(0), Some(5));
    assert_eq!(policy.get_node_size(0), Some(4));

    assert_eq!(policy.get_node_offset(1), Some(2));
    assert_eq!(policy.get_node_size(1), Some(3));

    assert_eq!(policy.get_node_offset(2), Some(9));
    assert_eq!(policy.get_node_size(2), Some(5));
    assert_eq!(policy.get_node_offset(3), Some(14));
    assert_eq!(policy.get_node_size(3), Some(2));
}

#[test]
fn test_vanilla_log_policy_compact_no_gaps() {
    let mut policy = VanillaLogPolicy::new();
    policy.init(&[2, 3, 4]);

    assert_eq!(policy.total_size(), 9);
    assert_eq!(policy.total_capacity(), 9);

    assert_eq!(policy.get_node_offset(0), Some(0));
    assert_eq!(policy.get_node_offset(1), Some(2));
    assert_eq!(policy.get_node_offset(2), Some(5));
}

#[test]
fn test_vanilla_log_policy_compact_with_gaps() {
    let mut policy = VanillaLogPolicy::new();
    policy.init(&[2, 3, 4]);

    let updates = vec![(0, 5)];
    policy.realloc(3, &updates);

    assert_eq!(policy.total_size(), 12);
    assert_eq!(policy.total_capacity(), 14);

    assert_eq!(policy.get_node_offset(0), Some(9));
    assert_eq!(policy.get_node_size(0), Some(5));
    assert_eq!(policy.get_node_offset(1), Some(2));
    assert_eq!(policy.get_node_size(1), Some(3));
    assert_eq!(policy.get_node_offset(2), Some(5));
    assert_eq!(policy.get_node_size(2), Some(4));
}

#[test]
fn test_vanilla_log_policy_compact_empty_nodes() {
    let mut policy = VanillaLogPolicy::new();
    policy.init(&[0, 2, 0, 3, 0]);

    assert_eq!(policy.get_node_offset(0), Some(0));
    assert_eq!(policy.get_node_size(0), Some(0));
    assert_eq!(policy.get_node_offset(1), Some(0));
    assert_eq!(policy.get_node_size(1), Some(2));
    assert_eq!(policy.get_node_offset(2), Some(2));
    assert_eq!(policy.get_node_size(2), Some(0));
    assert_eq!(policy.get_node_offset(3), Some(2));
    assert_eq!(policy.get_node_size(3), Some(3));
    assert_eq!(policy.get_node_offset(4), Some(5));
    assert_eq!(policy.get_node_size(4), Some(0));
}

#[test]
fn test_vanilla_log_policy_relocation_ops() {
    let mut policy = VanillaLogPolicy::new();
    policy.init(&[2, 3, 4]);

    let updates = vec![(0, 5)];
    policy.realloc(3, &updates);

    // After realloc, node 0 is moved to the end
    assert_eq!(policy.get_node_offset(0), Some(9));
    assert_eq!(policy.get_node_size(0), Some(5));
}

#[test]
fn test_vanilla_log_policy_empty_init() {
    let mut policy = VanillaLogPolicy::new();
    policy.init(&[]);

    assert_eq!(policy.num_nodes(), 0);
    assert_eq!(policy.total_size(), 0);
    assert_eq!(policy.total_capacity(), 0);
}

#[test]
fn test_vanilla_log_policy_single_node() {
    let mut policy = VanillaLogPolicy::new();
    policy.init(&[10]);

    assert_eq!(policy.num_nodes(), 1);
    assert_eq!(policy.total_size(), 10);
    assert_eq!(policy.get_node_offset(0), Some(0));
    assert_eq!(policy.get_node_size(0), Some(10));
}

#[test]
fn test_vanilla_log_policy_multiple_reallocs() {
    let mut policy = VanillaLogPolicy::new();
    policy.init(&[2, 3]);

    let updates1 = vec![(0, 4), (2, 5)];
    policy.realloc(3, &updates1);

    let updates2 = vec![(1, 6), (3, 3)];
    policy.realloc(4, &updates2);

    assert_eq!(policy.num_nodes(), 4);
    assert_eq!(policy.total_size(), 18);
}