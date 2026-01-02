use dcsr::{DynamicCSR, MemPolicy, VanillaLogPolicy};
use ulib::Device;
use std::collections::BTreeMap;

#[test]
fn test_dynamic_csr_new() {
    let csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();
    assert_eq!(csr.policy().num_nodes(), 0);
    assert_eq!(csr.mem().len(), 0);
}

#[test]
fn test_dynamic_csr_with_policy() {
    let policy = VanillaLogPolicy::with_size(5);
    let csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::with_policy(policy);
    assert_eq!(csr.policy().num_nodes(), 5);
}

#[test]
fn test_dynamic_csr_append_single() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 10);
    csr.append(0, 20);
    csr.append(0, 30);

    csr.commit();

    assert_eq!(csr.policy().num_nodes(), 1);
    assert_eq!(csr.policy().get_node_size(0), Some(3));
    assert_eq!(csr.policy().get_node_offset(0), Some(0));
}

#[test]
fn test_dynamic_csr_append_multiple_nodes() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 1);
    csr.append(0, 2);

    csr.append(1, 10);
    csr.append(1, 20);
    csr.append(1, 30);

    csr.append(2, 100);

    csr.commit();

    assert_eq!(csr.policy().num_nodes(), 3);
    assert_eq!(csr.policy().get_node_size(0), Some(2));
    assert_eq!(csr.policy().get_node_size(1), Some(3));
    assert_eq!(csr.policy().get_node_size(2), Some(1));
}

#[test]
fn test_dynamic_csr_replace() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 10);
    csr.append(0, 20);
    csr.append(0, 30);
    csr.commit();

    csr.replace(0, 1, 99);
    csr.commit();

    assert_eq!(csr.policy().get_node_size(0), Some(3));
    let offset = csr.policy().get_node_offset(0).unwrap();
    assert_eq!(csr.mem()[offset + 1], 99);
}

#[test]
fn test_dynamic_csr_replace_element() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 10);
    csr.append(0, 20);
    csr.append(0, 30);
    csr.commit();

    csr.append(0, 10);
    csr.append(0, 20);
    csr.append(0, 30);
    csr.replace_element(0, 20, 99);
    csr.commit();

    let offset = csr.policy().get_node_offset(0).unwrap();
    assert_eq!(csr.mem()[offset], 10);
    assert_eq!(csr.mem()[offset + 1], 99);
    assert_eq!(csr.mem()[offset + 2], 30);
}

#[test]
fn test_dynamic_csr_replace_element_not_found() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 10);
    csr.append(0, 20);
    csr.commit();

    csr.replace_element(0, 99, 100);
    csr.commit();

    assert_eq!(csr.policy().get_node_size(0), Some(2));
}

#[test]
fn test_dynamic_csr_remove() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 10);
    csr.append(0, 20);
    csr.append(0, 30);
    csr.commit();

    let removed = csr.remove(0, 1);
    assert_eq!(removed, 20);
    csr.commit();

    assert_eq!(csr.policy().get_node_size(0), Some(2));
    let offset = csr.policy().get_node_offset(0).unwrap();
    assert_eq!(csr.mem()[offset], 10);
    assert_eq!(csr.mem()[offset + 1], 30);
}

#[test]
fn test_dynamic_csr_remove_element() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 10);
    csr.append(0, 20);
    csr.append(0, 30);
    csr.commit();

    csr.remove_element(0, 20);
    csr.commit();

    assert_eq!(csr.policy().get_node_size(0), Some(2));
    let offset = csr.policy().get_node_offset(0).unwrap();
    assert_eq!(csr.mem()[offset], 10);
    assert_eq!(csr.mem()[offset + 1], 30);
}

#[test]
fn test_dynamic_csr_erase() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 10);
    csr.append(0, 20);
    csr.append(0, 30);
    csr.commit();

    csr.erase(0);
    csr.commit();

    assert_eq!(csr.policy().get_node_size(0), Some(0));
}

#[test]
fn test_dynamic_csr_empty_commit() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.commit();

    assert_eq!(csr.policy().num_nodes(), 0);
}

#[test]
fn test_dynamic_csr_multiple_commits() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 1);
    csr.append(0, 2);
    csr.commit();

    csr.append(0, 3);
    csr.append(0, 4);
    csr.commit();

    csr.append(1, 10);
    csr.append(1, 20);
    csr.commit();

    assert_eq!(csr.policy().num_nodes(), 2);
    assert_eq!(csr.policy().get_node_size(0), Some(4));
    assert_eq!(csr.policy().get_node_size(1), Some(2));
}

#[test]
fn test_dynamic_csr_non_sequential_nodes() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(2, 20);
    csr.append(5, 50);
    csr.append(5, 51);
    csr.commit();

    assert_eq!(csr.policy().num_nodes(), 6);
    assert_eq!(csr.policy().get_node_size(0), Some(0));
    assert_eq!(csr.policy().get_node_size(1), Some(0));
    assert_eq!(csr.policy().get_node_size(2), Some(1));
    assert_eq!(csr.policy().get_node_size(3), Some(0));
    assert_eq!(csr.policy().get_node_size(4), Some(0));
    assert_eq!(csr.policy().get_node_size(5), Some(2));
}

#[test]
fn test_dynamic_csr_data_ptr() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 10);
    csr.append(0, 20);
    csr.commit();

    let ptr = csr.data_ptr(Device::CPU);
    assert!(!ptr.is_null());
}

#[test]
fn test_dynamic_csr_data_mut_ptr() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 10);
    csr.append(0, 20);
    csr.commit();

    let ptr = csr.data_mut_ptr(Device::CPU);
    assert!(!ptr.is_null());
}

#[test]
fn test_dynamic_csr_topology_ptrs() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 10);
    csr.append(1, 20);
    csr.commit();

    let (start_ptr, size_ptr): (*const usize, *const usize) = csr.topology_ptrs(Device::CPU);
    assert!(!start_ptr.is_null());
    assert!(!size_ptr.is_null());
}

#[test]
fn test_dynamic_csr_complex_workflow() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 1);
    csr.append(0, 2);
    csr.append(1, 0);
    csr.append(1, 2);
    csr.append(2, 0);
    csr.append(2, 1);
    csr.commit();

    csr.append(0, 3);
    csr.remove_element(0, 1);
    csr.commit();

    csr.erase(1);
    csr.commit();

    csr.append(3, 10);
    csr.append(3, 20);
    csr.commit();

    assert_eq!(csr.policy().num_nodes(), 4);
    assert_eq!(csr.policy().get_node_size(0), Some(2));
    assert_eq!(csr.policy().get_node_size(1), Some(0));
    assert_eq!(csr.policy().get_node_size(2), Some(2));
    assert_eq!(csr.policy().get_node_size(3), Some(2));
}

#[test]
fn test_dynamic_csr_with_custom_policy() {
    struct TestPolicy;

    impl MemPolicy for TestPolicy {
        fn new() -> Self { TestPolicy }
        fn with_size(_num_nodes: usize) -> Self { TestPolicy }
        fn init(&mut self, _sizes: &[usize]) {}
        fn realloc(&mut self, _new_num_nodes: usize, _updates: &[(usize, usize)]) {}
        fn compact(&mut self, new_num_nodes: usize, updates: &[(usize, usize)]) {
            let _ = new_num_nodes;
            let _ = updates;
        }
        fn get_node_offset(&self, _node_id: usize) -> Option<usize> { Some(0) }
        fn get_node_size(&self, _node_id: usize) -> Option<usize> { Some(0) }
        fn num_nodes(&self) -> usize { 0 }
        fn total_size(&self) -> usize { 0 }
        fn total_capacity(&self) -> usize { 0 }
        fn start_ptr(&self, _device: Device) -> *const usize { std::ptr::null() }
        fn size_ptr(&self, _device: Device) -> *const usize { std::ptr::null() }
        fn start_mut_ptr(&mut self, _device: Device) -> *mut usize { std::ptr::null_mut() }
        fn size_mut_ptr(&mut self, _device: Device) -> *mut usize { std::ptr::null_mut() }
        fn get_node_starts(&self) -> Vec<usize> {
            Vec::new()
        }
        fn get_dirty_ranges(&self) -> &BTreeMap<usize, usize> {
            static EMPTY_MAP: BTreeMap<usize, usize> = BTreeMap::new();
            &EMPTY_MAP
        }
        fn clear_dirty_ranges(&mut self) {}
        fn mem_usage(&self) -> usize { 0 }
    }

    let csr: DynamicCSR<i32, TestPolicy> = DynamicCSR::new();
    assert_eq!(csr.policy().num_nodes(), 0);
}

#[test]
fn test_dynamic_csr_replace_on_empty_node() {
    let mut csr: DynamicCSR<i32, VanillaLogPolicy> = DynamicCSR::new();

    csr.append(0, 10);
    csr.commit();

    csr.replace(0, 0, 20);
    csr.commit();

    assert_eq!(csr.policy().get_node_size(0), Some(1));
    let offset = csr.policy().get_node_offset(0).unwrap();
    assert_eq!(csr.mem()[offset], 20);
}