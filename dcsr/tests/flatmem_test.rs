use dcsr::FlatMem;
use ulib::Device;

#[test]
fn test_flatmem_new() {
    let mem: FlatMem<i32> = FlatMem::new();
    assert_eq!(mem.len(), 0);
    assert!(mem.is_empty());
}

#[test]
fn test_flatmem_with_device() {
    let mem: FlatMem<i32> = FlatMem::with_device(Device::CPU);
    assert_eq!(mem.len(), 0);
    assert!(mem.is_empty());
}

#[test]
fn test_flatmem_with_capacity() {
    let mem: FlatMem<i32> = FlatMem::with_capacity(10, Device::CPU);
    assert_eq!(mem.len(), 0);
    assert!(mem.capacity() >= 10);
}

#[test]
fn test_flatmem_resize() {
    let mut mem: FlatMem<i32> = FlatMem::with_capacity(5, Device::CPU);

    mem.resize_fill_all(3, 42, Device::CPU);
    assert_eq!(mem.len(), 3);
    assert_eq!(mem[0], 42);
    assert_eq!(mem[1], 42);
    assert_eq!(mem[2], 42);

    mem.resize_fill_all(5, 99, Device::CPU);
    assert_eq!(mem.len(), 5);
    assert_eq!(mem[0], 99);
    assert_eq!(mem[1], 99);
    assert_eq!(mem[2], 99);
    assert_eq!(mem[3], 99);
    assert_eq!(mem[4], 99);

    // 缩容
    mem.resize_fill_all(2, 0, Device::CPU);
    assert_eq!(mem.len(), 2);
    assert_eq!(mem[0], 99);
    assert_eq!(mem[1], 99);
}

#[test]
fn test_flatmem_clear() {
    let mut mem: FlatMem<i32> = FlatMem::with_capacity(5, Device::CPU);
    mem.resize_fill_all(3, 42, Device::CPU);
    assert_eq!(mem.len(), 3);

    mem.clear();
    assert_eq!(mem.len(), 0);
    assert!(mem.is_empty());
}

#[test]
fn test_flatmem_index() {
    let mut mem: FlatMem<i32> = FlatMem::with_capacity(5, Device::CPU);
    mem.resize_fill_all(3, 42, Device::CPU);

    assert_eq!(mem[0], 42);
    assert_eq!(mem[1], 42);
    assert_eq!(mem[2], 42);

    mem[0] = 100;
    assert_eq!(mem[0], 100);
}

#[test]
fn test_flatmem_range() {
    let mut mem: FlatMem<i32> = FlatMem::with_capacity(10, Device::CPU);
    mem.resize_fill_all(5, 0, Device::CPU);
    for i in 0..5 {
        mem[i] = i as i32 * 10;
    }

    let slice = &mem[1..4];
    assert_eq!(slice.len(), 3);
    assert_eq!(slice[0], 10);
    assert_eq!(slice[1], 20);
    assert_eq!(slice[2], 30);
}

#[test]
fn test_flatmem_deref() {
    let mut mem: FlatMem<i32> = FlatMem::with_capacity(5, Device::CPU);
    mem.resize_fill_all(3, 42, Device::CPU);

    let slice: &[i32] = &mem;
    assert_eq!(slice.len(), 3);
    assert_eq!(slice[0], 42);
}

#[test]
fn test_flatmem_iter() {
    let mut mem: FlatMem<i32> = FlatMem::with_capacity(5, Device::CPU);
    mem.resize_fill_all(3, 42, Device::CPU);
    mem[0] = 10;
    mem[1] = 20;
    mem[2] = 30;

    let values: Vec<i32> = mem.iter().copied().collect();
    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn test_flatmem_clone() {
    let mut mem: FlatMem<i32> = FlatMem::with_capacity(5, Device::CPU);
    mem.resize_fill_all(3, 42, Device::CPU);
    mem[0] = 100;

    let mem2 = mem.clone();
    assert_eq!(mem2.len(), 3);
    assert_eq!(mem2[0], 100);
    assert_eq!(mem2[1], 42);

    mem[0] = 999;
    assert_eq!(mem2[0], 100);
}

#[test]
fn test_flatmem_default() {
    let mem: FlatMem<i32> = FlatMem::default();
    assert_eq!(mem.len(), 0);
    assert!(mem.is_empty());
}

#[test]
fn test_flatmem_as_ref() {
    let mut mem: FlatMem<i32> = FlatMem::with_capacity(5, Device::CPU);
    mem.resize_fill_all(3, 42, Device::CPU);

    let slice: &[i32] = mem.as_ref();
    assert_eq!(slice.len(), 3);
}

#[test]
fn test_flatmem_as_mut() {
    let mut mem: FlatMem<i32> = FlatMem::with_capacity(5, Device::CPU);
    mem.resize_fill_all(3, 42, Device::CPU);

    let slice: &mut [i32] = mem.as_mut();
    slice[0] = 100;
    assert_eq!(mem[0], 100);
}