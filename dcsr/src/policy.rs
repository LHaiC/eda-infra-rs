use ulib::{Device, UVec, AsUPtr, AsUPtrMut};
use std::collections::BTreeMap;

pub trait MemPolicy: Send + Sync {
    fn new() -> Self;

    fn with_size(num_nodes: usize) -> Self;

    fn init(&mut self, sizes: &[usize]);

    fn realloc(&mut self, new_num_nodes: usize, updates: &[(usize, usize)]);

    fn compact(&mut self, new_num_nodes: usize, updates: &[(usize, usize)]);

    fn get_node_offset(&self, node_id: usize) -> Option<usize>;

    fn get_node_size(&self, node_id: usize) -> Option<usize>;

    fn num_nodes(&self) -> usize;

    fn total_size(&self) -> usize;

    fn total_capacity(&self) -> usize;

    fn start_ptr(&self, device: Device) -> *const usize;

    fn size_ptr(&self, device: Device) -> *const usize;

    fn start_mut_ptr(&mut self, device: Device) -> *mut usize;

    fn size_mut_ptr(&mut self, device: Device) -> *mut usize;

    fn get_node_starts(&self) -> Vec<usize>;

    fn get_dirty_ranges(&self) -> &BTreeMap<usize, usize>;

    fn clear_dirty_ranges(&mut self);

    fn mem_usage(&self) -> usize;
}

pub struct VanillaLogPolicy {
    node_start: UVec<usize>,
    node_size: UVec<usize>,
    num_nodes: usize,
    total_size: usize,
    total_capacity: usize,
    dirty_ranges: BTreeMap<usize, usize>,
}

impl MemPolicy for VanillaLogPolicy {
    fn new() -> Self {
        Self {
            node_start: UVec::with_capacity(0, Device::CPU),
            node_size: UVec::with_capacity(0, Device::CPU),
            num_nodes: 0,
            total_size: 0,
            total_capacity: 0,
            dirty_ranges: BTreeMap::new(),
        }
    }

    fn with_size(num_nodes: usize) -> Self {
        let device = Device::CPU;
        let mut node_start = UVec::with_capacity(num_nodes + 1, device);
        let mut node_size = UVec::with_capacity(num_nodes + 1, device);
        node_start.fill(usize::MAX, device);
        node_size.fill(0, device);
        Self {
            node_start,
            node_size,
            num_nodes,
            total_size: 0,
            total_capacity: 0,
            dirty_ranges: BTreeMap::new(),
        }
    }

    fn init(&mut self, sizes: &[usize]) {
        let device = Device::CPU;
        let num_nodes = sizes.len();
        let mut offset = 0usize;
        unsafe {
            self.node_start.resize_uninit_nopreserve(num_nodes + 1, device);
            self.node_size.resize_uninit_nopreserve(num_nodes + 1, device);
        }
        for (i, &size) in sizes.iter().enumerate() {
            self.node_start[i] = offset;
            self.node_size[i] = size;
            offset += size;
        }
        self.num_nodes = num_nodes;
        self.total_size = offset;
        self.total_capacity = offset;
        self.dirty_ranges.clear();
        self.add_dirty_range(0, offset);
    }

    fn realloc(&mut self, new_num_nodes: usize, updates: &[(usize, usize)]) {
        let device = Device::CPU;
        let old_num_nodes = self.num_nodes;
        let mut offset = self.total_capacity;
        let old_offset = offset;
        let mut total_size = self.total_size;

        unsafe {
            self.node_start.resize_uninit_preserve(new_num_nodes + 1, device);
            self.node_size.resize_uninit_preserve(new_num_nodes + 1, device);
        }

        for i in old_num_nodes..new_num_nodes {
            self.node_start[i] = usize::MAX;
            self.node_size[i] = 0;
        }

        for &(node_id, new_size) in updates.iter() {
            if node_id < old_num_nodes {
                total_size = total_size.wrapping_sub(self.node_size[node_id]).wrapping_add(new_size);
            } else {
                total_size += new_size;
            }
            self.node_start[node_id] = offset;
            self.node_size[node_id] = new_size;
            offset += new_size;
        }
        self.add_dirty_range(old_offset, offset);
        self.num_nodes = new_num_nodes;
        self.total_size = total_size;
        self.total_capacity = offset;
    }

    fn compact(&mut self, new_num_nodes: usize, updates: &[(usize, usize)]) {
        let device = Device::CPU;
        let mut offset = 0usize;

        unsafe {
            self.node_start.resize_uninit_preserve(new_num_nodes + 1, device);
            self.node_size.resize_uninit_preserve(new_num_nodes + 1, device);
        }

        for &(node_id, new_size) in updates.iter() {
            self.node_size[node_id] = new_size;
        }

        for i in 0..new_num_nodes {
            self.node_start[i] = offset;
            offset += self.node_size[i];
        }

        self.num_nodes = new_num_nodes;
        self.total_size = offset;
        self.total_capacity = offset;
        self.dirty_ranges.clear();
        self.add_dirty_range(0, offset);
    }

    fn get_node_offset(&self, node_id: usize) -> Option<usize> {
        if node_id >= self.num_nodes {
            return None;
        }
        Some(self.node_start[node_id])
    }

    fn get_node_size(&self, node_id: usize) -> Option<usize> {
        if node_id >= self.num_nodes {
            return None;
        }
        Some(self.node_size[node_id])
    }

    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    fn total_size(&self) -> usize {
        self.total_size
    }

    fn total_capacity(&self) -> usize {
        self.total_capacity
    }

    fn start_ptr(&self, device: Device) -> *const usize {
        self.node_start.as_uptr(device)
    }

    fn size_ptr(&self, device: Device) -> *const usize {
        self.node_size.as_uptr(device)
    }

    fn start_mut_ptr(&mut self, device: Device) -> *mut usize {
        self.node_start.as_mut_uptr(device)
    }

    fn size_mut_ptr(&mut self, device: Device) -> *mut usize {
        self.node_size.as_mut_uptr(device)
    }

    fn get_node_starts(&self) -> Vec<usize> {
        self.node_start.to_vec()
    }

    fn get_dirty_ranges(&self) -> &BTreeMap<usize, usize> {
        &self.dirty_ranges
    }

    fn clear_dirty_ranges(&mut self) {
        self.dirty_ranges.clear();
    }

    fn mem_usage(&self) -> usize {
        (self.node_start.capacity() + self.node_size.capacity()) * std::mem::size_of::<usize>()
    }
}

impl VanillaLogPolicy {
    /// Add a dirty range with greedy merging.
    fn add_dirty_range(&mut self, start: usize, end: usize) {
        if start >= end {
            return;
        }

        let mut new_start = start;
        let mut new_end = end;
        let mut to_remove = Vec::new();

        // Check if we can merge with the previous range (ends at or overlaps start)
        if let Some((&prev_start, &prev_end)) = self.dirty_ranges.range(..=start).next_back() {
            if prev_end >= start {
                // Adjacent or overlapping: merge
                new_start = prev_start;
                new_end = new_end.max(prev_end);
                to_remove.push(prev_start);
            }
        }

        // Check and merge with all subsequent ranges that overlap or are adjacent
        for (&next_start, &next_end) in self.dirty_ranges.range(start..) {
            if next_start <= new_end {
                new_end = new_end.max(next_end);
                to_remove.push(next_start);
            } else {
                break;
            }
        }

        // Remove merged ranges and insert the new merged range
        for key in to_remove {
            self.dirty_ranges.remove(&key);
        }
        self.dirty_ranges.insert(new_start, new_end);
    }

    pub fn update_offsets_direct(&mut self, new_offsets: &[usize], new_total_size: usize) {
        for (i, &offset) in new_offsets.iter().enumerate() {
            if i < self.num_nodes {
                self.node_start[i] = offset;
            }
        }
        self.total_capacity = new_total_size;
    }
}

pub struct PowerOfTwoSlabPolicy {
    node_start: UVec<usize>,
    node_size: UVec<usize>,
    num_nodes: usize,

    // here use Vec<Vec<usize>> as free lists for each size class
    // in future we can optimize this with a more efficient data structure or 
    // store free lists in FlatMem itself
    collection_order_threshold: usize, // minimum order to trigger collection, 2 as default
    num_size_classes: usize, // number of size classes, 16 as default
    free_lists: Vec<Vec<usize>>, // free lists per size class

    total_size: usize,
    total_capacity: usize,
    dirty_ranges: BTreeMap<usize, usize>,
}

impl PowerOfTwoSlabPolicy {
    #[inline]
    fn alloc_class(&self, size: usize) -> Option<usize> {
        if size == 0 {
            return None;
        }
        let min_cap = 1 << self.collection_order_threshold;
        if size <= min_cap {
            return Some(0);
        }
        let order = size.next_power_of_two().ilog2() as usize;
        let class = order.saturating_sub(self.collection_order_threshold);
        if class >= self.num_size_classes {
            return None;
        }
        Some(class)
    }

    #[inline]
    fn free_class(&self, size: usize) -> Option<usize> {
        let min_cap = 1 << self.collection_order_threshold;
        if size < min_cap {
            return None;
        }
        let order = size.ilog2() as usize;
        let class = order.saturating_sub(self.collection_order_threshold);
        if class >= self.num_size_classes {
            return Some(self.num_size_classes - 1);
        }
        Some(class)
    }

    #[inline]
    fn push_free(&mut self, offset: usize, size: usize) {
        if let Some(class) = self.free_class(size) {
            self.free_lists[class].push(offset);
        }
    }

    #[inline]
    fn pop_free(&mut self, size: usize) -> Option<usize> {
        if let Some(class) = self.alloc_class(size) {
            if let Some(offset) = self.free_lists[class].pop() {
                return Some(offset);
            }
        }
        None
    }
}

impl MemPolicy for PowerOfTwoSlabPolicy {
    fn new() -> Self {
        Self {
            node_start: UVec::with_capacity(0, ulib::Device::CPU),
            node_size: UVec::with_capacity(0, ulib::Device::CPU),
            num_nodes: 0,
            collection_order_threshold: 2,
            num_size_classes: 16,
            free_lists: vec![Vec::new(); 16],
            total_size: 0,
            total_capacity: 0,
            dirty_ranges: BTreeMap::new(),
        }
    }

    fn with_size(num_nodes: usize) -> Self {
        let device = ulib::Device::CPU;
        let mut node_start = UVec::with_capacity(num_nodes + 1, device);
        let mut node_size = UVec::with_capacity(num_nodes + 1, device);
        node_start.fill(usize::MAX, device);
        node_size.fill(0, device);
        Self {
            node_start,
            node_size,
            num_nodes,
            collection_order_threshold: 2,
            num_size_classes: 16,
            free_lists: vec![Vec::new(); 16],
            total_size: 0,
            total_capacity: 0,
            dirty_ranges: BTreeMap::new(),
        }
    }

    fn init(&mut self, sizes: &[usize]) {
        let device = Device::CPU;
        let num_nodes = sizes.len();
        let mut offset = 0usize;
        unsafe {
            self.node_start.resize_uninit_nopreserve(num_nodes + 1, device);
            self.node_size.resize_uninit_nopreserve(num_nodes + 1, device);
        }
        for (i, &size) in sizes.iter().enumerate() {
            self.node_start[i] = offset;
            self.node_size[i] = size;
            offset += size;
        }
        self.num_nodes = num_nodes;
        self.total_size = offset;
        self.total_capacity = offset;
        for free_list in self.free_lists.iter_mut() {
            free_list.clear();
        }
        self.dirty_ranges.clear();
        self.add_dirty_range(0, offset);
    }

    fn realloc(&mut self, new_num_nodes: usize, updates: &[(usize, usize)]) {
        let device = Device::CPU;
        let old_num_nodes = self.num_nodes;
        let mut offset = self.total_capacity;
        let old_offset = offset;
        let mut total_size = self.total_size;
        unsafe {
            self.node_start.resize_uninit_preserve(new_num_nodes + 1, device);
            self.node_size.resize_uninit_preserve(new_num_nodes + 1, device);
        }
        for i in old_num_nodes..new_num_nodes {
            self.node_start[i] = usize::MAX;
            self.node_size[i] = 0;
        }

        // Pass 1: collect old nodes to free lists
        for &(node_id, new_size) in updates.iter() {
            if node_id < old_num_nodes {
                total_size = total_size.wrapping_sub(self.node_size[node_id]).wrapping_add(new_size);
                let old_offset = self.node_start[node_id];
                let old_size = self.node_size[node_id];
                self.push_free(old_offset, old_size);
            } else {
                total_size += new_size;
                self.node_start[node_id] = offset;
                self.node_size[node_id] = new_size;
                offset += new_size;
            }
        }

        // Pass 2: allocate from free lists or append
        for &(node_id, new_size) in updates.iter() {
            if node_id < old_num_nodes {
                if let Some(free_offset) = self.pop_free(new_size) {
                    // Allocate from free list
                    self.node_start[node_id] = free_offset;
                    self.node_size[node_id] = new_size;
                    // Read next free from the first element
                    self.add_dirty_range(free_offset, free_offset + new_size);
                    continue;
                }
                // No suitable free block, append
                self.node_start[node_id] = offset;
                self.node_size[node_id] = new_size;
                offset += new_size;
            }
        }

        self.add_dirty_range(old_offset, offset);
        self.num_nodes = new_num_nodes;
        self.total_size = total_size;
        self.total_capacity = offset;
        for free_list in self.free_lists.iter_mut() {
            free_list.clear();
        }
    }

    fn compact(&mut self, new_num_nodes: usize, updates: &[(usize, usize)]) {
        let device = Device::CPU;
        let mut offset = 0usize;

        unsafe {
            self.node_start.resize_uninit_preserve(new_num_nodes + 1, device);
            self.node_size.resize_uninit_preserve(new_num_nodes + 1, device);
        }

        for &(node_id, new_size) in updates.iter() {
            self.node_size[node_id] = new_size;
        }

        for i in 0..new_num_nodes {
            self.node_start[i] = offset;
            offset += self.node_size[i];
        }

        self.num_nodes = new_num_nodes;
        self.total_size = offset;
        self.total_capacity = offset;
        self.dirty_ranges.clear();
        self.add_dirty_range(0, offset);
    }

    fn get_node_offset(&self, node_id: usize) -> Option<usize> {
        if node_id >= self.num_nodes {
            return None;
        }
        Some(self.node_start[node_id])
    }

    fn get_node_size(&self, node_id: usize) -> Option<usize> {
        if node_id >= self.num_nodes {
            return None;
        }
        Some(self.node_size[node_id])
    }

    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    fn total_size(&self) -> usize {
        self.total_size
    }

    fn total_capacity(&self) -> usize {
        self.total_capacity
    }

    fn start_ptr(&self, device: ulib::Device) -> *const usize {
        self.node_start.as_uptr(device)
    }

    fn size_ptr(&self, device: ulib::Device) -> *const usize {
        self.node_size.as_uptr(device)
    }

    fn start_mut_ptr(&mut self, device: Device) -> *mut usize {
        self.node_start.as_mut_uptr(device)
    }

    fn size_mut_ptr(&mut self, device: Device) -> *mut usize {
        self.node_size.as_mut_uptr(device)
    }

    fn get_node_starts(&self) -> Vec<usize> {
        self.node_start.to_vec()
    }

    fn get_dirty_ranges(&self) -> &BTreeMap<usize, usize> {
        &self.dirty_ranges
    }

    fn clear_dirty_ranges(&mut self) {
        self.dirty_ranges.clear();
    }

    fn mem_usage(&self) -> usize {
        (self.node_start.capacity() + self.node_size.capacity()) * std::mem::size_of::<usize>()
    }
}

impl PowerOfTwoSlabPolicy {
    /// Add a dirty range with greedy merging.
    fn add_dirty_range(&mut self, start: usize, end: usize) {
        if start >= end {
            return;
        }

        let mut new_start = start;
        let mut new_end = end;
        let mut to_remove = Vec::new();

        // Check if we can merge with the previous range (ends at or overlaps start)
        if let Some((&prev_start, &prev_end)) = self.dirty_ranges.range(..=start).next_back() {
            if prev_end >= start {
                // Adjacent or overlapping: merge
                new_start = prev_start;
                new_end = new_end.max(prev_end);
                to_remove.push(prev_start);
            }
        }

        // Check and merge with all subsequent ranges that overlap or are adjacent
        for (&next_start, &next_end) in self.dirty_ranges.range(start..) {
            if next_start <= new_end {
                new_end = new_end.max(next_end);
                to_remove.push(next_start);
            } else {
                break;
            }
        }

        // Remove merged ranges and insert the new merged range
        for key in to_remove {
            self.dirty_ranges.remove(&key);
        }
        self.dirty_ranges.insert(new_start, new_end);
    }

    pub fn update_offsets_direct(&mut self, new_offsets: &[usize], new_total_size: usize) {
        for (i, &offset) in new_offsets.iter().enumerate() {
            if i < self.num_nodes {
                self.node_start[i] = offset;
            }
        }
        self.total_capacity = new_total_size;
    }
}