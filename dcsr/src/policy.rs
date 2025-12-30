use ulib::{Device, UVec, AsUPtr};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct RelocationOp {
    pub src: usize,
    pub dst: usize,
    pub len: usize,
}

pub trait MemPolicy: Send + Sync {
    fn new() -> Self;

    fn with_size(num_nodes: usize) -> Self;

    fn init(&mut self, sizes: &[usize]);

    fn realloc(&mut self, new_num_nodes: usize, updates: &[(usize, usize)]);

    fn get_node_offset(&self, node_id: usize) -> Option<usize>;

    fn get_node_size(&self, node_id: usize) -> Option<usize>;

    fn compact(&mut self) -> Vec<RelocationOp>;

    fn num_nodes(&self) -> usize;

    fn total_size(&self) -> usize;

    fn total_capacity(&self) -> usize;

    fn start_ptr(&self, device: Device) -> *const usize;

    fn size_ptr(&self, device: Device) -> *const usize;

    fn get_dirty_ranges(&self) -> &BTreeMap<usize, usize>;

    fn clear_dirty_ranges(&mut self);
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
        let mut node_start = UVec::with_capacity(num_nodes, device);
        let mut node_size = UVec::with_capacity(num_nodes, device);
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
            self.node_start.resize_uninit_nopreserve(num_nodes, device);
            self.node_size.resize_uninit_nopreserve(num_nodes, device);
        }
        for (i, &size) in sizes.iter().enumerate() {
            self.node_start[i] = offset;
            self.node_size[i] = size;
            offset += size;
        }
        self.num_nodes = num_nodes;
        self.total_size = offset;
        self.total_capacity = offset;
    }

    fn realloc(&mut self, new_num_nodes: usize, updates: &[(usize, usize)]) {
        let device = Device::CPU;
        let old_num_nodes = self.num_nodes;
        let mut offset = self.total_capacity;
        let old_offset = offset;
        let mut total_size = self.total_size;

        unsafe {
            self.node_start.resize_uninit_preserve(new_num_nodes, device);
            self.node_size.resize_uninit_preserve(new_num_nodes, device);
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

    fn compact(&mut self) -> Vec<RelocationOp> {
        let mut plan = Vec::with_capacity(self.num_nodes); 
        let mut write_pos = 0usize;

        for i in 0..self.num_nodes {
            let old_start = self.node_start[i];
            let size = self.node_size[i];

            if old_start != write_pos && size > 0 {
                plan.push(RelocationOp {
                    src: old_start,
                    dst: write_pos,
                    len: size,
                });
            }

            self.node_start[i] = write_pos;
            write_pos += size;
        }

        self.total_size = write_pos;
        self.total_capacity = write_pos;

        plan
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

    fn get_dirty_ranges(&self) -> &BTreeMap<usize, usize> {
        &self.dirty_ranges
    }

    fn clear_dirty_ranges(&mut self) {
        self.dirty_ranges.clear();
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
}
