use ulib::{Device, UniversalCopy, AsUPtr, AsUPtrMut};
use std::collections::BTreeMap;
use std::cmp::PartialEq;
use std::default::Default;
use std::fmt::Debug;
use crate::flatmem::FlatMem;
use crate::policy::{MemPolicy, VanillaLogPolicy};
use std::time::Instant;

pub struct DynamicCSR<T, P = VanillaLogPolicy>
where
    T: UniversalCopy + PartialEq + Default + Debug,
    P: MemPolicy,
{
    mem: FlatMem<T>,
    policy: P,
    pending: BTreeMap<usize, Vec<T>>,
}

impl<T: UniversalCopy + PartialEq + Default + Debug, P: MemPolicy> DynamicCSR<T, P> {
    #[inline]
    pub fn new() -> Self {
        Self {
            mem: FlatMem::new(),
            policy: P::new(),
            pending: BTreeMap::new(),
        }
    }

    #[inline]
    pub fn with_policy(policy: P) -> Self {
        Self {
            mem: FlatMem::new(),
            policy,
            pending: BTreeMap::new(),
        }
    }

    fn ensure_staging(&mut self, node_id: usize) -> &mut Vec<T> {
        if !self.pending.contains_key(&node_id) {
            let offset = self.policy.get_node_offset(node_id);
            let size = self.policy.get_node_size(node_id);

            let data = match (offset, size) {
                (Some(off), Some(len)) if len > 0 && off + len <= self.mem.len() => {
                    self.mem[off..off + len].to_vec()
                }
                _ => Vec::new(),
            };
            self.pending.insert(node_id, data);
        }
        self.pending.get_mut(&node_id).unwrap()
    }

    pub fn append(&mut self, node_id: usize, val: T) {
        let buf = self.ensure_staging(node_id);
        buf.push(val);
    }

    pub fn replace(&mut self, node_id: usize, index: usize, val: T) {
        let buf = self.ensure_staging(node_id);
        buf[index] = val;
    }

    pub fn replace_element(&mut self, node_id: usize, old_val: T, new_val: T) {
        let buf = self.ensure_staging(node_id);
        if let Some(pos) = buf.iter().position(|x| *x == old_val) {
            buf[pos] = new_val;
        }
    }

    pub fn remove(&mut self, node_id: usize, index: usize) -> T {
        let buf = self.ensure_staging(node_id);
        buf.remove(index)
    }

    pub fn remove_element(&mut self, node_id: usize, old_val: T) {
        let buf = self.ensure_staging(node_id);
        if let Some(pos) = buf.iter().position(|x| *x == old_val) {
            buf.remove(pos);
        }
    }

    pub fn erase(&mut self, node_id: usize) {
        self.pending.insert(node_id, Vec::new());
    }

    pub fn commit(&mut self) {
        if self.pending.is_empty() {
            return;
        }

        let max_node_id = *self.pending.keys().next_back().unwrap_or(&0);
        let current_num_nodes = self.policy.num_nodes();
        let new_num_nodes = std::cmp::max(current_num_nodes, max_node_id + 1) as usize;

        let updates: Vec<(usize, usize)> = self.pending.iter()
            .map(|(&id, vec)| (id, vec.len()))
            .collect();

        self.policy.realloc(new_num_nodes, &updates);

        let required_capacity = self.policy.total_capacity();
        if self.mem.len() < required_capacity {
            unsafe {
                self.mem.resize_uninit_preserve(required_capacity, Device::CPU);
            }
        }

        self.mem.fill_from_buffer(&self.pending, &self.policy);
        self.pending.clear();
    }

    #[inline]
    pub fn policy(&self) -> &P {
        &self.policy
    }

    #[inline]
    pub fn mem(&self) -> &FlatMem<T> {
        &self.mem
    }

    #[inline]
    pub fn data_ptr(&mut self, device: Device) -> *const T {
        unsafe {
            self.mem.copy_dirty_ranges_to_gpu(device, self.policy.get_dirty_ranges());
        }
        self.policy.clear_dirty_ranges();
        self.mem.as_uptr(device)
    }

    #[inline]
    pub fn data_mut_ptr(&mut self, device: Device) -> *mut T {
        unsafe {
            self.mem.copy_dirty_ranges_to_gpu(device, self.policy.get_dirty_ranges());
        }
        self.policy.clear_dirty_ranges();
        self.mem.as_mut_uptr(device)
    }

    #[inline]
    pub fn topology_ptrs(&self, device: Device) -> (*const usize, *const usize) {
        (
            self.policy.start_ptr(device),
            self.policy.size_ptr(device)
        )
    }
}