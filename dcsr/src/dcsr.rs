use ulib::{Device, UniversalCopy};
use std::collections::BTreeMap;
use std::cmp::PartialEq;
use std::default::Default;
use std::fmt::Debug;
use crate::flatmem::{FlatStorage, FlatMem};
use crate::policy::{MemPolicy, VanillaLogPolicy};

pub struct DynamicCSR<T, P = VanillaLogPolicy, S = FlatMem<T>>
where
    T: UniversalCopy + PartialEq + Default + Debug,
    P: MemPolicy,
    S: FlatStorage<T>,
{
    mem: S,
    policy: P,
    pending: BTreeMap<usize, Vec<T>>,
}

impl<T: UniversalCopy + PartialEq + Default + Debug, P: MemPolicy, S: FlatStorage<T> + Default> DynamicCSR<T, P, S> {
    #[inline]
    pub fn new() -> Self {
        Self {
            mem: S::default(),
            policy: P::new(),
            pending: BTreeMap::new(),
        }
    }

    #[inline]
    pub fn with_size(num_nodes: usize) -> Self {
        Self {
            mem: S::default(),
            policy: P::with_size(num_nodes),
            pending: BTreeMap::new(),
        }
    }

    #[inline]
    pub fn with_policy(policy: P) -> Self {
        Self {
            mem: S::default(),
            policy,
            pending: BTreeMap::new(),
        }
    }

    #[inline]
    pub fn from_csr(starts: &[usize], items: &[T]) -> Self {
        let mut dcsr = Self::new();
        dcsr.init_from_csr(starts, items);
        dcsr
    }

    fn ensure_staging(&mut self, node_id: usize) -> &mut Vec<T> {
        if !self.pending.contains_key(&node_id) {
            let offset = self.policy.get_node_offset(node_id);
            let size = self.policy.get_node_size(node_id);

            let data = match (offset, size) {
                (Some(off), Some(len)) if len > 0 && off + len <= self.mem.len() => {
                    self.mem.as_ref()[off..off + len].to_vec()
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

    pub fn init(&mut self, data: &Vec<Vec<T>>) {
        let sizes: Vec<usize> = data.iter().map(|vec| vec.len()).collect();
        self.policy.init(&sizes);
        let required_capacity = self.policy.total_capacity();
        if self.mem.len() < required_capacity {
            unsafe {
                self.mem.resize_uninit_preserve(required_capacity, Device::CPU);
            }
        }
        self.mem.init(data, &self.policy);
        self.pending.clear();
    }

    pub fn init_from_csr(&mut self, starts: &[usize], items: &[T]) {
        let num_nodes = starts.len() - 1;
        self.policy.init_from_flat(num_nodes, starts);
        let required_capacity = self.policy.total_capacity();
        if self.mem.len() < required_capacity {
            unsafe {
                self.mem.resize_uninit_preserve(required_capacity, Device::CPU);
            }
        }
        self.mem.init_from_csr(items);
        self.pending.clear();
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

        if self.need_compact() {
            // compact stragegy
            let old_node_starts = self.policy.get_node_starts();

            self.policy.compact(new_num_nodes, &updates);

            let required_capacity = self.policy.total_capacity();
            if self.mem.len() < required_capacity {
                unsafe {
                    self.mem.resize_uninit_preserve(required_capacity, Device::CPU);
                }
            }

            self.mem.compact(old_node_starts, &self.pending, &self.policy);
        } else {
            // realloc strategy
            self.policy.realloc(new_num_nodes, &updates);

            let required_capacity = self.policy.total_capacity();
            if self.mem.len() < required_capacity {
                unsafe {
                    self.mem.resize_uninit_preserve(required_capacity, Device::CPU);
                }
            }

            self.mem.fill_from_buffer(&self.pending, &self.policy);
        }

        self.pending.clear();
    }

    #[inline]
    pub fn policy(&self) -> &P {
        &self.policy
    }

    #[inline]
    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }

    #[inline]
    pub fn mem(&self) -> &S {
        &self.mem
    }

    #[inline]
    pub fn mem_mut(&mut self) -> &mut S {
        &mut self.mem
    }

    #[inline]
    fn sync_data_dirty_ranges(&mut self, device: Device) {
        #[cfg(feature = "cuda")]
        {
            if matches!(device, Device::CUDA(_)) {
                unsafe {
                    self.mem.copy_dirty_ranges_to_gpu(device, self.policy.get_dirty_ranges());
                }
            }
        }
        self.policy.clear_dirty_ranges();
    }

    #[inline]
    pub fn data_ptr(&mut self, device: Device) -> *const T {
        self.sync_data_dirty_ranges(device);
        self.mem.as_uptr(device)
    }

    #[inline]
    pub fn data_mut_ptr(&mut self, device: Device) -> *mut T {
        self.sync_data_dirty_ranges(device);
        if device == Device::CPU {
            self.policy.mark_all_dirty();
        }
        self.mem.as_mut_uptr(device)
    }

    #[inline]
    pub fn topology_ptrs(&self, device: Device) -> (*const usize, *const usize) {
        (
            self.policy.start_ptr(device),
            self.policy.size_ptr(device)
        )
    }

    pub fn mem_usage(&self) -> usize {
        let mut total = 0;
        total += self.mem.capacity() * std::mem::size_of::<T>();
        total += self.policy.mem_usage();
        for vec in self.pending.values() {
            total += vec.capacity() * std::mem::size_of::<T>();
        }
        total
    }

    fn need_compact(&self) -> bool {
        let total_size = self.policy.total_size();
        let total_capacity = self.policy.total_capacity();
        total_capacity > total_size * 2
    }
}

impl<T: UniversalCopy + PartialEq + Default + Debug + Clone, P: MemPolicy + Clone, S: FlatStorage<T> + Clone> Clone for DynamicCSR<T, P, S> {
    fn clone(&self) -> Self {
        Self {
            mem: self.mem.clone(),
            policy: self.policy.clone(),
            pending: self.pending.clone(),
        }
    }
}