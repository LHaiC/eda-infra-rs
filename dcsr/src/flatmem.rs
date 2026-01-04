use ulib::{UniversalCopy, Device, UVec, AsUPtr, AsUPtrMut};
use std::ops::{ Deref, DerefMut, Index, IndexMut };
use std::collections::BTreeMap;
use crate::policy::MemPolicy;

/// Trait for flat storage that can hold data in a contiguous buffer
pub trait FlatStorage<T: UniversalCopy>: AsRef<[T]> + AsMut<[T]> {
    /// Get the length of the storage
    fn len(&self) -> usize;

    /// Check if the storage is empty
    fn is_empty(&self) -> bool;

    /// Get the capacity of the storage
    fn capacity(&self) -> usize;

    /// Resize the storage to the given length, preserving existing data
    unsafe fn resize_uninit_preserve(&mut self, len: usize, device: Device);

    /// Resize the storage to the given length, without preserving data
    unsafe fn resize_uninit_nopreserve(&mut self, len: usize, device: Device);

    /// Get a raw pointer to the data for the specified device
    fn as_uptr(&self, device: Device) -> *const T;

    /// Get a mutable raw pointer to the data for the specified device
    fn as_mut_uptr(&mut self, device: Device) -> *mut T;

    /// Copy dirty ranges to GPU (CUDA-only)
    #[cfg(feature = "cuda")]
    unsafe fn copy_dirty_ranges_to_gpu(&mut self, device: Device, dirty_ranges: &BTreeMap<usize, usize>);

    /// Get memory usage in bytes
    fn mem_usage(&self) -> usize;

    /// Initialize storage from data using policy
    fn init<P: MemPolicy>(&mut self, data: &Vec<Vec<T>>, policy: &P);

    /// Initialize storage from csr items
    fn init_from_csr(&mut self, items: &[T]);

    /// Fill storage from buffer using policy
    fn fill_from_buffer<P: MemPolicy>(&mut self, buffer: &BTreeMap<usize, Vec<T>>, policy: &P);

    /// Compact storage using old node starts and buffer
    fn compact<P: MemPolicy>(&mut self, old_node_starts: Vec<usize>, buffer: &BTreeMap<usize, Vec<T>>, policy: &P);
}

pub struct FlatMem<T: UniversalCopy> {
    data: UVec<T>,
}

impl<T: UniversalCopy> Default for FlatMem<T> {
    #[inline]
    fn default() -> Self {
        Self { data: UVec::with_capacity(0, Device::CPU) }
    }
}

impl<T: UniversalCopy + Clone> Clone for FlatMem<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<T: UniversalCopy> Deref for FlatMem<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data.as_ref()
    }
}

impl<T: UniversalCopy> DerefMut for FlatMem<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut()
    }
}

impl<T: UniversalCopy> AsRef<[T]> for FlatMem<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.data.as_ref()
    }
}

impl<T: UniversalCopy> AsMut<[T]> for FlatMem<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
    }
}

impl<T: UniversalCopy, I> Index<I> for FlatMem<T>
where
    [T]: Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    #[inline]
    fn index(&self, i: I) -> &Self::Output {
        self.data.index(i)
    }
}

impl<T: UniversalCopy, I> IndexMut<I> for FlatMem<T>
where
    [T]: IndexMut<I>,
{
    #[inline]
    fn index_mut(&mut self, i: I) -> &mut Self::Output {
        self.data.index_mut(i)
    }
}

impl<'i, T: UniversalCopy> IntoIterator for &'i FlatMem<T> {
    type Item = &'i T;
    type IntoIter = <&'i [T] as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_ref().into_iter()
    }
}

impl<T: UniversalCopy> AsUPtr<T> for FlatMem<T> {
    #[inline]
    fn as_uptr(&self, device: Device) -> *const T {
        self.data.as_uptr(device)
    }
}

impl<T: UniversalCopy> AsUPtrMut<T> for FlatMem<T> {
    #[inline]
    fn as_mut_uptr(&mut self, device: Device) -> *mut T {
        self.data.as_mut_uptr(device)
    }
}

impl<T: UniversalCopy> FlatStorage<T> for FlatMem<T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.data.capacity()
    }

    #[inline]
    unsafe fn resize_uninit_preserve(&mut self, len: usize, device: Device) {
        self.data.resize_uninit_preserve(len, device);
    }

    #[inline]
    unsafe fn resize_uninit_nopreserve(&mut self, len: usize, device: Device) {
        self.data.resize_uninit_nopreserve(len, device);
    }

    #[inline]
    fn as_uptr(&self, device: Device) -> *const T {
        self.data.as_uptr(device)
    }

    #[inline]
    fn as_mut_uptr(&mut self, device: Device) -> *mut T {
        self.data.as_mut_uptr(device)
    }

    #[cfg(feature = "cuda")]
    unsafe fn copy_dirty_ranges_to_gpu(&mut self, device: Device, dirty_ranges: &BTreeMap<usize, usize>) {
        self.data.copy_multi_ranges_to_device(Device::CPU, device, dirty_ranges);
        self.data.mark_valid(device);
    }

    #[inline]
    fn mem_usage(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<T>()
    }

    #[inline]
    fn init<P: MemPolicy>(&mut self, data: &Vec<Vec<T>>, policy: &P) {
        let slice = self.data.as_mut();

        for (node_id, node_data) in data.iter().enumerate() {
            let offset = match policy.get_node_offset(node_id) {
                Some(offset) => offset,
                None => continue,
            };

            let size = match policy.get_node_size(node_id) {
                Some(size) => size,
                None => continue,
            };

            let write_len = size.min(node_data.len());
            for i in 0..write_len {
                slice[offset + i] = node_data[i];
            }
        }
    }

    #[inline]
    fn init_from_csr(&mut self, items: &[T]) {
        self.data[..items.len()].copy_from_slice(items);
    }

    #[inline]
    fn fill_from_buffer<P: MemPolicy>(&mut self, buffer: &BTreeMap<usize, Vec<T>>, policy: &P) {
        let slice = self.data.as_mut();

        for (&node_id, data) in buffer.iter() {
            let offset = match policy.get_node_offset(node_id) {
                Some(offset) => offset,
                None => continue,
            };

            let size = match policy.get_node_size(node_id) {
                Some(size) => size,
                None => continue,
            };

            let write_len = size.min(data.len());
            for i in 0..write_len {
                slice[offset + i] = data[i];
            }
        }
    }

    #[inline]
    fn compact<P: MemPolicy>(&mut self, old_node_starts: Vec<usize>, buffer: &BTreeMap<usize, Vec<T>>, policy: &P) {
        let old_data = self.data.as_ref().to_vec(); // Backup old data
        let slice = self.data.as_mut();
        let num_nodes = policy.num_nodes();

        for node_id in 0..num_nodes {
            let new_offset = match policy.get_node_offset(node_id) {
                Some(offset) => offset,
                None => continue,
            };
            let size = match policy.get_node_size(node_id) {
                Some(size) => size,
                None => continue,
            };

            if let Some(node_data) = buffer.get(&node_id) {
                // Write new data from buffer
                let write_len = size.min(node_data.len());
                for i in 0..write_len {
                    slice[new_offset + i] = node_data[i];
                }
            } else {
                // Move existing data within the buffer (zero-copy)
                let old_offset = old_node_starts[node_id];
                for i in 0..size {
                    slice[new_offset + i] = old_data[old_offset + i];
                }
            }
        }
    }
}


