use ulib::{UniversalCopy, Device, UVec, AsUPtr, AsUPtrMut};
use std::ops::{ Deref, DerefMut, Index, IndexMut };
use std::collections::BTreeMap;
use crate::policy::MemPolicy;

pub struct FlatMem<T: UniversalCopy> {
    data: UVec<T>,
}

impl<T: UniversalCopy> FlatMem<T> {
    #[inline]
    pub fn new() -> Self {
        Self { data: UVec::with_capacity(0, Device::CPU) }
    }

    #[inline]
    pub fn with_device(device: Device) -> Self {
        Self { data: UVec::with_capacity(0, device) }
    }

    #[inline]
    pub fn with_capacity(capacity: usize, device: Device) -> Self {
        Self { data: UVec::with_capacity(capacity, device) }
    }

    #[inline]
    pub fn from_uvec(data: UVec<T>) -> Self {
        Self { data }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    #[inline]
    pub fn as_uvec(&self) -> &UVec<T> {
        &self.data
    }

    #[inline]
    pub fn as_uvec_mut(&mut self) -> &mut UVec<T> {
        &mut self.data
    }

    #[inline]
    pub unsafe fn resize_uninit_preserve(&mut self, len: usize, device: Device) {
        self.data.resize_uninit_preserve(len, device);
    }

    #[inline]
    pub unsafe fn resize_uninit_nopreserve(&mut self, len: usize, device: Device) {
        self.data.resize_uninit_nopreserve(len, device);
    }

    #[inline]
    pub fn resize_fill_all(&mut self, len: usize, value: T, device: Device) {
        let old_len = self.data.len();
        if len > old_len {
            self.data.reserve(len - old_len, device);
            unsafe { self.data.set_len(len); }
            self.data.fill_len(value, len, device);
        } else {
            unsafe { self.data.set_len(len); }
        }
    }

    #[inline]
    pub fn fill_from_buffer<P: MemPolicy>(
        &mut self,
        buffer: &BTreeMap<usize, Vec<T>>,
        policy: &P,
    ) {
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

            for i in 0..size.min(data.len()) {
                slice[offset + i] = data[i];
            }
        }
    }

    #[inline]
    pub fn fill_from_vecvec<P: MemPolicy>(
        &mut self,
        vecvec: &Vec<Vec<T>>,
        policy: &P,
    ) {
        let slice = self.data.as_mut();

        for (node_id, data) in vecvec.iter().enumerate() {
            let offset = match policy.get_node_offset(node_id) {
                Some(offset) => offset,
                None => continue,
            };

            let size = match policy.get_node_size(node_id) {
                Some(size) => size,
                None => continue,
            };

            for i in 0..size.min(data.len()) {
                slice[offset + i] = data[i];
            }
        }
    }
    
    #[inline]
    pub fn clear(&mut self) {
        unsafe { self.data.set_len(0); }
    }

    #[inline]
    pub fn into_uvec(self) -> UVec<T> {
        self.data
    }
}

impl<T: UniversalCopy> Default for FlatMem<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: UniversalCopy + Clone> Clone for FlatMem<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self { data: self.data.clone() }
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
