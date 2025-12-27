//! # DCSR: Dynamic Compressed Sparse Row library
//!
//! Zero-overhead Rust wrapper around C++ DCSR implementation.
//! Rust maintains `Vec<Vec<T>>` as Truth, C++ is pure I/O engine.

use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ffi::c_void;

use ulib::{UniversalCopy, Device};

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
mod dcsr_ffi {
    use super::*;

    include!(concat!(env!("OUT_DIR"), "/uccbind/dcsr.rs"));

    extern "C" {
        pub fn dcsr_create_from_bulk(
            num_nodes: u32,
            element_size: usize,
            ptrs: *const *const c_void,
            sizes: *const u32,
            count: usize,
        ) -> *mut c_void;

        pub fn dcsr_update_sparse(
            handle: *mut c_void,
            ids: *const u32,
            ptrs: *const *const c_void,
            sizes: *const u32,
            count: usize,
        );

        pub fn dcsr_destroy(handle: *mut c_void);
        pub fn dcsr_get_view(handle: *mut c_void) -> RawDcsrView;
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RawDcsrView {
    pub node_chunk_id: *const u32,
    pub node_chunk_offset: *const u32,
    pub node_data_size_bytes: *const u32,
    pub device_chunks: *const *const u8,
    pub num_nodes: u32,
    pub element_size: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DcsrView<'a> {
    pub node_chunk_id: *const u32,
    pub node_chunk_offset: *const u32,
    pub node_data_size_bytes: *const u32,
    pub device_chunks: *const *const u8,
    pub num_nodes: u32,
    pub element_size: u32,
    _phantom: PhantomData<&'a ()>,
}

unsafe impl<'a> Send for DcsrView<'a> {}
unsafe impl<'a> Sync for DcsrView<'a> {}

pub struct Dcsr<T> {
    handle: *mut c_void,
    device: Device,

    adj: Vec<Vec<T>>,
    dirty_nodes: BTreeSet<u32>,

    ffi_ids: Vec<u32>,
    ffi_ptrs: Vec<*const c_void>,
    ffi_sizes: Vec<u32>,

    cached_view: RawDcsrView,

    _phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for Dcsr<T> {}
unsafe impl<T: Sync> Sync for Dcsr<T> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcsrError {
    NodeNotFound(u32),
    IndexOutOfBounds { node_id: u32, index: usize },
    InvalidRange { node_id: u32, start: usize, end: usize },
    InitFailed,
    CommitFailed,
}

impl std::fmt::Display for DcsrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DcsrError::NodeNotFound(id) => write!(f, "Node {} not found", id),
            DcsrError::IndexOutOfBounds { node_id, index } => {
                write!(f, "Index {} out of bounds for node {}", index, node_id)
            }
            DcsrError::InvalidRange { node_id, start, end } => {
                write!(f, "Invalid range {}..{} for node {}", start, end, node_id)
            }
            DcsrError::InitFailed => write!(f, "DCSR initialization failed"),
            DcsrError::CommitFailed => write!(f, "DCSR commit failed"),
        }
    }
}

impl std::error::Error for DcsrError {}

pub type DcsrResult<T> = Result<T, DcsrError>;

impl<T: UniversalCopy + Clone + PartialEq + Sync> Dcsr<T> {

    pub fn new_from_data(data: Vec<Vec<T>>, device: Device) -> DcsrResult<Self> {
        let _ctx = device.get_context();
        let num_nodes = data.len() as u32;
        let element_size = size_of::<T>() as u32;

        let mut ptrs = Vec::with_capacity(data.len());
        let mut sizes = Vec::with_capacity(data.len());

        for v in &data {
            if v.is_empty() {
                ptrs.push(std::ptr::null());
                sizes.push(0);
            } else {
                ptrs.push(v.as_ptr() as *const c_void);
                sizes.push((v.len() * size_of::<T>()) as u32);
            }
        }

        let handle = unsafe {
            dcsr_ffi::dcsr_create_from_bulk(
                num_nodes,
                element_size as usize,
                ptrs.as_ptr(),
                sizes.as_ptr(),
                data.len()
            )
        };

        if handle.is_null() {
            return Err(DcsrError::InitFailed);
        }

        let cached_view = unsafe { dcsr_ffi::dcsr_get_view(handle) };

        Ok(Self {
            handle,
            device,
            adj: data,
            dirty_nodes: BTreeSet::new(),
            ffi_ids: Vec::new(),
            ffi_ptrs: Vec::new(),
            ffi_sizes: Vec::new(),
            cached_view,
            _phantom: PhantomData,
        })
    }

    fn ensure_node(&mut self, node_id: u32) {
        if node_id as usize >= self.adj.len() {
            self.adj.resize_with((node_id + 1) as usize, || Vec::new());
        }
    }

    pub fn append(&mut self, node_id: u32, data: &[T]) -> DcsrResult<()> {
        self.ensure_node(node_id);
        self.adj[node_id as usize].extend_from_slice(data);
        self.dirty_nodes.insert(node_id);
        Ok(())
    }

    pub fn append_owned(&mut self, node_id: u32, data: Vec<T>) -> DcsrResult<()> {
        self.ensure_node(node_id);
        self.adj[node_id as usize].extend(data);
        self.dirty_nodes.insert(node_id);
        Ok(())
    }

    pub fn insert(&mut self, node_id: u32, index: usize, val: T) -> DcsrResult<()> {
        self.ensure_node(node_id);
        let vec = &mut self.adj[node_id as usize];
        if index > vec.len() {
            return Err(DcsrError::IndexOutOfBounds { node_id, index });
        }
        vec.insert(index, val);
        self.dirty_nodes.insert(node_id);
        Ok(())
    }

    pub fn replace_range<R>(&mut self, node_id: u32, range: R, replace_with: &[T]) -> DcsrResult<()>
    where
        R: std::ops::RangeBounds<usize> + std::fmt::Debug,
    {
        self.ensure_node(node_id);
        let vec = &mut self.adj[node_id as usize];
        let len = vec.len();

        let start = match range.start_bound() {
            std::ops::Bound::Included(&s) => s,
            std::ops::Bound::Excluded(&s) => s + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(&e) => e + 1,
            std::ops::Bound::Excluded(&e) => e,
            std::ops::Bound::Unbounded => len,
        };

        if start > len || end > len || start > end {
            return Err(DcsrError::InvalidRange { node_id, start, end });
        }

        vec.splice(start..end, replace_with.iter().cloned());
        self.dirty_nodes.insert(node_id);
        Ok(())
    }

    pub fn overwrite(&mut self, node_id: u32, data: &[T]) -> DcsrResult<()> {
        self.ensure_node(node_id);
        let vec = &mut self.adj[node_id as usize];
        vec.clear();
        vec.extend_from_slice(data);
        self.dirty_nodes.insert(node_id);
        Ok(())
    }

    pub fn overwrite_owned(&mut self, node_id: u32, data: Vec<T>) -> DcsrResult<()> {
        self.ensure_node(node_id);
        let vec = &mut self.adj[node_id as usize];
        vec.clear();
        vec.extend_from_slice(&data);
        self.dirty_nodes.insert(node_id);
        Ok(())
    }

    pub fn remove_at(&mut self, node_id: u32, index: usize) -> DcsrResult<T> {
        if node_id as usize >= self.adj.len() {
            return Err(DcsrError::NodeNotFound(node_id));
        }
        let vec = &mut self.adj[node_id as usize];
        if index >= vec.len() {
            return Err(DcsrError::IndexOutOfBounds { node_id, index });
        }
        let val = vec.remove(index);
        self.dirty_nodes.insert(node_id);
        Ok(val)
    }

    pub fn remove_element(&mut self, node_id: u32, element: &T) -> DcsrResult<bool> {
        if node_id as usize >= self.adj.len() {
            return Err(DcsrError::NodeNotFound(node_id));
        }
        let vec = &mut self.adj[node_id as usize];
        if let Some(pos) = vec.iter().position(|x| x == element) {
            vec.remove(pos);
            self.dirty_nodes.insert(node_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn clear_node(&mut self, node_id: u32) -> DcsrResult<()> {
        if node_id as usize >= self.adj.len() {
            return Err(DcsrError::NodeNotFound(node_id));
        }
        self.adj[node_id as usize].clear();
        self.dirty_nodes.insert(node_id);
        Ok(())
    }

    pub fn get_node(&self, node_id: u32) -> DcsrResult<&[T]> {
        if node_id as usize >= self.adj.len() {
            return Err(DcsrError::NodeNotFound(node_id));
        }
        Ok(&self.adj[node_id as usize])
    }

    pub fn get_node_mut(&mut self, node_id: u32) -> DcsrResult<&mut Vec<T>> {
        if node_id as usize >= self.adj.len() {
            return Err(DcsrError::NodeNotFound(node_id));
        }
        self.dirty_nodes.insert(node_id);
        Ok(&mut self.adj[node_id as usize])
    }

    pub fn num_nodes(&self) -> u32 {
        self.adj.len() as u32
    }

    pub fn is_dirty(&self, node_id: u32) -> bool {
        self.dirty_nodes.contains(&node_id)
    }

    pub fn commit(&mut self) -> DcsrResult<()> {
        if self.dirty_nodes.is_empty() {
            return Ok(());
        }

        let _ctx = self.device.get_context();

        let dirty_count = self.dirty_nodes.len();
        self.ffi_ids.clear();
        self.ffi_ptrs.clear();
        self.ffi_sizes.clear();
        self.ffi_ids.reserve(dirty_count);
        self.ffi_ptrs.reserve(dirty_count);
        self.ffi_sizes.reserve(dirty_count);

        for &u in &self.dirty_nodes {
            let vec = &self.adj[u as usize];
            self.ffi_ids.push(u);

            if vec.is_empty() {
                self.ffi_ptrs.push(std::ptr::null());
                self.ffi_sizes.push(0);
            } else {
                self.ffi_ptrs.push(vec.as_ptr() as *const c_void);
                self.ffi_sizes.push((vec.len() * size_of::<T>()) as u32);
            }
        }

        unsafe {
            dcsr_ffi::dcsr_update_sparse(
                self.handle,
                self.ffi_ids.as_ptr(),
                self.ffi_ptrs.as_ptr(),
                self.ffi_sizes.as_ptr(),
                self.ffi_ids.len()
            );

            self.cached_view = dcsr_ffi::dcsr_get_view(self.handle);
        }

        self.device.synchronize();
        self.dirty_nodes.clear();
        Ok(())
    }

    #[inline(always)]
    pub fn view(&self) -> DcsrView<'_> {
        debug_assert_eq!(
            std::mem::size_of::<DcsrView<'static>>(),
            std::mem::size_of::<RawDcsrView>(),
            "Rust DcsrView layout mismatch!"
        );

        unsafe { std::mem::transmute(self.cached_view) }
    }
}

impl<T> Drop for Dcsr<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ctx = self.device.get_context();
            unsafe {
                dcsr_ffi::dcsr_destroy(self.handle);
            }
        }
    }
}