//! Compressed sparse row (CSR) implementation.

use std::fmt::{self, Debug};
use ulib::UVec;
use dcsr::{DynamicCSR, MemPolicy, VanillaLogPolicy};

/// A helper type for simple 1-layer CSR.
#[derive(Debug, Default, Clone)]
pub struct VecCSR {
    /// flattened list start, analogous to `flat_net2pin_start`
    pub start: UVec<usize>,
    /// flattened list, analogous to `netpin`, indexed by [VecCSR::start]
    pub items: UVec<usize>,
}

impl VecCSR {
    /// build CSR from the mapping between items to set indices.
    pub fn from(num_sets: usize, num_items: usize, inset: &[usize]) -> VecCSR {
        assert_eq!(inset.len(), num_items);
        let mut start: Vec<usize> = vec![0; num_sets + 1];
        let mut items: Vec<usize> = vec![0; num_items];
        for s in inset {
            start[*s] += 1;
        }
        // todo: parallelizable
        for i in 1..num_sets + 1 {
            start[i] += start[i - 1];
        }
        assert_eq!(start[num_sets], num_items);
        // todo: parallelizable
        for i in (0..num_items).rev() {
            let s = inset[i];
            let pos = start[s] - 1;
            start[s] -= 1;
            items[pos] = i;
        }
        VecCSR {
            start: start.into(),
            items: items.into()
        }
    }

    /// convenient method to get an iterator of set items.
    #[inline]
    pub fn iter_set(&self, set_id: usize)
                    -> impl Iterator<Item = usize> + '_
    {
        let l = self.start[set_id];
        let r = self.start[set_id + 1];
        self.items[l..r].iter().copied()
    }
    
    /// get size of a set
    #[inline]
    pub fn len(&self, set_id: usize) -> usize {
        let l = self.start[set_id];
        let r = self.start[set_id + 1];
        r - l
    }
}

/// Dynamic CSR wrapper using DCSR internally.
/// Provides zero-copy conversion from VecCSR via `from_vec_csr`.
///
/// # Type Parameters
/// * `P` - Memory policy, defaults to VanillaLogPolicy
pub struct DVecCSR<P: MemPolicy = VanillaLogPolicy> {
    /// The internal DCSR structure (policy stores node_starts, mem stores items)
    dcsr: DynamicCSR<usize, P>,
}

impl<P: MemPolicy + Clone> Clone for DVecCSR<P> {
    fn clone(&self) -> Self {
        Self {
            dcsr: self.dcsr.clone(),
        }
    }
}

impl<P: MemPolicy> Debug for DVecCSR<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DVecCSR")
            .field("num_sets", &self.dcsr.policy().num_nodes())
            .field("num_items", &self.dcsr.policy().total_size())
            .finish()
    }
}

impl<P: MemPolicy + Default> DVecCSR<P> {
    /// Build DVecCSR from the mapping between items to set indices.
    #[inline]
    pub fn from(num_sets: usize, num_items: usize, inset: &[usize]) -> Self {
        assert_eq!(inset.len(), num_items);
        let mut start: Vec<usize> = vec![0; num_sets + 1];
        let mut items: Vec<usize> = vec![0; num_items];
        for s in inset {
            start[*s] += 1;
        }
        // todo: parallelizable
        for i in 1..num_sets + 1 {
            start[i] += start[i - 1];
        }
        assert_eq!(start[num_sets], num_items);
        // todo: parallelizable
        for i in (0..num_items).rev() {
            let s = inset[i];
            let pos = start[s] - 1;
            start[s] -= 1;
            items[pos] = i;
        }

        let dcsr = DynamicCSR::<usize, P>::from_csr(&start, &items);
        Self { dcsr }
    }
}

impl<P: MemPolicy> DVecCSR<P> {
    /// Get the number of sets (nets/cells).
    #[inline]
    pub fn num_sets(&self) -> usize {
        self.dcsr.policy().num_nodes()
    }

    /// Get the total number of items (pins).
    #[inline]
    pub fn num_items(&self) -> usize {
        self.dcsr.policy().total_size()
    }

    /// Get reference to the internal DCSR for advanced operations.
    #[inline]
    pub fn dcsr(&self) -> &DynamicCSR<usize, P> {
        &self.dcsr
    }

    /// Get mutable reference to the internal DCSR for advanced operations.
    #[inline]
    pub fn dcsr_mut(&mut self) -> &mut DynamicCSR<usize, P> {
        &mut self.dcsr
    }

    /// Commit pending changes in the DCSR.
    #[inline]
    pub fn commit(&mut self) {
        self.dcsr.commit();
    }

    /// Get an iterator over items in a set.
    #[inline]
    pub fn iter_set(&self, set_id: usize) -> impl Iterator<Item = &usize> + '_ {
        let offset = self.dcsr.policy().get_node_offset(set_id).unwrap_or(0);
        let size = self.dcsr.policy().get_node_size(set_id).unwrap_or(0);
        self.dcsr.mem()[offset..offset + size].iter()
    }

    /// Get the size of a set.
    #[inline]
    pub fn len(&self, set_id: usize) -> usize {
        self.dcsr.policy().get_node_size(set_id).unwrap_or(0)
    }

    /// Get the start offset of a set.
    #[inline]
    pub fn offset(&self, set_id: usize) -> usize {
        self.dcsr.policy().get_node_offset(set_id).unwrap_or(0)
    }

    /// Get reference to the start array.
    #[inline]
    pub fn start(&self) -> &[usize] {
        self.dcsr.policy().starts(ulib::Device::CPU)
    }

    /// Get reference to the items array.
    #[inline]
    pub fn items(&self) -> &[usize] {
        self.dcsr.mem().as_ref()
    }

    /// Mutable reference to items
    #[inline]
    pub fn items_mut(&mut self) -> &mut [usize] {
        self.dcsr.mem_mut().as_mut()
    }

    /// Get the internal policy for advanced operations.
    #[inline]
    pub fn policy(&self) -> &P {
        self.dcsr.policy()
    }

    /// Get mutable reference to the internal policy.
    #[inline]
    pub fn policy_mut(&mut self) -> &mut P {
        self.dcsr.policy_mut()
    }

    /// Get the internal memory storage.
    #[inline]
    pub fn mem(&self) -> &dcsr::FlatMem<usize> {
        self.dcsr.mem()
    }

    /// Get mutable reference to the internal memory storage.
    #[inline]
    pub fn mem_mut(&mut self) -> &mut dcsr::FlatMem<usize> {
        self.dcsr.mem_mut()
    }
}

impl Default for DVecCSR<VanillaLogPolicy> {
    fn default() -> Self {
        Self {
            dcsr: DynamicCSR::<usize, VanillaLogPolicy>::new(),
        }
    }
}
