//! A flattened gate-level circuit netlist database.

use std::collections::HashMap;
use std::sync::Arc;
use std::collections::HashSet;
use compact_str::CompactString;
use ulib::{UVec, Device, UniversalCopy, Zeroable};

/// types of directions: input or output.
/// 
/// note: inout is not supported yet.
/// **should be identical to `csrc/lib.h`**.
#[derive(Zeroable, Debug, PartialEq, Eq, Clone, UniversalCopy)]
#[repr(u8)]
pub enum Direction {
    /// input
    I = 0,
    /// output
    O = 1,
    /// unknown (unassigned)
    Unknown = 2
}

mod csr;
pub use csr::{VecCSR, DVecCSR};

mod hier_name;
pub use hier_name::{
    HierName, GeneralHierName,
    GeneralPinName, RefPinName,
    GeneralMacroPinName, RefMacroPinName
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum LogicPinType {
    TopPort,
    Net,
    LeafCellPin,
    Others
}

impl LogicPinType {
    #[inline]
    pub fn is_pin(self) -> bool {
        use LogicPinType::*;
        if let TopPort | LeafCellPin = self { true } else { false }
    }

    #[inline]
    pub fn is_net(self) -> bool {
        use LogicPinType::*;
        if let TopPort | Net = self { true } else { false }
    }
}

/// The netlist storage.
///
/// The public members are all READ-ONLY outside. Please modify
/// them through the ECO commands that will be available
/// in the future.
#[readonly::make]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct NetlistDB {
    /// top-level design name.
    pub name: CompactString,
    /// number of cells/nodes/instances in the netlist.
    ///
    /// This is always greater than 1, as the 0th cell is always
    /// the top-level macro.
    pub num_cells: usize,
    /// number of logical pins.
    /// 
    /// A logical pin is not necessarily a pin. It might
    /// be the I/O port of non-leaf modules, or the result
    /// of an assign operation.
    pub num_logic_pins: usize,
    /// number of pins.
    pub num_pins: usize,
    /// number of nets/wires.
    pub num_nets: usize,

    /// Cell name to index.
    ///
    /// The top-level macro is always the 0th cell, which has a
    /// special name of empty string.
    /// Also, the hierarchical non-leaf cells do NOT reside in here,
    /// yet -- they are to-be-added in the future.
    /// This map only contains leaf cells.
    pub cellname2id: HashMap<HierName, usize>,
    /// Logical pin name tuple (cell hier name, macro pin type, vec idx) to logical pin index.
    ///
    /// Logic pin names are always unique without ambiguity.
    /// The case of logic pins include:
    /// 1. net wires  (yes, nets are also ``logic pins''.)
    /// 2. I/O ports of top module and submodules
    /// 3. pins of leaf cells.
    logicpinname2id: HashMap<(HierName, CompactString, Option<isize>), usize>,
    /// Pin name tuple (cell hier name, macro pin type, vec idx) to index.
    /// 
    /// Pin names are always unique without ambiguity.
    /// For top-level named port connections, only the port names are
    /// created as valid pin names. The I/O definition can be referred
    /// in logicpinname2id (private member).
    pub pinname2id: HashMap<(HierName, CompactString, Option<isize>), usize>,
    /// Net name tuple (net hier name, vec idx) to index.
    ///
    /// Multiple nets can be mapped to one single
    /// index, due to connected nets across hierarchy boundaries.
    pub netname2id: HashMap<(HierName, CompactString, Option<isize>), usize>,
    /// Port name tuple (port name, vec idx) to pin index.
    ///
    /// For a design with only normal ports, this is a subset of pinname2id,
    /// but for named ports like .port({portx, porty}),
    /// pinname2id will store port\[0\] and port\[1\], where portname2id
    /// will store portx and porty.
    pub portname2pinid: HashMap<(CompactString, Option<isize>), usize>,

    /// Cell index to macro name.
    pub celltypes: Vec<CompactString>,
    /// Cell index to name (hierarchical).
    ///
    /// This information actually contains the tree structure that
    /// might be useful later when we implement verilog writer.
    pub cellnames: Vec<HierName>,
    /// Logic pin classes.
    logicpintypes: Vec<LogicPinType>,
    /// Logic pin index to name.
    logicpinnames: Vec<(HierName, CompactString, Option<isize>)>,
    /// Pin index to corresponding logic pin index.
    pinid2logicpinid: Vec<usize>,
    /// Net index to net hier and index.
    pub netnames: Vec<(HierName, CompactString, Option<isize>)>,
    /// Pin index to cell hier, macro pin name, and pin index.
    pub pinnames: Vec<(HierName, CompactString, Option<isize>)>,

    /// Pin to parent cell.
    pub pin2cell: UVec<usize>,
    /// Pin to parent net.
    pub pin2net: UVec<usize>,
    /// Cell CSR (using DCSR for dynamic updates).
    pub cell2pin: DVecCSR,
    /// Net CSR (using DCSR for dynamic updates).
    ///
    /// **Caveat**: After assigning directions, it is guaranteed that
    /// the net root would be the first in net CSR.
    /// Before such assignment, the order is not determined.
    pub net2pin: DVecCSR,

    /// Pin direction.
    pub pindirect: UVec<Direction>,

    pub cell2noutputs: UVec<usize>,

    /// Constant zero net index.
    pub net_zero: Option<usize>,
    /// Constant one net index.
    pub net_one: Option<usize>,
}

impl NetlistDB {
    /// This changes the type (i.e. macro name) of a leaf cell.
    pub fn change_cell_type(&mut self, cellid: usize, new_cell_type: CompactString) {
        self.celltypes[cellid] = new_cell_type;
    }

    /// Create a child HierName under a parent cell, reusing Arc references.
    ///
    /// This method is memory-efficient for ECO operations as it reuses
    /// the parent's Arc reference instead of creating a full new path.
    ///
    /// # Arguments
    /// * `parent_cell_id` - Parent cell ID (0 for top-level module)
    /// * `child_name` - Name of the child instance
    ///
    /// # Returns
    /// New HierName with prev pointing to parent via shared Arc
    ///
    /// # Example
    /// ```ignore
    /// let new_hier = db.make_child_hier_name(5, "patch_inst");
    /// // Result: top(shared) -> u_cpu(shared) -> patch_inst(new)
    /// ```
    pub fn make_child_hier_name(&self, parent_cell_id: usize, child_name: &str) -> HierName {
        let parent_hier = self.cellnames[parent_cell_id].clone();
        HierName {
            cur: CompactString::from(child_name),
            prev: Some(Arc::new(parent_hier))
        }
    }

    /// Create HierName from path, attempting to reuse existing memory structure.
    ///
    /// # Arguments
    /// * `parent_path` - Parent HierName path
    /// * `child_name` - Name of the child instance
    ///
    /// # Returns
    /// New HierName, preferring to reuse existing memory structure
    ///
    /// # Behavior
    /// 1. If parent_path exists in cellname2id, reuse its memory
    /// 2. If parent_path is empty, use top-level module (id=0)
    /// 3. Otherwise, create new chain (warns if detached)
    pub fn create_hier_from_path(&self, parent_path: &HierName, child_name: &str) -> HierName {
        if let Some(&id) = self.cellname2id.get(parent_path) {
            return self.make_child_hier_name(id, child_name);
        }
        
        if parent_path.is_empty() {
            return self.make_child_hier_name(0, child_name);
        }
        
        clilog::warn!(NL_ECO, "Creating detached hierarchy for {} under {:?}", child_name, parent_path);
        HierName {
            cur: CompactString::from(child_name),
            prev: Some(Arc::new(parent_path.clone()))
        }
    }

    // ==================== ECO (Engineering Change Order) APIs ====================

    /// Create a new instance (cell) in the netlist.
    ///
    /// # Arguments
    /// * `name` - Hierarchical name of the instance
    /// * `cell_type` - Macro name (type) of the instance
    /// * `lib` - LeafPinProvider to query pin information
    /// * `device` - Device for UVec operations (CPU or CUDA)
    ///
    /// # Returns
    /// * `Ok(cell_id)` - ID of the newly created instance
    /// * `Err(String)` - Error message if creation fails
    ///
    /// # Example
    /// ```ignore
    /// use ulib::Device;
    ///
    /// // Create instance on CPU
    /// let cell_id = db.create_inst(
    ///     HierName::from_topdown_hier_iter(["top", "new_inst"]),
    ///     "NAND2".into(),
    ///     &lib,
    ///     Device::CPU
    /// )?;
    ///
    /// // Create instance on CUDA device 0
    /// let cell_id = db.create_inst(
    ///     HierName::from_topdown_hier_iter(["top", "new_inst"]),
    ///     "NAND2".into(),
    ///     &lib,
    ///     Device::CUDA(0)
    /// )?;
    /// ```
    pub fn create_inst(
        &mut self,
        name: HierName,
        cell_type: CompactString,
        device: Device
    ) -> Result<usize, String> {
        if self.cellname2id.contains_key(&name) {
            return Err(format!("Instance '{}' already exists", name));
        }

        let cell_id = self.num_cells;
        self.num_cells += 1;

        self.cellname2id.insert(name.clone(), cell_id);
        self.celltypes.push(cell_type.clone());
        self.cellnames.push(name.clone());

        unsafe {
            self.cell2noutputs.resize_uninit_preserve(self.num_cells, device);
            self.cell2noutputs[cell_id] = 0;
        }

        clilog::info!(NL_SV_REF, "Created instance '{}' (id={}) of type '{}'",
                      name, cell_id, cell_type);

        Ok(cell_id)
    }

    /// Delete an instance from the netlist.
    ///
    /// **Cascade Delete Mode**: Automatically deletes all pins belonging to this instance
    /// and disconnects them from their nets.
    ///
    /// # Arguments
    /// * `name` - Hierarchical name of the instance to delete
    ///
    /// # Returns
    /// * `Ok(())` - Successfully deleted
    /// * `Err(String)` - Error message if deletion fails
    ///
    /// # 2-Phase ECO Workflow
    /// This operation can be called in Phase 1 (Destructive) without requiring
    /// `apply_eco_changes()` first. All deletions are accumulated in the
    /// pending buffer and committed together with other ECO operations.
    pub fn delete_inst(&mut self, name: &HierName) -> Result<(), String> {
        let cell_id = self.cellname2id.get(name)
            .ok_or_else(|| format!("Instance '{}' not found", name))?;

        let cell_id = *cell_id;

        if cell_id == 0 {
            return Err("Cannot delete the top-level module".to_string());
        }

        let pins_to_delete: Vec<usize> = self.cell2pin.iter_set(cell_id)
            .cloned()
            .collect();

        for pin_id in &pins_to_delete {
            self.delete_pin_internal(*pin_id)?;
        }

        self.cell2pin.dcsr_mut().erase(cell_id);
        self.cellname2id.remove(name);

        clilog::info!(NL_SV_REF, "Deleted instance '{}' (id={}) and deleted {} pins",
                      name, cell_id, pins_to_delete.len());

        Ok(())
    }

    /// Create a new net in the netlist.
    ///
    /// # Arguments
    /// * `name` - Hierarchical name of the net
    /// * `width` - Width of the net (1 for scalar, >1 for vector)
    ///
    /// # Returns
    /// * `Ok(net_id)` - ID of the newly created net
    /// * `Err(String)` - Error message if creation fails
    ///
    /// # Example
    /// ```ignore
    /// // Create a scalar net
    /// let net_id = db.create_net(
    ///     HierName::from_topdown_hier_iter(["top", "new_net"]),
    ///     1
    /// )?;
    ///
    /// // Create a 4-bit vector net
    /// let bus_id = db.create_net(
    ///     HierName::from_topdown_hier_iter(["top", "data_bus"]),
    ///     4
    /// )?;
    /// ```
    pub fn create_net(
        &mut self,
        name: HierName,
        width: usize
    ) -> Result<usize, String> {
        if width == 0 {
            return Err("Net width must be at least 1".to_string());
        }
    
        let parent_hier = if let Some(ref prev) = name.prev {
            (**prev).clone()
        } else {
            HierName::empty()
        };
    
        let base_net_id = self.num_nets;

        for idx in 0..width {
            let net_id = self.num_nets;
            self.num_nets += 1;
    
            let bus_idx = if width == 1 { None } else { Some(idx as isize) };
            let net_name_key = (parent_hier.clone(), name.cur.clone(), bus_idx);
    
            self.netname2id.insert(net_name_key, net_id);
            self.netnames.push((parent_hier.clone(), name.cur.clone(), bus_idx));
        }
    
        clilog::info!(NL_SV_REF, "Created net '{}' with width {} (base id={})",
                      name, width, base_net_id);
    
        Ok(base_net_id)
    }

    /// Delete a net from the netlist.
    ///
    /// **Cascade Delete Mode**: Automatically disconnects all pins connected to this net.
    /// This is designed for 2-Phase ECO workflow where you can delete nets without
    /// manually disconnecting all pins first.
    ///
    /// # Arguments
    /// * `name` - Hierarchical name of the net to delete
    /// * `idx` - Optional bus index (None for scalar, Some(i) for vector)
    ///
    /// # Returns
    /// * `Ok(())` - Successfully deleted
    /// * `Err(String)` - Error message if deletion fails
    ///
    /// # 2-Phase ECO Workflow
    /// This operation can be called in Phase 1 (Destructive) without requiring
    /// `apply_eco_changes()` first. All disconnections are accumulated in the
    /// pending buffer and committed together with other ECO operations.
    pub fn delete_net(
        &mut self,
        name: &HierName,
        idx: Option<isize>
    ) -> Result<(), String> {
        let parent_hier = if let Some(ref prev) = name.prev {
            (**prev).clone()
        } else {
            HierName::empty()
        };

        let net_key = (parent_hier.clone(), name.cur.clone(), idx);

        let net_id = self.netname2id.get(&net_key)
            .ok_or_else(|| format!("Net '{:?}' not found", net_key))?;

        let net_id = *net_id;

        if Some(net_id) == self.net_zero || Some(net_id) == self.net_one {
            return Err("Cannot delete constant nets (0 or 1)".to_string());
        }

        let connected_pins: Vec<usize> = self.net2pin.iter_set(net_id)
            .cloned()
            .collect();

        for pin_id in &connected_pins {
            let _ = self.disconnect_pin_internal(*pin_id);
        }

        self.net2pin.dcsr_mut().erase(net_id);
        self.netname2id.remove(&net_key);

        clilog::info!(NL_SV_REF, "Deleted net '{}{:?}' (id={}) and disconnected {} pins",
                      name, idx, net_id, connected_pins.len());

        Ok(())
    }

    /// Connect a pin to a net.
    ///
    /// # Arguments
    /// * `pin_name` - Hierarchical name of the pin (HierName)
    /// * `pin_type` - Pin name (e.g., "A", "B", "Y")
    /// * `pin_idx` - Optional bus index
    /// * `net_name` - Hierarchical name of the net
    /// * `net_idx` - Optional net bus index
    /// * `direction` - Pin direction (I/O/Unknown)
    /// * `device` - Device for UVec operations (CPU or CUDA)
    ///
    /// # Returns
    /// * `Ok(pin_id)` - ID of the pin (newly created or existing)
    /// * `Err(String)` - Error message if connection fails
    pub fn connect_pin(
        &mut self,
        pin_hier: &HierName,
        pin_type: &str,
        pin_idx: Option<isize>,
        net_hier: &HierName,
        net_idx: Option<isize>,
        lib: &impl LeafPinProvider,
        device: Device
    ) -> Result<usize, String> {
        let net_parent_hier = if let Some(ref prev) = net_hier.prev {
            (**prev).clone()
        } else {
            HierName::empty()
        };

        let net_key = (net_parent_hier.clone(), net_hier.cur.clone(), net_idx);

        let net_id = *self.netname2id.get(&net_key)
            .ok_or_else(|| format!("Net '{:?}' not found", net_key))?;

        let cell_id = self.cellname2id.get(pin_hier)
            .ok_or_else(|| format!("Cell '{}' not found", pin_hier))?;

        let cell_id = *cell_id;

        let cell_type = &self.celltypes[cell_id];
        let direction = lib.direction_of(cell_type, &pin_type.into(), pin_idx);

        let pin_key = (pin_hier.clone(), pin_type.into(), pin_idx);

        let pin_id = if let Some(&existing_id) = self.pinname2id.get(&pin_key) {
            let old_net_id = self.pin2net[existing_id];
            
            if old_net_id != usize::MAX && old_net_id != net_id {
                self.disconnect_pin_internal(existing_id)?;
            }
            
            existing_id
        } else {
            let new_pin_id = self.num_pins;
            self.num_pins += 1;

            let logic_pin_key = (pin_hier.clone(), pin_type.into(), pin_idx);
            let logic_pin_id = if let Some(&existing_logic_id) = self.logicpinname2id.get(&logic_pin_key) {
                existing_logic_id
            } else {
                let new_logic_id = self.num_logic_pins;
                self.num_logic_pins += 1;

                self.logicpinname2id.insert(logic_pin_key.clone(), new_logic_id);
                self.logicpintypes.push(LogicPinType::LeafCellPin);
                self.logicpinnames.push(logic_pin_key);

                new_logic_id
            };

            self.pinname2id.insert(pin_key.clone(), new_pin_id);
            self.pinid2logicpinid.push(logic_pin_id);
            self.pinnames.push(pin_key.clone());

            unsafe {
                self.pin2cell.resize_uninit_preserve(self.num_pins, device);
                self.pin2cell[new_pin_id] = cell_id;

                self.pin2net.resize_uninit_preserve(self.num_pins, device);
                self.pin2net[new_pin_id] = net_id;

                self.pindirect.resize_uninit_preserve(self.num_pins, device);
                self.pindirect[new_pin_id] = direction;
            }

            self.cell2pin.dcsr_mut().append(cell_id, new_pin_id);

            new_pin_id
        };

        self.pin2net[pin_id] = net_id;
        self.net2pin.dcsr_mut().append(net_id, pin_id);

        if direction == Direction::O {
            self.cell2noutputs[cell_id] += 1;
        }

        clilog::info!(NL_SV_REF, "Connected pin '{}{}{:?}' to net '{:?}'",
                      pin_hier, pin_type, pin_idx, net_key);

        Ok(pin_id)
    }

    /// Disconnect a pin from its net.
    ///
    /// # Arguments
    /// * `pin_name` - Hierarchical name of the pin
    /// * `pin_type` - Pin name
    /// * `pin_idx` - Optional bus index
    ///
    /// # Returns
    /// * `Ok(())` - Successfully disconnected
    /// * `Err(String)` - Error message if disconnection fails
    pub fn disconnect_pin(
        &mut self,
        pin_hier: &HierName,
        pin_type: &str,
        pin_idx: Option<isize>
    ) -> Result<(), String> {
        let pin_key = (pin_hier.clone(), pin_type.into(), pin_idx);

        let pin_id = self.pinname2id.get(&pin_key)
            .ok_or_else(|| format!("Pin '{:?}' not found", pin_key))?;

        let pin_id = *pin_id;

        self.disconnect_pin_internal(pin_id)?;

        clilog::info!(NL_SV_REF, "Disconnected pin '{:?}'", pin_key);

        Ok(())
    }

    /// Commit all pending ECO changes to the internal CSR storage.
    ///
    /// This method flushes the pending buffers in the DCSR structures and applies
    /// all pending insertions, deletions, and modifications to the flat memory layout.
    ///
    /// It also performs post-processing to ensure each net's first pin is the driver (output).
    ///
    /// # When to call
    /// You SHOULD call this method after a batch of `connect_pin` or `disconnect_pin`
    /// operations and BEFORE calling operations that read the CSR state, such as:
    /// - `delete_net` (which checks for connected pins)
    /// - Any custom queries that use `iter_set` or other CSR accessors
    ///
    /// # Performance considerations
    /// This operation triggers memory compaction and reallocation, which can be expensive.
    /// Avoid calling it inside tight loops. Instead, batch your ECO operations and call
    /// this method once per batch.
    ///
    /// # Example
    /// ```ignore
    /// let mut db = create_test_db();
    /// let lib = NoDirection;
    ///
    /// // Batch ECO operations
    /// for i in 0..1000 {
    ///     db.create_inst(...)?;
    ///     db.connect_pin(...)?;
    /// }
    ///
    /// // Commit all changes at once
    /// db.apply_eco_changes()?;
    ///
    /// // Now it's safe to query the CSR state
    /// db.delete_net(...)?;
    /// ```
    pub fn apply_eco_changes(&mut self) -> Result<(), String> {
        let mut num_undriven_nets = 0;

        let pending_nets: Vec<usize> = self.net2pin.dcsr_mut().pending_keys().collect();

        for net_id in &pending_nets {
            let mut output_pins = Vec::new();
            for (idx, &pin_id) in self.net2pin.iter_set(*net_id).enumerate() {
                if self.pindirect[pin_id] == Direction::O {
                    output_pins.push((idx, pin_id));
                }
            }

            if output_pins.is_empty() {
                if Some(*net_id) != self.net_zero && Some(*net_id) != self.net_one {
                    num_undriven_nets += 1;
                    clilog::warn!(NL_SV_REF, "Net {} has no driver (undriven net)", net_id);
                }
                continue;
            }

            if output_pins.len() > 1 {
                let pin_ids: Vec<usize> = output_pins.iter().map(|&(_, pin_id)| pin_id).collect();
                clilog::error!(NL_SV_REF, "Net {} has multiple drivers {:?} (short circuit)", net_id, pin_ids);
                return Err(format!("Net {} has {} drivers (short circuit). Pin IDs: {:?}", net_id, output_pins.len(), pin_ids));
            }

            if output_pins[0].0 == 0 {
                continue;
            }

            let driver_idx = output_pins[0].0;
            self.net2pin.swap(*net_id, 0, driver_idx);
        }

        if num_undriven_nets != 0 {
            clilog::warn!(NL_SV_REF, "Found {} undriven nets (excluding constant nets)", num_undriven_nets);
        }

        self.net2pin.dcsr_mut().commit();
        self.cell2pin.dcsr_mut().commit();

        clilog::debug!(NL_ECO, "ECO changes committed to CSR storage.");

        Ok(())
    }

    // ==================== Internal Helper Methods ====================

    /// Internal method to delete a pin.
    ///
    /// This method is idempotent: calling it multiple times on the same pin
    /// is safe and will return Ok().
    fn delete_pin_internal(&mut self, pin_id: usize) -> Result<(), String> {
        let pin_name = self.pinnames.get(pin_id)
            .ok_or_else(|| format!("Pin id {} not found", pin_id))?;

        if !self.pinname2id.contains_key(pin_name) {
            return Ok(());
        }

        let cell_id = self.pin2cell[pin_id];
        let net_id = self.pin2net[pin_id];
        let direction = self.pindirect[pin_id];

        if net_id != usize::MAX {
            self.net2pin.dcsr_mut().remove_element(net_id, pin_id);
        }

        self.cell2pin.dcsr_mut().remove_element(cell_id, pin_id);

        if direction == Direction::O {
            let output_count = self.cell2noutputs[cell_id];
            if output_count > 0 {
                self.cell2noutputs[cell_id] = output_count - 1;
            }
        }

        self.pinname2id.remove(pin_name);

        Ok(())
    }

    /// Internal method to disconnect a pin.
    ///
    /// This method is idempotent: calling it multiple times on the same pin
    /// is safe and will return Ok().
    fn disconnect_pin_internal(&mut self, pin_id: usize) -> Result<(), String> {
        let net_id = self.pin2net[pin_id];

        if net_id == usize::MAX {
            return Ok(());
        }

        let cell_id = self.pin2cell[pin_id];
        let direction = self.pindirect[pin_id];

        self.net2pin.dcsr_mut().remove_element(net_id, pin_id);
        self.pin2net[pin_id] = usize::MAX;

        if direction == Direction::O {
            let output_count = self.cell2noutputs[cell_id];
            if output_count > 0 {
                self.cell2noutputs[cell_id] = output_count - 1;
            }
        }

        Ok(())
    }
}

mod utils;
use utils::*;

mod disjoint_set;
use disjoint_set::*;

mod builder;
pub use builder::{LeafPinProvider, NoDirection};

#[doc(hidden)]
pub use builder::DirectionProvider;
