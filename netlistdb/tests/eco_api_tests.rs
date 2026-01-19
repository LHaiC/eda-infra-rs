//! Comprehensive tests for ECO (Engineering Change Order) APIs.
//!
//! This file combines tests from:
//! - eco_api_test.rs: Basic ECO API unit tests
//! - commit_optimization_test.rs: Performance and commit optimization tests
//! - real_eco_scenario.rs: Real-world ECO scenario integration tests

use netlistdb::{NetlistDB, HierName, Direction, LeafPinProvider};
use sverilogparse::SVerilog;
use ulib::Device;
use compact_str::CompactString;

// =============================================================================
// Helper Functions and Test Fixtures
// =============================================================================

/// Test direction provider that returns known directions for standard cells
struct TestDirectionProvider;

impl LeafPinProvider for TestDirectionProvider {
    fn direction_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString,
        _pin_idx: Option<isize>
    ) -> Direction {
        match (macro_name.as_str(), pin_name.as_str()) {
            // NAND2: A, B are inputs, Y is output
            ("NAND2", "A" | "B") => Direction::I,
            ("NAND2", "Y") => Direction::O,
            // NOR2: A, B are inputs, Y is output
            ("NOR2", "A" | "B") => Direction::I,
            ("NOR2", "Y") => Direction::O,
            // XOR2: A, B are inputs, Y is output
            ("XOR2", "A" | "B") => Direction::I,
            ("XOR2", "Y") => Direction::O,
            // Default: unknown
            _ => Direction::Unknown,
        }
    }

    fn width_of(
        &self,
        _macro_name: &CompactString,
        _pin_name: &CompactString
    ) -> Option<sverilogparse::SVerilogRange> {
        None
    }
}

/// No direction provider for simple tests that don't need pin direction
pub struct NoDirection;

impl LeafPinProvider for NoDirection {
    fn direction_of(
        &self,
        _: &CompactString,
        _: &CompactString,
        _: Option<isize>
    ) -> Direction {
        Direction::Unknown
    }

    fn width_of(
        &self,
        _: &CompactString,
        _: &CompactString
    ) -> Option<sverilogparse::SVerilogRange> {
        None
    }
}

/// Helper function to create a minimal test netlist database
fn create_test_db() -> NetlistDB {
    let verilog_code = r#"
module test();
    // Empty module for testing
endmodule
"#;

    let sverilog = SVerilog::parse_str(verilog_code).unwrap();
    NetlistDB::from_sverilog(sverilog, Some("test"), &NoDirection).unwrap()
}

/// Helper function to create a test database with direction support
fn create_test_db_with_direction() -> NetlistDB {
    let verilog_code = r#"
module test();
    // Empty module for testing
endmodule
"#;

    let sverilog = SVerilog::parse_str(verilog_code).unwrap();
    NetlistDB::from_sverilog(sverilog, Some("test"), &TestDirectionProvider).unwrap()
}

// =============================================================================
// Basic ECO API Tests
// =============================================================================

#[test]
fn test_create_inst() {
    let mut db = create_test_db();

    let inst_name = db.make_child_hier_name(0, "test_inst");
    
    db.create_inst(inst_name.clone(), "NAND2".into(), Device::CPU)
        .expect("Failed to create instance");

    assert_eq!(db.num_cells, 2);  // top + test_inst
    assert!(db.cellname2id.contains_key(&inst_name));
}

#[test]
fn test_delete_inst() {
    let mut db = create_test_db();

    let inst_name = db.make_child_hier_name(0, "temp_inst");
    db.create_inst(inst_name.clone(), "NOR2".into(), Device::CPU)
        .expect("Failed to create instance");

    db.delete_inst(&inst_name).expect("Failed to delete instance");

    assert!(!db.cellname2id.contains_key(&inst_name));
}

#[test]
fn test_create_net() {
    let mut db = create_test_db();

    let net_name = db.make_child_hier_name(0, "test_net");
    
    db.create_net(net_name.clone(), 1)
        .expect("Failed to create net");

    assert_eq!(db.num_nets, 1);
    assert!(db.netname2id.contains_key(&(HierName::empty(), "test_net".into(), None)));
}

#[test]
fn test_delete_net() {
    let mut db = create_test_db();

    let net_name = db.make_child_hier_name(0, "temp_net");
    db.create_net(net_name.clone(), 1).unwrap();

    // 删除网络
    db.delete_net(&net_name, None).expect("Failed to delete net");

    assert!(!db.netname2id.contains_key(&(HierName::empty(), "temp_net".into(), None)));
}

#[test]
fn test_connect_disconnect_pin() {
    let mut db = create_test_db();
    let lib = NoDirection;

    let inst_name = db.make_child_hier_name(0, "test_inst");
    db.create_inst(inst_name.clone(), "NAND2".into(), Device::CPU)
        .expect("Failed to create instance");

    let net_name = db.make_child_hier_name(0, "test_net");
    db.create_net(net_name.clone(), 1)
        .expect("Failed to create net");

    // Connect a pin
    let pin_id = db.connect_pin(
        &inst_name,
        "A",
        None,
        &net_name,
        None,
        &lib,
        Device::CPU
    ).expect("Failed to connect pin");

    assert_eq!(db.num_pins, 1);
    assert_eq!(db.num_logic_pins, 1);

    // Disconnect the pin
    db.disconnect_pin(&inst_name, "A", None)
        .expect("Failed to disconnect pin");
}

#[test]
fn test_eco_workflow() {
    let mut db = create_test_db();
    let lib = NoDirection;

    let inst1 = db.make_child_hier_name(0, "inst1");
    let inst2 = db.make_child_hier_name(0, "inst2");
    db.create_inst(inst1.clone(), "NAND2".into(), Device::CPU).unwrap();
    db.create_inst(inst2.clone(), "NOR2".into(), Device::CPU).unwrap();

    let net_a = db.make_child_hier_name(0, "net_a");
    let net_b = db.make_child_hier_name(0, "net_b");
    let net_y = db.make_child_hier_name(0, "net_y");
    db.create_net(net_a.clone(), 1).unwrap();
    db.create_net(net_b.clone(), 1).unwrap();
    db.create_net(net_y.clone(), 1).unwrap();

    // Connect pins
    db.connect_pin(&inst1, "A", None, &net_a, None, &lib, Device::CPU).unwrap();
    db.connect_pin(&inst1, "B", None, &net_b, None, &lib, Device::CPU).unwrap();
    db.connect_pin(&inst1, "Y", None, &net_y, None, &lib, Device::CPU).unwrap();

    db.connect_pin(&inst2, "A", None, &net_a, None, &lib, Device::CPU).unwrap();
    db.connect_pin(&inst2, "B", None, &net_b, None, &lib, Device::CPU).unwrap();
    db.connect_pin(&inst2, "Y", None, &net_y, None, &lib, Device::CPU).unwrap();

    // Verify connections
    assert_eq!(db.num_pins, 6);
    assert_eq!(db.num_logic_pins, 6);

    // Commit before deletion
    db.apply_eco_changes().unwrap();

    // Delete one instance
    db.delete_inst(&inst1).unwrap();

    // Verify deletion
    assert!(!db.cellname2id.contains_key(&inst1));
    assert!(db.cellname2id.contains_key(&inst2));
}

#[test]
fn test_make_child_hier_name_memory_efficiency() {
    let mut db = create_test_db();

    // 创建第一个实例
    let inst1 = db.make_child_hier_name(0, "inst1");
    db.create_inst(inst1.clone(), "NAND2".into(), Device::CPU).unwrap();

    // 创建第二个实例（应该共享 top 的 Arc）
    let inst2 = db.make_child_hier_name(0, "inst2");
    db.create_inst(inst2.clone(), "NOR2".into(), Device::CPU).unwrap();

    // 创建第三个实例（应该共享 top 的 Arc）
    let inst3 = db.make_child_hier_name(0, "inst3");
    db.create_inst(inst3.clone(), "XOR2".into(), Device::CPU).unwrap();

    // 创建嵌套的层次结构来测试 Arc 共享
    let parent_inst = db.make_child_hier_name(0, "parent_module");
    db.create_inst(parent_inst.clone(), "MODULE".into(), Device::CPU).unwrap();
    
    let parent_id = db.cellname2id.get(&parent_inst).unwrap();
    
    let child1 = db.make_child_hier_name(*parent_id, "child1");
    let child2 = db.make_child_hier_name(*parent_id, "child2");
    
    // 验证两个子实例的 prev 都指向同一个 Arc
    let child1_ref = &child1;
    let child2_ref = &child2;
    
    assert!(child1_ref.prev.is_some());
    assert!(child2_ref.prev.is_some());
    
    // 验证它们指向的内容是否相同
    let parent1 = child1_ref.prev.as_ref().unwrap();
    let parent2 = child2_ref.prev.as_ref().unwrap();
    assert_eq!(parent1.cur, parent2.cur, "Parent names should be equal");
    assert_eq!(parent1.prev, parent2.prev, "Parent prev should be equal");
    
    // 验证它们指向的父节点是否是同一个 cell
    let parent_hier = &db.cellnames[*parent_id];
    assert_eq!(parent1.cur, parent_hier.cur);
}

#[test]
fn test_connect_pin_auto_disconnect() {
    let mut db = create_test_db();
    let lib = NoDirection;

    let inst_name = db.make_child_hier_name(0, "test_inst");
    db.create_inst(inst_name.clone(), "NAND2".into(), Device::CPU)
        .expect("Failed to create instance");

    let net1 = db.make_child_hier_name(0, "net1");
    let net2 = db.make_child_hier_name(0, "net2");
    db.create_net(net1.clone(), 1).expect("Failed to create net1");
    db.create_net(net2.clone(), 1).expect("Failed to create net2");

    // 将引脚连接到 net1
    db.connect_pin(&inst_name, "A", None, &net1, None, &lib, Device::CPU)
        .expect("Failed to connect pin to net1");

    db.apply_eco_changes();

    let net1_id = *db.netname2id.get(&(HierName::empty(), "net1".into(), None)).unwrap();
    let net2_id = *db.netname2id.get(&(HierName::empty(), "net2".into(), None)).unwrap();

    let net1_pins: Vec<usize> = db.net2pin.iter_set(net1_id).cloned().collect();
    assert_eq!(net1_pins.len(), 1, "net1 should have 1 pin");

    // 将同一个引脚重新连接到 net2
    db.connect_pin(&inst_name, "A", None, &net2, None, &lib, Device::CPU)
        .expect("Failed to reconnect pin to net2");

    db.apply_eco_changes();

    let net1_pins_after: Vec<usize> = db.net2pin.iter_set(net1_id).cloned().collect();
    let net2_pins_after: Vec<usize> = db.net2pin.iter_set(net2_id).cloned().collect();
    
    assert_eq!(net1_pins_after.len(), 0, "net1 should have 0 pins after reconnect");
    assert_eq!(net2_pins_after.len(), 1, "net2 should have 1 pin after reconnect");
}

#[test]
fn test_delete_net_with_connected_pins() {
    let mut db = create_test_db();
    let lib = NoDirection;

    let inst_name = db.make_child_hier_name(0, "test_inst");
    db.create_inst(inst_name.clone(), "NAND2".into(), Device::CPU)
        .expect("Failed to create instance");

    let net_name = db.make_child_hier_name(0, "test_net");
    db.create_net(net_name.clone(), 1).expect("Failed to create net");

    db.connect_pin(&inst_name, "A", None, &net_name, None, &lib, Device::CPU)
        .expect("Failed to connect pin");

    db.apply_eco_changes();

    let pin_key = (inst_name.clone(), "A".into(), None);
    let pin_id = *db.pinname2id.get(&pin_key).unwrap();

    // 验证引脚已连接
    assert_eq!(db.pin2net[pin_id], *db.netname2id.get(&(HierName::empty(), "test_net".into(), None)).unwrap());

    // 删除网络（应该成功，自动断开所有连接的引脚）
    db.delete_net(&net_name, None)
        .expect("delete_net should succeed with cascade delete");

    db.apply_eco_changes();

    assert!(!db.netname2id.contains_key(&(HierName::empty(), "test_net".into(), None)));
    assert_eq!(db.pin2net[pin_id], usize::MAX);
}

#[test]
fn test_delete_net_after_disconnect() {
    let mut db = create_test_db();
    let lib = NoDirection;

    let inst_name = db.make_child_hier_name(0, "test_inst");
    db.create_inst(inst_name.clone(), "NAND2".into(), Device::CPU)
        .expect("Failed to create instance");

    let net_name = db.make_child_hier_name(0, "test_net");
    db.create_net(net_name.clone(), 1).expect("Failed to create net");

    db.connect_pin(&inst_name, "A", None, &net_name, None, &lib, Device::CPU)
        .expect("Failed to connect pin");

    db.disconnect_pin(&inst_name, "A", None)
        .expect("Failed to disconnect pin");

    db.apply_eco_changes();

    db.delete_net(&net_name, None)
        .expect("delete_net should succeed after disconnecting all pins");

    assert!(!db.netname2id.contains_key(&(HierName::empty(), "test_net".into(), None)));
}

#[test]
fn test_multiple_pins_disconnect_before_delete() {
    let mut db = create_test_db();
    let lib = NoDirection;

    let inst1 = db.make_child_hier_name(0, "inst1");
    let inst2 = db.make_child_hier_name(0, "inst2");
    db.create_inst(inst1.clone(), "NAND2".into(), Device::CPU).unwrap();
    db.create_inst(inst2.clone(), "NOR2".into(), Device::CPU).unwrap();

    let net_name = db.make_child_hier_name(0, "shared_net");
    db.create_net(net_name.clone(), 1).expect("Failed to create net");

    db.connect_pin(&inst1, "A", None, &net_name, None, &lib, Device::CPU)
        .expect("Failed to connect inst1.A");
    db.connect_pin(&inst2, "A", None, &net_name, None, &lib, Device::CPU)
        .expect("Failed to connect inst2.A");

    db.apply_eco_changes();

    db.delete_net(&net_name, None)
        .expect("delete_net should succeed with cascade delete");

    db.apply_eco_changes();

    assert!(!db.netname2id.contains_key(&(HierName::empty(), "shared_net".into(), None)));

    let pin1_key = (inst1.clone(), "A".into(), None);
    let pin2_key = (inst2.clone(), "A".into(), None);
    let pin1_id = *db.pinname2id.get(&pin1_key).unwrap();
    let pin2_id = *db.pinname2id.get(&pin2_key).unwrap();
    assert_eq!(db.pin2net[pin1_id], usize::MAX);
    assert_eq!(db.pin2net[pin2_id], usize::MAX);
}

#[test]
fn test_delete_net_without_commit_should_fail() {
    let mut db = create_test_db();
    let lib = NoDirection;

    let inst_name = db.make_child_hier_name(0, "test_inst");
    db.create_inst(inst_name.clone(), "NAND2".into(), Device::CPU)
        .expect("Failed to create instance");

    let net_name = db.make_child_hier_name(0, "test_net");
    db.create_net(net_name.clone(), 1).expect("Failed to create net");

    db.connect_pin(&inst_name, "A", None, &net_name, None, &lib, Device::CPU)
        .expect("Failed to connect pin");

    db.delete_net(&net_name, None)
        .expect("delete_net should succeed even with pending ECO changes (cascade delete)");

    db.apply_eco_changes();

    assert!(!db.netname2id.contains_key(&(HierName::empty(), "test_net".into(), None)));
}

#[test]
fn test_delete_inst_without_commit_should_fail() {
    let mut db = create_test_db();
    let lib = NoDirection;

    let inst_name = db.make_child_hier_name(0, "test_inst");
    db.create_inst(inst_name.clone(), "NAND2".into(), Device::CPU)
        .expect("Failed to create instance");

    let net_name = db.make_child_hier_name(0, "test_net");
    db.create_net(net_name.clone(), 1).expect("Failed to create net");

    db.connect_pin(&inst_name, "A", None, &net_name, None, &lib, Device::CPU)
        .expect("Failed to connect pin");

    db.delete_inst(&inst_name)
        .expect("delete_inst should succeed even with pending ECO changes (cascade delete)");

    db.apply_eco_changes();

    assert!(!db.cellname2id.contains_key(&inst_name));
}

// =============================================================================
// Driver Pin Ordering Tests
// =============================================================================

#[test]
fn test_driver_pin_ordering_with_pending_data() {
    let mut db = create_test_db_with_direction();
    let lib = TestDirectionProvider;

    let inst1 = db.make_child_hier_name(0, "inst1");
    db.create_inst(inst1.clone(), "NAND2".into(), Device::CPU).unwrap();

    let net_name = db.make_child_hier_name(0, "test_net");
    db.create_net(net_name.clone(), 1).unwrap();

    // Connect inputs first (wrong order)
    db.connect_pin(&inst1, "A", None, &net_name, None, &lib, Device::CPU).unwrap();
    db.connect_pin(&inst1, "B", None, &net_name, None, &lib, Device::CPU).unwrap();

    // Connect output last
    db.connect_pin(&inst1, "Y", None, &net_name, None, &lib, Device::CPU).unwrap();

    // Commit - this should reorder pins so driver is first
    db.apply_eco_changes().unwrap();

    // Verify driver is first
    let net_id = *db.netname2id.get(&(HierName::empty(), "test_net".into(), None)).unwrap();
    let pins: Vec<usize> = db.net2pin.iter_set(net_id).cloned().collect();

    assert_eq!(pins.len(), 3, "Should have 3 pins");
    assert_eq!(
        db.pindirect[pins[0]],
        Direction::O,
        "First pin should be output (driver)"
    );

    // Verify the driver pin is inst1.Y
    let driver_pin_name = &db.pinnames[pins[0]];
    assert_eq!(driver_pin_name.0, inst1);
    assert_eq!(driver_pin_name.1, "Y");
}

#[test]
fn test_post_process_eco_multiple_drivers_error() {
    let mut db = create_test_db_with_direction();
    let lib = TestDirectionProvider;

    let inst1 = db.make_child_hier_name(0, "inst1");
    let inst2 = db.make_child_hier_name(0, "inst2");
    db.create_inst(inst1.clone(), "NAND2".into(), Device::CPU).unwrap();
    db.create_inst(inst2.clone(), "NOR2".into(), Device::CPU).unwrap();

    let net_name = db.make_child_hier_name(0, "test_net");
    db.create_net(net_name.clone(), 1).expect("Failed to create net");

    // 连接两个输出引脚到同一个网络（多驱动！）
    db.connect_pin(&inst1, "Y", None, &net_name, None, &lib, Device::CPU).unwrap();
    db.connect_pin(&inst2, "Y", None, &net_name, None, &lib, Device::CPU).unwrap();

    // apply_eco_changes 应该检测到多驱动错误
    let result = db.apply_eco_changes();

    assert!(result.is_err(), "apply_eco_changes should return error for multiple drivers");
    let error_msg = result.unwrap_err();
    assert!(error_msg.contains("2 drivers") || error_msg.contains("short circuit"),
            "Error should mention multiple drivers or short circuit, got: {}", error_msg);
}

#[test]
fn test_two_phase_eco_workflow() {
    let mut db = create_test_db_with_direction();
    let lib = TestDirectionProvider;

    // ========== 初始设置 ==========
    let old_inst = db.make_child_hier_name(0, "old_inst");
    db.create_inst(old_inst.clone(), "NAND2".into(), Device::CPU).unwrap();

    let old_net = db.make_child_hier_name(0, "old_net");
    db.create_net(old_net.clone(), 1).unwrap();

    db.connect_pin(&old_inst, "A", None, &old_net, None, &lib, Device::CPU).unwrap();
    db.connect_pin(&old_inst, "B", None, &old_net, None, &lib, Device::CPU).unwrap();
    db.connect_pin(&old_inst, "Y", None, &old_net, None, &lib, Device::CPU).unwrap();

    db.apply_eco_changes().expect("Failed to commit initial setup");

    // ========== Phase 1: Destructive (只做减法) ==========
    db.delete_net(&old_net, None).unwrap();
    db.delete_inst(&old_inst).unwrap();

    // ========== Phase 2: Constructive (只做加法) ==========
    let new_net = db.make_child_hier_name(0, "new_net");
    db.create_net(new_net.clone(), 1).unwrap();

    let new_inst = db.make_child_hier_name(0, "new_inst");
    db.create_inst(new_inst.clone(), "NOR2".into(), Device::CPU).unwrap();

    db.connect_pin(&new_inst, "A", None, &new_net, None, &lib, Device::CPU).unwrap();
    db.connect_pin(&new_inst, "Y", None, &new_net, None, &lib, Device::CPU).unwrap();

    // ========== Phase 3: Finalize (统一提交) ==========
    db.apply_eco_changes().expect("Failed to commit final changes");

    // 验证旧的实例和网络已被删除
    assert!(!db.cellname2id.contains_key(&old_inst));
    assert!(!db.netname2id.contains_key(&(HierName::empty(), "old_net".into(), None)));

    // 验证新的实例和网络已创建
    assert!(db.cellname2id.contains_key(&new_inst));
    assert!(db.netname2id.contains_key(&(HierName::empty(), "new_net".into(), None)));

    let new_net_id = *db.netname2id.get(&(HierName::empty(), "new_net".into(), None)).unwrap();
    let new_net_pins: Vec<usize> = db.net2pin.iter_set(new_net_id).cloned().collect();
    assert_eq!(new_net_pins.len(), 2);

    // 验证驱动引脚在第一位
    let first_pin = new_net_pins[0];
    assert_eq!(db.pindirect[first_pin], Direction::O);
}

// =============================================================================
// Commit Optimization Tests
// =============================================================================

#[test]
fn test_single_commit_optimization() {
    let mut db = create_test_db();
    let lib = TestDirectionProvider;

    let num_instances = 100;

    for i in 0..num_instances {
        let inst_name = db.make_child_hier_name(0, &format!("inst{}", i));
        db.create_inst(inst_name.clone(), "NAND2".into(), Device::CPU).unwrap();

        let net_name = db.make_child_hier_name(0, &format!("net{}", i));
        db.create_net(net_name.clone(), 1).unwrap();

        db.connect_pin(&inst_name, "A", None, &net_name, None, &lib, Device::CPU).unwrap();
        db.connect_pin(&inst_name, "B", None, &net_name, None, &lib, Device::CPU).unwrap();
        db.connect_pin(&inst_name, "Y", None, &net_name, None, &lib, Device::CPU).unwrap();
    }

    let result = db.apply_eco_changes();

    assert!(result.is_ok(), "apply_eco_changes should succeed");

    assert_eq!(db.num_cells - 1, num_instances, "Should have created all instances");
    assert_eq!(db.num_nets, num_instances, "Should have created all nets");
    assert_eq!(db.num_pins, num_instances * 3, "Should have created all pins");

    // Verify driver pins are in the first position
    for i in 0..num_instances {
        let net_key = (HierName::empty(), format!("net{}", i).into(), None);

        if let Some(&net_id) = db.netname2id.get(&net_key) {
            let pins: Vec<usize> = db.net2pin.iter_set(net_id).cloned().collect();
            if !pins.is_empty() {
                let first_pin = pins[0];
                assert_eq!(
                    db.pindirect[first_pin],
                    Direction::O,
                    "First pin of net {} should be output (driver)",
                    net_id
                );
            }
        }
    }
}

// =============================================================================
// Real-World ECO Scenario Tests
// =============================================================================

/// Direction provider for real ECO scenario tests
struct SimpleLibProvider;

impl LeafPinProvider for SimpleLibProvider {
    fn direction_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString,
        _pin_idx: Option<isize>
    ) -> Direction {
        match (macro_name.as_str(), pin_name.as_str()) {
            ("na02s01", "o") => Direction::O,
            ("ms00f80", "o") => Direction::O,
            ("in01s01", "o") => Direction::O,
            ("BUF_X1", "Y") => Direction::O,
            ("BUF_X1", "A") => Direction::I,
            _ => Direction::I,
        }
    }

    fn width_of(
        &self,
        _macro_name: &CompactString,
        _pin_name: &CompactString
    ) -> Option<sverilogparse::SVerilogRange> {
        None
    }
}

#[test]
fn test_real_eco_scenario_bypass_flipflop() {
    clilog::init_stdout_simple_trace();
    let lib = SimpleLibProvider;

    let verilog_source = r#"
module simple(inp1, inp2, tau2015_clk, out);
    input inp1, inp2, tau2015_clk;
    output out;
    wire n1, n2, n3;
    na02s01 u1(.a(inp1), .b(inp2), .o(n1));
    ms00f80 u2(.d(n1), .ck(tau2015_clk), .o(n2)); // Target: Remove this DFF
    in01s01 u3(.i(n2), .o(n3));
    in01s01 u4(.i(n3), .o(out));
endmodule
"#;

    let mut db = NetlistDB::from_sverilog_source(
        verilog_source,
        None,
        &lib
    ).expect("Failed to parse initial netlist");

    println!("--- Initial State ---");
    println!("Cells: {}, Nets: {}, Pins: {}", db.num_cells, db.num_nets, db.num_pins);

    let u1_name = HierName::from_topdown_hier_iter(["u1"]);
    let u2_name = HierName::from_topdown_hier_iter(["u2"]);
    let u3_name = HierName::from_topdown_hier_iter(["u3"]);

    let n1_key = (HierName::empty(), "n1".into(), None);
    let n2_key = (HierName::empty(), "n2".into(), None);

    let n1_id = *db.netname2id.get(&n1_key).expect("n1 should exist");

    // Phase 1: Destruction
    println!("\n--- Phase 1: ECO Delete (Operation Skuld) ---");

    db.delete_inst(&u2_name).expect("Failed to delete u2");

    let n2_name = HierName::from_topdown_hier_iter(["n2"]);
    db.delete_net(&n2_name, None).expect("Failed to delete n2");

    // Phase 2: Construction
    println!("\n--- Phase 2: ECO Create ---");

    let u_new_name = HierName::from_topdown_hier_iter(["u_eco_buf"]);
    db.create_inst(u_new_name.clone(), "BUF_X1".into(), Device::CPU)
        .expect("Failed to create ECO buffer");

    let n_new_name = HierName::from_topdown_hier_iter(["n_eco_wire"]);
    db.create_net(n_new_name.clone(), 1)
        .expect("Failed to create ECO net");

    // Phase 3: Connection
    println!("\n--- Phase 3: ECO Connect ---");

    let n1_name = HierName::from_topdown_hier_iter(["n1"]);
    db.connect_pin(
        &u_new_name, "A", None,
        &n1_name, None,
        &lib, Device::CPU
    ).expect("Failed to connect n1 to buf.A");

    db.connect_pin(
        &u_new_name, "Y", None,
        &n_new_name, None,
        &lib, Device::CPU
    ).expect("Failed to connect buf.Y to n_eco");

    db.connect_pin(
        &u3_name, "i", None,
        &n_new_name, None,
        &lib, Device::CPU
    ).expect("Failed to connect n_eco to u3.i");

    // Phase 4: Commit
    println!("\n--- Committing Changes ---");
    db.apply_eco_changes().expect("ECO Commit failed! The timeline is unstable!");

    // Phase 5: Verification
    println!("\n--- Verification ---");

    assert!(!db.cellname2id.contains_key(&u2_name), "u2 should be deleted");
    assert!(db.cellname2id.contains_key(&u_new_name), "u_eco_buf should exist");

    assert!(!db.netname2id.contains_key(&n2_key), "n2 should be deleted");
    let n_new_key = (HierName::empty(), "n_eco_wire".into(), None);
    assert!(db.netname2id.contains_key(&n_new_key), "n_eco_wire should exist");

    // Check Connectivity of n1
    let n1_pins: Vec<usize> = db.net2pin.iter_set(n1_id).cloned().collect();
    println!("Net n1 pins: {:?}", n1_pins);
    
    let mut found_u1_o = false;
    let mut found_buf_a = false;
    let mut found_u2_d = false;

    for pin_id in n1_pins {
        let (hier, pin_type, _) = &db.pinnames[pin_id];
        if hier == &u1_name && *pin_type == "o" { found_u1_o = true; }
        if hier == &u_new_name && *pin_type == "A" { found_buf_a = true; }
        if hier == &u2_name { found_u2_d = true; } 
    }

    assert!(found_u1_o, "n1 must still be driven by u1.o");
    assert!(found_buf_a, "n1 must now drive u_eco_buf.A");
    assert!(!found_u2_d, "n1 must NOT drive u2.d (Ghost connection detected!)");

    // Check Connectivity of New Net (n_eco_wire)
    let n_new_real_id = *db.netname2id.get(&n_new_key).unwrap();
    let n_new_pins: Vec<usize> = db.net2pin.iter_set(n_new_real_id).cloned().collect();
    println!("Net n_eco_wire pins: {:?}", n_new_pins);

    assert_eq!(n_new_pins.len(), 2, "n_eco_wire should have exactly 2 pins");
    
    let driver_pin_id = n_new_pins[0];
    assert_eq!(db.pindirect[driver_pin_id], Direction::O, "First pin must be driver (Output)");
    assert_eq!(db.pinnames[driver_pin_id].0, u_new_name, "Driver must be u_eco_buf");

    let load_pin_id = n_new_pins[1];
    assert_eq!(db.pindirect[load_pin_id], Direction::I, "Second pin must be load (Input)");
    assert_eq!(db.pinnames[load_pin_id].0, u3_name, "Load must be u3");

    println!("\nEl Psy Kongroo. The operation was successful.");
}
