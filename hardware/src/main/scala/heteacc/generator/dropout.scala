//===------------------------------------------------------------*- Scala -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//

package heteacc.generator

import chipsalliance.rocketchip.config._
import chisel3._
import chisel3.Module._
import chisel3.iotesters._
import chisel3.testers._
import chisel3.util._
import heteacc.config._
import heteacc.execution._
import heteacc.fpu._
import heteacc.interfaces._
import heteacc.junctions._
import heteacc.loop._
import heteacc.memory._
import heteacc.node._
import utility._

abstract class dropoutDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new Call(List(32, 32))))
    val out = Decoupled(new Call(List(32)))
  })
}

class dropoutDF(implicit p: Parameters) extends dropoutDFIO()(p) {

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1)))
  FineGrainedArgCall.io.In <> io.in

  val mem_ctrl_cache = Module(new MemoryEngine(Size = 1152, ID = 0, NumRead = 2, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/dropout/dropout.txt")

  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c7_i32 = arith.constant 7 : i32
  val int_const_0 = Module(new ConstFastNode(value = 7, ID = 0))

  //%c7_0 = arith.constant 7 : i32
  val int_const_2 = Module(new ConstFastNode(value = 7, ID = 2))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_3 = Module(new ConstFastNode(value = 0, ID = 3))

  //%c3_i32 = arith.constant 3 : i32
  val int_const_4 = Module(new ConstFastNode(value = 3, ID = 4))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

  //%c0_i32_0 = arith.constant 0 : i32
  val int_const_6 = Module(new ConstFastNode(value = 0, ID = 6))

  //%c2_i32 = arith.constant 2 : i32
  val int_const_7 = Module(new ConstFastNode(value = 2, ID = 7))

  //%c0_i32_1 = arith.constant 0 : i32
  val int_const_8 = Module(new ConstFastNode(value = 0, ID = 8))

  //%c1 = arith.constant 1 : index
  val int_const_9 = Module(new ConstFastNode(value = 1, ID = 9))

  //%c1_0 = arith.constant 1 : index
  val int_const_10 = Module(new ConstFastNode(value = 1, ID = 10))

  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 12, NumPhi = 0, BID = 0))
  /* ================================================================== *
   *                   Printing Operation nodes.                        *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%2 = arith.andi %arg2, %c7_i32 ; %4 = arith.addi %2, %c7_0 : i32
  val fused_mask_add = Module(new Chain(ID = 2, NumOps = 2, OpCodes = Array("And", "Add"))(sign = false)(p))

  //%6 = arith.cmpi eq, %4, %c0_i32 : i32
  val int_cmp_6 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 6, opCode = "eq")(sign = false, Debug = false))

  //%7 = arith.shrui %arg2, %c3_i32 : i32
  val int_lshr_7 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 7, opCode = "lshr")(sign = false, Debug = false))

  //%9 = dataflow.addr %arg1[%7] {memShape = [128]} : memref<128xi32>[index] -> i32
  val address_9 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 9)(ElementSize = 1, ArraySize = List()))

  //%10 = dataflow.load %9 {ID = 0 : i32} : i32 -> i32
  val load_10 = Module(new Load(NumOuts = 1, ID = 10, RouteID = 0))

  //%11 = dataflow.select %6, %10, %arg3 : i32
  val select_11 = Module(new SelectNodeWithoutState(NumOuts = 2, ID = 11))

  //%12 = arith.andi %11, %c1_i32 ; %13 = arith.cmpi ne, %12, %c0_i32_0 : i32
  val fused_select_cmp = Module(new Chain(ID = 12, NumOps = 2, OpCodes = Array("and", "ne"))(sign = false)(p))

  //%14 = dataflow.addr %arg0[%arg4] {memShape = [1024]} : memref<1024xi32>[index] -> i32
  val address_14 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 14)(ElementSize = 1, ArraySize = List()))

  //%15 = dataflow.load %14 {ID = 1 : i32} : i32 -> i32
  val load_15 = Module(new Load(NumOuts = 1, ID = 15, RouteID = 1))

  //%16 = arith.muli %15, %c2_i32 : i32
  val int_mul_16 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 16, opCode = "Mul")(sign = false, Debug = false))

  //%17 = dataflow.select %13, %16, %c0_i32_1 : i32
  val select_17 = Module(new SelectNodeWithoutState(NumOuts = 1, ID = 17))

  //%18 = arith.addi %arg5, %17 : i32
  val int_add_18 = Module(new ComputeNodeWithoutStateSupportCarry(NumOuts = 2, ID = 18, opCode = "Add")(sign = false, Debug = false))

  //%19 = arith.shrsi %11, %c1 : i32
  val int_shr_19 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 19, opCode = "ashr")(sign = false, Debug = false))

  //%20 = arith.addi %arg2, %c1_0 {Exe = "Loop"} : index
  val int_add_20 = Module(new ComputeNodeWithoutStateSupportCarry(NumOuts = 1, ID = 20, opCode = "Add")(sign = false, Debug = false))

  //func.return %0 : i32
  val return_23 = Module(new RetNode2(retTypes = List(32), ID = 23))

  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNodeExperimental(NumIns = List(1, 1), NumOuts = List(1), NumCarry = List(4, 1, 1), NumExits = 1, ID = 0, LoopCounterMax = 1024, LoopCounterStep = 1))

  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_0.io.loopExit(0) <> return_23.io.In.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back

  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> state_branch_0.io.Out(0)

  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

  loop_0.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)

  loop_0.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  address_9.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  address_14.io.baseAddress <> loop_0.io.OutLiveIn.elements("field1")(0)

  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> int_add_18.io.Out(0)

  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_23.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)

  /* ================================================================== *
   *                   Carry dependencies.                              *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_20.io.Out(0)

  fused_mask_add.io.In(0) <> loop_0.io.CarryDepenOut.elements("field0")(1)

  address_14.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(2)

  loop_0.io.CarryDepenIn(1) <> int_shr_19.io.Out(0)

  select_11.io.InData2 <> loop_0.io.CarryDepenOut.elements("field1")(0)

  loop_0.io.CarryDepenIn(2) <> int_add_18.io.Out(1)

  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  fused_mask_add.io.In(1) <> int_const_0.io.Out

  fused_mask_add.io.In(2) <> int_const_2.io.Out

  int_cmp_6.io.RightIO <> int_const_3.io.Out

  int_lshr_7.io.RightIO <> int_const_4.io.Out

  fused_select_cmp.io.In(1) <> int_const_5.io.Out

  fused_select_cmp.io.In(2) <> int_const_6.io.Out

  int_mul_16.io.RightIO <> int_const_7.io.Out

  select_17.io.InData2 <> int_const_8.io.Out

  int_shr_19.io.RightIO <> int_const_9.io.Out

  int_add_20.io.RightIO <> int_const_10.io.Out

  int_lshr_7.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(0)

  int_cmp_6.io.LeftIO <> fused_mask_add.io.Out(0)

  select_11.io.Select <> int_cmp_6.io.Out(0)

  address_9.io.idx(0) <> int_lshr_7.io.Out(0)

  load_10.GepAddr <> address_9.io.Out(0)

  select_11.io.InData1 <> load_10.io.Out(0)

  fused_select_cmp.io.In(0) <> select_11.io.Out(0)

  int_shr_19.io.LeftIO <> select_11.io.Out(1)

  select_17.io.Select <> fused_select_cmp.io.Out(2)

  load_15.GepAddr <> address_14.io.Out(0)

  int_mul_16.io.LeftIO <> load_15.io.Out(0)

  select_17.io.InData1 <> int_mul_16.io.Out(0)

  int_add_18.io.RightIO <> select_17.io.Out(0)

  fused_mask_add.io.Out(1).ready := true.B
  fused_mask_add.io.Out(2).ready := true.B

  fused_select_cmp.io.Out(0).ready := true.B
  fused_select_cmp.io.Out(1).ready := true.B

  loop_0.io.CarryDepenOut.elements("field0")(3).ready := true.B
  loop_0.io.CarryDepenOut.elements("field2")(0).ready := true.B

  mem_ctrl_cache.io.load_address(0) <> load_10.address_out

  load_10.data_in <> mem_ctrl_cache.io.load_data(0)

  mem_ctrl_cache.io.load_address(1) <> load_15.address_out

  load_15.data_in <> mem_ctrl_cache.io.load_data(1)

  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_0.io.enable <> exe_block_0.io.Out(0)

  int_const_2.io.enable <> exe_block_0.io.Out(1)

  int_const_3.io.enable <> exe_block_0.io.Out(2)

  int_const_4.io.enable <> exe_block_0.io.Out(3)

  int_const_5.io.enable <> exe_block_0.io.Out(4)

  int_const_6.io.enable <> exe_block_0.io.Out(5)

  int_const_7.io.enable <> exe_block_0.io.Out(6)

  int_const_8.io.enable <> exe_block_0.io.Out(7)

  int_const_9.io.enable <> exe_block_0.io.Out(8)

  int_const_10.io.enable <> exe_block_0.io.Out(9)

  fused_mask_add.io.enable <> exe_block_0.io.Out(10)

  fused_select_cmp.io.enable <> exe_block_0.io.Out(11)

  io.out <> return_23.io.Out
}
