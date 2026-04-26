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
   *                   Const nodes.                                     *
   * ================================================================== */

  val int_const_0 = Module(new ConstFastNode(value = 7, ID = 0))

  val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

  val int_const_2 = Module(new ConstFastNode(value = 7, ID = 2))

  val int_const_3 = Module(new ConstFastNode(value = 0, ID = 3))

  val int_const_4 = Module(new ConstFastNode(value = 3, ID = 4))

  val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

  val int_const_6 = Module(new ConstFastNode(value = 0, ID = 6))

  val int_const_7 = Module(new ConstFastNode(value = 2, ID = 7))

  val int_const_8 = Module(new ConstFastNode(value = 0, ID = 8))

  val int_const_9 = Module(new ConstFastNode(value = 1, ID = 9))

  val int_const_10 = Module(new ConstFastNode(value = 1, ID = 10))

  /* ================================================================== *
   *                   Execution Block nodes.                           *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 14, NumPhi = 0, BID = 0))
  /* ================================================================== *
   *                   Operation nodes.                                 *
   * ================================================================== */

  val state_branch_0 = Module(new UBranchNode(ID = 0))

  val int_andi_2 = Module(new ComputeNodeWithoutState(NumOuts = 3, ID = 2, opCode = "And")(sign = false, Debug = false))

  val int_cmp_3 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 3, opCode = "slt")(sign = false, Debug = false))

  val int_add_4 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 4, opCode = "Add")(sign = false, Debug = false))

  val select_5 = Module(new SelectNode(NumOuts = 1, ID = 5))

  val int_cmp_6 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 6, opCode = "eq")(sign = false, Debug = false))

  val int_lshr_7 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 7, opCode = "lshr")(sign = false, Debug = false))

  val address_9 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 9)(ElementSize = 1, ArraySize = List()))

  val load_10 = Module(new Load(NumOuts = 1, ID = 10, RouteID = 0))

  val select_11 = Module(new SelectNode(NumOuts = 2, ID = 11))

  val int_andi_12 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 12, opCode = "and")(sign = false, Debug = false))

  val int_cmp_13 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 13, opCode = "ne")(sign = false, Debug = false))

  val address_14 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 14)(ElementSize = 1, ArraySize = List()))

  val load_15 = Module(new Load(NumOuts = 1, ID = 15, RouteID = 1))

  val int_mul_16 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 16, opCode = "Mul")(sign = false, Debug = false))

  val select_17 = Module(new SelectNode(NumOuts = 1, ID = 17))

  val int_add_18 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 18, opCode = "Add")(sign = false, Debug = false))

  val int_shr_19 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 19, opCode = "ashr")(sign = false, Debug = false))

  val int_add_20 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 20, opCode = "Add")(sign = false, Debug = false))

  val return_23 = Module(new RetNode2(retTypes = List(32), ID = 23))

  /* ================================================================== *
   *                   Loop nodes.                                      *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNodeExperimental(
    NumIns = List(1, 1),
    NumOuts = List(1),
    NumCarry = List(4, 1, 1),
    NumExits = 1,
    ID = 0,
    LoopCounterMax = 1024,
    LoopCounterStep = 1
  ))

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

  int_andi_2.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(1)

  address_14.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(2)

  int_add_20.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(3)

  loop_0.io.CarryDepenIn(1) <> int_shr_19.io.Out(0)

  select_11.io.InData2 <> loop_0.io.CarryDepenOut.elements("field1")(0)

  loop_0.io.CarryDepenIn(2) <> int_add_18.io.Out(1)

  int_add_18.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field2")(0)

  /* ================================================================== *
   *                   Connections.                                     *
   * ================================================================== */

  int_andi_2.io.RightIO <> int_const_0.io.Out

  int_cmp_3.io.RightIO <> int_const_1.io.Out

  int_add_4.io.RightIO <> int_const_2.io.Out

  int_cmp_6.io.RightIO <> int_const_3.io.Out

  int_lshr_7.io.RightIO <> int_const_4.io.Out

  int_andi_12.io.RightIO <> int_const_5.io.Out

  int_cmp_13.io.RightIO <> int_const_6.io.Out

  int_mul_16.io.RightIO <> int_const_7.io.Out

  select_17.io.InData2 <> int_const_8.io.Out

  int_shr_19.io.RightIO <> int_const_9.io.Out

  int_add_20.io.RightIO <> int_const_10.io.Out

  int_lshr_7.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(0)

  int_cmp_3.io.LeftIO <> int_andi_2.io.Out(0)

  int_add_4.io.LeftIO <> int_andi_2.io.Out(1)

  select_5.io.InData2 <> int_andi_2.io.Out(2)

  select_5.io.Select <> int_cmp_3.io.Out(0)

  select_5.io.InData1 <> int_add_4.io.Out(0)

  int_cmp_6.io.LeftIO <> select_5.io.Out(0)

  select_11.io.Select <> int_cmp_6.io.Out(0)

  address_9.io.idx(0) <> int_lshr_7.io.Out(0)

  load_10.GepAddr <> address_9.io.Out(0)

  select_11.io.InData1 <> load_10.io.Out(0)

  int_andi_12.io.LeftIO <> select_11.io.Out(0)

  int_shr_19.io.LeftIO <> select_11.io.Out(1)

  int_cmp_13.io.LeftIO <> int_andi_12.io.Out(0)

  select_17.io.Select <> int_cmp_13.io.Out(0)

  load_15.GepAddr <> address_14.io.Out(0)

  int_mul_16.io.LeftIO <> load_15.io.Out(0)

  select_17.io.InData1 <> int_mul_16.io.Out(0)

  int_add_18.io.RightIO <> select_17.io.Out(0)

  mem_ctrl_cache.io.load_address(0) <> load_10.address_out

  load_10.data_in <> mem_ctrl_cache.io.load_data(0)

  mem_ctrl_cache.io.load_address(1) <> load_15.address_out

  load_15.data_in <> mem_ctrl_cache.io.load_data(1)

  /* ================================================================== *
   *                   Execution Block Enable.                          *
   * ================================================================== */

  int_const_0.io.enable <> exe_block_0.io.Out(0)

  int_const_1.io.enable <> exe_block_0.io.Out(1)

  int_const_2.io.enable <> exe_block_0.io.Out(2)

  int_const_3.io.enable <> exe_block_0.io.Out(3)

  int_const_4.io.enable <> exe_block_0.io.Out(4)

  int_const_5.io.enable <> exe_block_0.io.Out(5)

  int_const_6.io.enable <> exe_block_0.io.Out(6)

  int_const_7.io.enable <> exe_block_0.io.Out(7)

  int_const_8.io.enable <> exe_block_0.io.Out(8)

  int_const_9.io.enable <> exe_block_0.io.Out(9)

  int_const_10.io.enable <> exe_block_0.io.Out(10)

  select_5.io.enable <> exe_block_0.io.Out(11)

  select_11.io.enable <> exe_block_0.io.Out(12)

  select_17.io.enable <> exe_block_0.io.Out(13)

  io.out <> return_23.io.Out
}
