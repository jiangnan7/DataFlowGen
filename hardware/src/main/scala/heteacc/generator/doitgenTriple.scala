
//===------------------------------------------------------------*- Scala -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//

package heteacc.generator

import chipsalliance.rocketchip.config._
import chisel3._
import chisel3.util._
import chisel3.Module._
import chisel3.testers._
import chisel3.iotesters._


import heteacc.config._
import heteacc.fpu._
import heteacc.interfaces._
import heteacc.junctions._
import heteacc.memory._
import heteacc.node._
import heteacc.loop._
import heteacc.execution._
import utility._

abstract class doitgenTripleDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32, 32, 32))))
	  val out = Decoupled(new Call(List()))
	})
}

class doitgenTripleDF(implicit p: Parameters) extends doitgenTripleDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1, 1)))
  FineGrainedArgCall.io.In <> io.in

  val mem_ctrl_cache = Module(new MemoryEngine(Size = 272, ID = 0, NumRead = 2, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/doitgenTriple/in.txt")

  val mem_ctrl_cache_store = Module(new MemoryEngine(Size = 16, ID = 0, NumRead = 0, NumWrite = 1))
  mem_ctrl_cache_store.initMem("dataset/doitgenTriple/sum.txt")

  /* ================================================================== *
   *                   Const nodes.                                     *
   * ================================================================== */

  val int_const_1 = Module(new ConstFastNode(value = 0, ID = 2))

  val int_const_3 = Module(new ConstFastNode(value = 16, ID = 3))

  val int_const_4 = Module(new ConstFastNode(value = 0, ID = 4))

  val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

  val int_const_7 = Module(new ConstFastNode(value = 1, ID = 7))

  val int_const_8 = Module(new ConstFastNode(value = 16, ID = 8))

  /* ================================================================== *
   *                   Execution Block nodes.                           *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 5, NumPhi = 1, BID = 0))

  val exe_block_1 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 2, NumPhi = 0, BID = 1))

  /* ================================================================== *
   *                   Operation nodes.                                 *
   * ================================================================== */

  val state_branch_0 = Module(new UBranchNode(ID = 0))

  val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 2, Res = false))

  val address_4 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 4)(ElementSize = 1, ArraySize = List()))

  val load_5 = Module(new Load(NumOuts = 3, ID = 5, RouteID = 0))

  val int_mul_6 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 6, opCode = "Mul")(sign = false, Debug = false))

  val int_add_7 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 7, opCode = "Add")(sign = false, Debug = false))

  val address_8 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 8)(ElementSize = 1, ArraySize = List()))

  val load_9 = Module(new Load(NumOuts = 2, ID = 9, RouteID = 1))

  val int_cmp_10 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 10, opCode = "sgt")(sign = false, Debug = false))

  val int_mul_11 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 11, opCode = "Mul")(sign = false, Debug = false))

  val int_add_12 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 12, opCode = "Add")(sign = false, Debug = false))

  val int_mul_13 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 13, opCode = "Mul")(sign = false, Debug = false))

  val int_add_14 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 14, opCode = "Add")(sign = false, Debug = false))

  val select_15 = Module(new SelectNodeWithoutState(NumOuts = 2, ID = 15))

  val int_add_16 = Module(new ComputeNodeWithoutStateSupportCarry(NumOuts = 1, ID = 16, opCode = "Add")(sign = false, Debug = false))

  val address_19 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 19)(ElementSize = 1, ArraySize = List()))

  val store_20 = Module(new Store(NumOuts = 1, ID = 20, RouteID = 2))

  val int_add_21 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 21, opCode = "Add")(sign = false, Debug = false))

  val int_cmp_22 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 22, opCode = "eq")(sign = false, Debug = false))

  val state_branch_23 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 23))

  val return_24 = Module(new RetNode2(retTypes = List(), ID = 24))

  /* ================================================================== *
   *                   Loop nodes.                                      *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNodeExperimental(
    NumIns = List(1, 1, 1),
    NumOuts = List(1),
    NumCarry = List(1),
    NumExits = 1,
    ID = 0,
    LoopCounterMax = 16,
    LoopCounterStep = 1
  ))

  val loop_1 = Module(new LoopBlockNode(NumIns = List(1, 1, 1), NumOuts = List(), NumCarry = List(3), NumExits = 1, ID = 1))

  loop_1.io.loopExit(0) <> return_24.io.In.enable

  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back

  exe_block_1.io.predicateIn(0) <> loop_1.io.activate_loop_start

  exe_block_1.io.predicateIn(1) <> loop_1.io.activate_loop_back

  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> exe_block_1.io.Out(0)

  loop_1.io.enable <> state_branch_0.io.Out(0)

  loop_1.io.loopBack(0) <> state_branch_23.io.FalseOutput(0)

  loop_1.io.loopFinish(0) <> state_branch_23.io.TrueOutput(0)

  store_20.io.Out(0).ready := true.B

  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

  loop_0.io.InLiveIn(0) <> loop_1.io.OutLiveIn.elements("field1")(0)

  loop_0.io.InLiveIn(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)

  loop_0.io.InLiveIn(2) <> loop_1.io.OutLiveIn.elements("field2")(0)

  loop_1.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)

  loop_1.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

  loop_1.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)

  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  address_4.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  int_mul_6.io.RightIO <> loop_0.io.OutLiveIn.elements("field1")(0)

  address_8.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(0)

  address_19.io.baseAddress <> loop_1.io.OutLiveIn.elements("field0")(0)

  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> select_15.io.Out(0)

  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  loop_0.io.OutLiveOut.elements("field0")(0) <> store_20.inData

  store_20.GepAddr <> address_19.io.Out(0)

  /* ================================================================== *
   *                   Carry dependencies.                              *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> select_15.io.Out(1)

  merge_2.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  loop_1.io.CarryDepenIn(0) <> int_add_21.io.Out(1)

  /* ================================================================== *
   *                   Connections.                                     *
   * ================================================================== */

  merge_2.io.Mask <> exe_block_0.io.MaskBB(0)

  merge_2.io.InData(0) <> int_const_1.io.Out

  int_mul_6.io.LeftIO <> int_const_3.io.Out

  int_cmp_10.io.RightIO <> int_const_4.io.Out

  int_add_16.io.RightIO <> int_const_5.io.Out

  int_add_21.io.LeftIO <> int_const_7.io.Out

  int_cmp_22.io.LeftIO <> int_const_8.io.Out

  address_19.io.idx(0) <> loop_1.io.CarryDepenOut.elements("field0")(1)

  int_add_21.io.RightIO <> loop_1.io.CarryDepenOut.elements("field0")(2)

  int_cmp_22.io.RightIO <> int_add_21.io.Out(0)

  int_add_14.io.LeftIO <> merge_2.io.Out(0)

  select_15.io.InData2 <> merge_2.io.Out(1)

  address_4.io.idx(0) <> int_add_16.io.Out(0)

  int_add_7.io.LeftIO <> int_add_16.io.Out(0)

  load_5.GepAddr <> address_4.io.Out(0)

  int_cmp_10.io.LeftIO <> load_5.io.Out(0)

  int_mul_11.io.LeftIO <> load_5.io.Out(1)

  int_mul_13.io.RightIO <> load_5.io.Out(2)

  int_add_7.io.RightIO <> int_mul_6.io.Out(0)

  address_8.io.idx(0) <> int_add_7.io.Out(0)

  load_9.GepAddr <> address_8.io.Out(0)

  int_mul_11.io.RightIO <> load_9.io.Out(0)

  int_add_12.io.RightIO <> load_9.io.Out(1)

  select_15.io.Select <> int_cmp_10.io.Out(0)

  int_add_12.io.LeftIO <> int_mul_11.io.Out(0)

  int_mul_13.io.LeftIO <> int_add_12.io.Out(0)

  int_add_14.io.RightIO <> int_mul_13.io.Out(0)

  select_15.io.InData1 <> int_add_14.io.Out(0)

  state_branch_23.io.CmpIO <> int_cmp_22.io.Out(0)

  mem_ctrl_cache.io.load_address(0) <> load_5.address_out

  load_5.data_in <> mem_ctrl_cache.io.load_data(0)

  mem_ctrl_cache.io.load_address(1) <> load_9.address_out

  load_9.data_in <> mem_ctrl_cache.io.load_data(1)

  mem_ctrl_cache_store.io.store_address(0) <> store_20.address_out

  store_20.io.Out(0) <> mem_ctrl_cache_store.io.store_data(0)

  /* ================================================================== *
   *                   Execution Block Enable.                          *
   * ================================================================== */

  int_const_1.io.enable <> exe_block_0.io.Out(0)

  int_const_3.io.enable <> exe_block_0.io.Out(1)

  int_const_4.io.enable <> exe_block_0.io.Out(2)

  int_const_5.io.enable <> exe_block_0.io.Out(3)

  merge_2.io.enable <> exe_block_0.io.Out(4)

  int_const_7.io.enable <> exe_block_1.io.Out(0)

  int_const_8.io.enable <> exe_block_1.io.Out(1)

  state_branch_23.io.enable <> loop_0.io.loopExit(0)

  io.out <> return_24.io.Out
}
