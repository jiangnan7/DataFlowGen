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

abstract class histogramDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32, 32, 32))))
	  val out = Decoupled(new Call(List()))
	})
}

class histogramDF(implicit p: Parameters) extends histogramDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1, 1)))
  FineGrainedArgCall.io.In <> io.in

  val mem_ctrl_cache = Module(new MemoryEngine(Size = 200, ID = 0, NumRead = 2, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/histogram/in.txt")

  val mem_ctrl_cache_out = Module(new MemoryEngine(Size = 100, ID = 0, NumRead = 1, NumWrite = 1))
  mem_ctrl_cache_out.initMem("dataset/histogram/out.txt")

  /* ================================================================== *
   *                   Const nodes.                                     *
   * ================================================================== */

  val int_const_0 = Module(new ConstFastNode(value = 1, ID = 0))

  /* ================================================================== *
   *                   Execution Block nodes.                           *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 2, NumPhi = 0, BID = 0))

  /* ================================================================== *
   *                   Operation nodes.                                 *
   * ================================================================== */

  val state_branch_0 = Module(new UBranchNode(ID = 0))

  val address_1 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 1)(ElementSize = 1, ArraySize = List()))

  val load_2 = Module(new Load(NumOuts = 1, ID = 2, RouteID = 0))

  val address_3 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 3)(ElementSize = 1, ArraySize = List()))

  val load_4 = Module(new Load(NumOuts = 1, ID = 4, RouteID = 1))

  val cast_5 = Module(new BitCastNode(NumOuts = 2, ID = 5))

  val address_6 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 6)(ElementSize = 1, ArraySize = List()))

  val load_7 = Module(new Load(NumOuts = 1, ID = 7, RouteID = 2))

  val int_add_8 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 8, opCode = "Add")(sign = false, Debug = false))

  val address_9 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 9)(ElementSize = 1, ArraySize = List()))

  val store_10 = Module(new Store(NumOuts = 1, ID = 10, RouteID = 3))

  val int_add_11 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 11, opCode = "Add")(sign = false, Debug = false))

  val return_14 = Module(new RetNode2(retTypes = List(), ID = 14))

  /* ================================================================== *
   *                   Loop nodes.                                      *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNodeExperimental(
    NumIns = List(1, 1, 2),
    NumOuts = List(),
    NumCarry = List(3),
    NumExits = 1,
    ID = 0,
    LoopCounterMax = 100,
    LoopCounterStep = 1
  ))

  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_0.io.loopExit(0) <> return_14.io.In.enable

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

  loop_0.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)

  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  address_1.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  address_3.io.baseAddress <> loop_0.io.OutLiveIn.elements("field1")(0)

  address_6.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(0)

  address_9.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(1)

  /* ================================================================== *
   *                   Carry dependencies.                              *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_11.io.Out(0)

  address_1.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  address_3.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(1)

  int_add_11.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(2)

  /* ================================================================== *
   *                   Connections.                                     *
   * ================================================================== */

  int_add_11.io.RightIO <> int_const_0.io.Out

  load_2.GepAddr <> address_1.io.Out(0)

  int_add_8.io.RightIO <> load_2.io.Out(0)

  load_4.GepAddr <> address_3.io.Out(0)

  cast_5.io.Input <> load_4.io.Out(0)

  address_6.io.idx(0) <> cast_5.io.Out(0)

  address_9.io.idx(0) <> cast_5.io.Out(1)

  load_7.GepAddr <> address_6.io.Out(0)

  int_add_8.io.LeftIO <> load_7.io.Out(0)

  store_10.inData <> int_add_8.io.Out(0)

  store_10.GepAddr <> address_9.io.Out(0)

  mem_ctrl_cache.io.load_address(0) <> load_2.address_out

  load_2.data_in <> mem_ctrl_cache.io.load_data(0)

  mem_ctrl_cache.io.load_address(1) <> load_4.address_out

  load_4.data_in <> mem_ctrl_cache.io.load_data(1)

  mem_ctrl_cache_out.io.load_address(0) <> load_7.address_out

  load_7.data_in <> mem_ctrl_cache_out.io.load_data(0)

  mem_ctrl_cache_out.io.store_address(0) <> store_10.address_out

  store_10.io.Out(0) <> mem_ctrl_cache_out.io.store_data(0)

  /* ================================================================== *
   *                   Execution Block Enable.                          *
   * ================================================================== */

  int_const_0.io.enable <> exe_block_0.io.Out(0)

  cast_5.io.enable <> exe_block_0.io.Out(1)

  io.out <> return_14.io.Out
}
