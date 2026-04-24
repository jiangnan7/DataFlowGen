
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

abstract class if_loop_1DFIO(implicit val p: Parameters) extends Module with HasAccelParams {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new Call(List(32))))
    val out = Decoupled(new Call(List(32)))
  })
}

class if_loop_1DF(implicit p: Parameters) extends if_loop_1DFIO()(p) {

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1)))
  FineGrainedArgCall.io.In <> io.in

  val mem_ctrl_cache = Module(new MemoryEngine(Size = 100, ID = 0, NumRead = 1, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/if_loop_1/if_loop_1.txt")

  /* ================================================================== *
   *                   Const nodes.                                     *
   * ================================================================== */

  val int_const_0 = Module(new ConstFastNode(value = 2, ID = 0))

  val int_const_1 = Module(new ConstFastNode(value = 10, ID = 1))

  val int_const_2 = Module(new ConstFastNode(value = 1, ID = 2))

  /* ================================================================== *
   *                   Execution Block nodes.                           *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 3, NumPhi = 0, BID = 0))

  /* ================================================================== *
   *                   Operation nodes.                                 *
   * ================================================================== */

  val state_branch_0 = Module(new UBranchNode(ID = 0))

  val address_1 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 1)(ElementSize = 1, ArraySize = List()))

  val load_2 = Module(new Load(NumOuts = 1, ID = 2, RouteID = 0))

  val int_mul_3 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 3, opCode = "Mul")(sign = false, Debug = false))

  val int_cmp_4 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 4, opCode = "ugt")(sign = false, Debug = false))

  val int_add_5 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 5, opCode = "Add")(sign = false, Debug = false))

  val select_6 = Module(new SelectNodeWithoutState(NumOuts = 2, ID = 6))

  val int_add_7 = Module(new ComputeNodeWithoutStateSupportCarry(NumOuts = 1, ID = 7, opCode = "Add")(sign = false, Debug = false))

  val return_10 = Module(new RetNode2(retTypes = List(32), ID = 10))

  /* ================================================================== *
   *                   Loop nodes.                                      *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNodeExperimental(NumIns = List(1), NumOuts = List(1), NumCarry = List(2), NumExits = 1, ID = 0, LoopCounterMax = 100, LoopCounterStep = 1))

  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_0.io.loopExit(0) <> return_10.io.In.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back

  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> state_branch_0.io.Out(0)

  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

  loop_0.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  address_1.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> select_6.io.Out(0)

  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_10.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)

  /* ================================================================== *
   *                   Carry dependencies.                              *
   * ================================================================== */

  address_1.io.idx(0) <> int_add_7.io.Out(0)

  loop_0.io.CarryDepenIn(0) <> select_6.io.Out(1)

  int_add_5.io.RightIO <> loop_0.io.CarryDepenOut.elements("field0")(0)

  select_6.io.InData2 <> loop_0.io.CarryDepenOut.elements("field0")(1)

  /* ================================================================== *
   *                   Connections.                                     *
   * ================================================================== */

  int_mul_3.io.RightIO <> int_const_0.io.Out

  int_cmp_4.io.RightIO <> int_const_1.io.Out

  int_add_7.io.RightIO <> int_const_2.io.Out

  load_2.GepAddr <> address_1.io.Out(0)

  int_mul_3.io.LeftIO <> load_2.io.Out(0)

  int_cmp_4.io.LeftIO <> int_mul_3.io.Out(0)

  int_add_5.io.LeftIO <> int_mul_3.io.Out(1)

  select_6.io.Select <> int_cmp_4.io.Out(0)

  select_6.io.InData1 <> int_add_5.io.Out(0)

  mem_ctrl_cache.io.load_address(0) <> load_2.address_out

  load_2.data_in <> mem_ctrl_cache.io.load_data(0)

  /* ================================================================== *
   *                   Execution Block Enable.                          *
   * ================================================================== */

  int_const_0.io.enable <> exe_block_0.io.Out(0)

  int_const_1.io.enable <> exe_block_0.io.Out(1)

  int_const_2.io.enable <> exe_block_0.io.Out(2)

  io.out <> return_10.io.Out

}
