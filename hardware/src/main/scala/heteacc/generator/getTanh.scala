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

abstract class getTanhDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new Call(List(32))))
    val out = Decoupled(new Call(List(32)))
  })
}

class getTanhDF(implicit p: Parameters) extends getTanhDFIO()(p) {

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1)))
  FineGrainedArgCall.io.In <> io.in

  val mem_ctrl_cache = Module(new MemoryEngine(Size = 100, ID = 0, NumRead = 1, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/getTanh/getTanh.txt")

  /* ================================================================== *
   *                   Const nodes.                                     *
   * ================================================================== */

  val int_const_0 = Module(new ConstFastNode(value = 1, ID = 0))

  val int_const_1 = Module(new ConstFastNode(value = 19, ID = 1))

  val int_const_2 = Module(new ConstFastNode(value = 3, ID = 2))

  val int_const_3 = Module(new ConstFastNode(value = 1, ID = 3))

  val int_const_4 = Module(new ConstFastNode(value = 1, ID = 4))

  /* ================================================================== *
   *                   Execution Block nodes.                           *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 6, NumPhi = 0, BID = 0))

  /* ================================================================== *
   *                   Operation nodes.                                 *
   * ================================================================== */

  val state_branch_0 = Module(new UBranchNode(ID = 0))

  val address_1 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 1)(ElementSize = 1, ArraySize = List()))

  val load_2 = Module(new Load(NumOuts = 6, ID = 2, RouteID = 0))

  val int_cmp_3 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 3, opCode = "slt")(sign = false, Debug = false))

  val m0 = Module(new Chain(NumOps = 6, ID = 0, OpCodes = Array("Mul", "Add", "Mul", "Mul", "Add", "Mul"))(sign = false)(p))

  val select_10 = Module(new SelectNodeWithoutState(NumOuts = 1, ID = 10))

  val int_add_11 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 11, opCode = "Add")(sign = false, Debug = false))

  val int_add_12 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 12, opCode = "Add")(sign = false, Debug = false))

  val return_15 = Module(new RetNode2(retTypes = List(32), ID = 15))

  /* ================================================================== *
   *                   Loop nodes.                                      *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNodeExperimental(
    NumIns = List(1),
    NumOuts = List(1),
    NumCarry = List(1, 2),
    NumExits = 1,
    ID = 0,
    LoopCounterMax = 100,
    LoopCounterStep = 1
  ))

  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_0.io.loopExit(0) <> return_15.io.In.enable

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

  loop_0.io.InLiveOut(0) <> int_add_11.io.Out(0)

  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_15.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)

  /* ================================================================== *
   *                   Carry dependencies.                              *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_11.io.Out(1)

  int_add_11.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(0)

  loop_0.io.CarryDepenIn(1) <> int_add_12.io.Out(0)

  address_1.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field1")(0)

  int_add_12.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field1")(1)

  /* ================================================================== *
   *                   Connections.                                     *
   * ================================================================== */

  int_cmp_3.io.RightIO <> int_const_0.io.Out

  select_10.io.InData2 <> int_const_3.io.Out

  int_add_12.io.RightIO <> int_const_4.io.Out

  load_2.GepAddr <> address_1.io.Out(0)

  int_cmp_3.io.LeftIO <> load_2.io.Out(0)

  m0.io.In(0) <> load_2.io.Out(1)

  m0.io.In(1) <> load_2.io.Out(2)

  m0.io.In(2) <> int_const_1.io.Out

  m0.io.In(3) <> load_2.io.Out(3)

  m0.io.In(4) <> load_2.io.Out(4)

  m0.io.In(5) <> int_const_2.io.Out

  m0.io.In(6) <> load_2.io.Out(5)

  select_10.io.Select <> int_cmp_3.io.Out(0)

  select_10.io.InData1 <> m0.io.Out(6)

  int_add_11.io.RightIO <> select_10.io.Out(0)

  mem_ctrl_cache.io.load_address(0) <> load_2.address_out

  load_2.data_in <> mem_ctrl_cache.io.load_data(0)

  for (i <- 0 until 6) {
    m0.io.Out(i).ready := true.B
  }

  /* ================================================================== *
   *                   Execution Block Enable.                          *
   * ================================================================== */

  int_const_0.io.enable <> exe_block_0.io.Out(0)

  int_const_1.io.enable <> exe_block_0.io.Out(1)

  int_const_2.io.enable <> exe_block_0.io.Out(2)

  int_const_3.io.enable <> exe_block_0.io.Out(3)

  int_const_4.io.enable <> exe_block_0.io.Out(4)

  m0.io.enable <> exe_block_0.io.Out(5)

  io.out <> return_15.io.Out
}
