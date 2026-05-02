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

abstract class firDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new Call(List(32, 32))))
    val out = Decoupled(new Call(List(32)))
  })
}

class firDF(implicit p: Parameters) extends firDFIO()(p) {

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1)))
  FineGrainedArgCall.io.In <> io.in

  val mem_ctrl_cache = Module(new MemoryEngine(Size = 200, ID = 0, NumRead = 2, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/fir/fir.txt")

  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c99 = arith.constant 99 : index
  val int_const_1 = Module(new ConstFastNode(value = 99, ID = 1))

  //%c1 = arith.constant 1 : index
  val int_const_2 = Module(new ConstFastNode(value = 1, ID = 2))

  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 3, NumPhi = 0, BID = 0))

  /* ================================================================== *
   *                   Printing Operation nodes.                        *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%1 = dataflow.addr %arg0[%arg2] {memShape = [100]} : memref<100xi32>[index] -> i32
  val address_1 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 1)(ElementSize = 1, ArraySize = List()))

  //%2 = dataflow.load %1 {ID = 0 : i32} : i32 -> i32
  val load_2 = Module(new Load(NumOuts = 1, ID = 2, RouteID = 0))

  //%4 = arith.subi %c99, %arg2 : index
  val int_sub_4 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 4, opCode = "Sub")(sign = false, Debug = false))

  //%5 = dataflow.addr %arg1[%4] {memShape = [100]} : memref<100xi32>[index] -> i32
  val address_5 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 5)(ElementSize = 1, ArraySize = List()))

  //%6 = dataflow.load %5 {ID = 1 : i32} : i32 -> i32
  val load_6 = Module(new Load(NumOuts = 1, ID = 6, RouteID = 1))

  //%7 = arith.muli %2, %6 ; %8 = arith.addi %7, %arg3 : i32
  val fused_mac = Module(new Chain(ID = 7, NumOps = 2, OpCodes = Array("Mul", "Add"))(sign = false)(p))

  //%9 = arith.addi %arg2, %c1 {Exe = "Loop"} : index
  val int_add_9 = Module(new ComputeNodeWithoutStateSupportCarry(NumOuts = 1, ID = 9, opCode = "Add")(sign = false, Debug = false))

  //func.return %0 : i32
  val return_12 = Module(new RetNode2(retTypes = List(32), ID = 12))

  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNodeExperimental(NumIns = List(1, 1), NumOuts = List(1), NumCarry = List(2, 1), NumExits = 1, ID = 0, LoopCounterMax = 100, LoopCounterStep = 1))

  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_0.io.loopExit(0) <> return_12.io.In.enable

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

  address_1.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  address_5.io.baseAddress <> loop_0.io.OutLiveIn.elements("field1")(0)

  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> fused_mac.io.Out(2)

  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_12.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)

  /* ================================================================== *
   *                   Carry dependencies.                              *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_9.io.Out(0)

  address_1.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  int_sub_4.io.RightIO <> loop_0.io.CarryDepenOut.elements("field0")(1)

  loop_0.io.CarryDepenIn(1) <> fused_mac.io.Out(2)

  fused_mac.io.In(2) <> loop_0.io.CarryDepenOut.elements("field1")(0)

  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  int_sub_4.io.LeftIO <> int_const_1.io.Out

  int_add_9.io.RightIO <> int_const_2.io.Out

  load_2.GepAddr <> address_1.io.Out(0)

  fused_mac.io.In(0) <> load_2.io.Out(0)

  address_5.io.idx(0) <> int_sub_4.io.Out(0)

  load_6.GepAddr <> address_5.io.Out(0)

  fused_mac.io.In(1) <> load_6.io.Out(0)

  fused_mac.io.Out(0).ready := true.B
  fused_mac.io.Out(1).ready := true.B

  loop_0.io.CarryDepenOut.elements("field0")(1).ready := true.B

  mem_ctrl_cache.io.load_address(0) <> load_2.address_out

  load_2.data_in <> mem_ctrl_cache.io.load_data(0)

  mem_ctrl_cache.io.load_address(1) <> load_6.address_out

  load_6.data_in <> mem_ctrl_cache.io.load_data(1)

  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_1.io.enable <> exe_block_0.io.Out(0)

  int_const_2.io.enable <> exe_block_0.io.Out(1)

  fused_mac.io.enable <> exe_block_0.io.Out(2)

  io.out <> return_12.io.Out
}
