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

abstract class matrix_addDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new Call(List(32, 32, 32, 32))))
    val out = Decoupled(new Call(List(32)))
  })
}

class matrix_addDF(implicit p: Parameters) extends matrix_addDFIO()(p) {

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1, 1, 1)))
  FineGrainedArgCall.io.In <> io.in

  val mem_ctrl_cache = Module(new MemoryEngine(Size = 200, ID = 0, NumRead = 4, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/matirx_add/in.txt")

  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c1 = arith.constant 1 : index
  val int_const_1 = Module(new ConstFastNode(value = 1, ID = 1))

  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */
  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 2, NumPhi = 0, BID = 0))

  /* ================================================================== *
   *                   Printing Operation nodes.                        *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%1 = dataflow.addr %arg0[%arg5] {memShape = [50]} : memref<50xi32>[index] -> i32
  val address_1 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 1)(ElementSize = 1, ArraySize = List()))
  //%2 = dataflow.addr %arg1[%arg5] {memShape = [50]} : memref<50xi32>[index] -> i32
  val address_2 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 2)(ElementSize = 1, ArraySize = List()))
  //%3 = dataflow.addr %arg2[%arg5] {memShape = [50]} : memref<50xi32>[index] -> i32
  val address_3 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 3)(ElementSize = 1, ArraySize = List()))
  //%4 = dataflow.addr %arg3[%arg5] {memShape = [50]} : memref<50xi32>[index] -> i32
  val address_4 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 4)(ElementSize = 1, ArraySize = List()))

  //%5 = dataflow.load %1 {ID = 0 : i32} : i32 -> i32
  val load_1 = Module(new Load(NumOuts = 1, ID = 5, RouteID = 0))
  //%6 = dataflow.load %2 {ID = 1 : i32} : i32 -> i32
  val load_2 = Module(new Load(NumOuts = 1, ID = 6, RouteID = 1))
  //%7 = dataflow.load %3 {ID = 2 : i32} : i32 -> i32
  val load_3 = Module(new Load(NumOuts = 1, ID = 7, RouteID = 2))
  //%8 = dataflow.load %4 {ID = 3 : i32} : i32 -> i32
  val load_4 = Module(new Load(NumOuts = 1, ID = 8, RouteID = 3))

  //%9 = arith.addi %arg5, %c1 {Exe = "Loop"} : index
  val int_add_9 = Module(new ComputeNodeWithoutStateSupportCarry(NumOuts = 1, ID = 9, opCode = "Add")(sign = false, Debug = false))
  //%10 = arith.addi %5, %6 ; %11 = arith.addi ... ; %12 = arith.addi ... ; %13 = arith.addi ... : i32
  val sum_4 = Module(new Chain(ID = 10, NumOps = 4, OpCodes = Array("Add", "Add", "Add", "Add"))(sign = false)(p))
  //func.return %0 : i32
  val return_11 = Module(new RetNode2(retTypes = List(32), ID = 11))

  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNodeExperimental(NumIns = List(1, 1, 1, 1), NumOuts = List(1), NumCarry = List(2, 1), NumExits = 1, ID = 0, LoopCounterMax = 50, LoopCounterStep = 1))

  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_0.io.loopExit(0) <> return_11.io.In.enable

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
  loop_0.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)
  loop_0.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)
  loop_0.io.InLiveIn(3) <> FineGrainedArgCall.io.Out.data.elements("field3")(0)

  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  address_1.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)
  address_2.io.baseAddress <> loop_0.io.OutLiveIn.elements("field1")(0)
  address_3.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(0)
  address_4.io.baseAddress <> loop_0.io.OutLiveIn.elements("field3")(0)

  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_9.io.Out(0)
  address_1.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(0)
  address_2.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(0)
  address_3.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(0)
  address_4.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(0)
  int_add_9.io.RightIO <> int_const_1.io.Out

  loop_0.io.CarryDepenIn(1) <> sum_4.io.Out(4)
  sum_4.io.In(4) <> loop_0.io.CarryDepenOut.elements("field1")(0)

  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> sum_4.io.Out(4)

  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_11.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)

  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  load_1.GepAddr <> address_1.io.Out(0)
  load_2.GepAddr <> address_2.io.Out(0)
  load_3.GepAddr <> address_3.io.Out(0)
  load_4.GepAddr <> address_4.io.Out(0)

  sum_4.io.In(0) <> load_1.io.Out(0)
  sum_4.io.In(1) <> load_2.io.Out(0)
  sum_4.io.In(2) <> load_3.io.Out(0)
  sum_4.io.In(3) <> load_4.io.Out(0)

  for (i <- 0 until 4) {
    sum_4.io.Out(i).ready := true.B
  }
  loop_0.io.CarryDepenOut.elements("field0")(1).ready := true.B

  mem_ctrl_cache.io.load_address(0) <> load_1.address_out
  load_1.data_in <> mem_ctrl_cache.io.load_data(0)
  mem_ctrl_cache.io.load_address(1) <> load_2.address_out
  load_2.data_in <> mem_ctrl_cache.io.load_data(1)
  mem_ctrl_cache.io.load_address(2) <> load_3.address_out
  load_3.data_in <> mem_ctrl_cache.io.load_data(2)
  mem_ctrl_cache.io.load_address(3) <> load_4.address_out
  load_4.data_in <> mem_ctrl_cache.io.load_data(3)

  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_1.io.enable <> exe_block_0.io.Out(0)
  sum_4.io.enable <> exe_block_0.io.Out(1)

  io.out <> return_11.io.Out
}
