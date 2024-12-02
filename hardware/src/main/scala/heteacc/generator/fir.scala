
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

abstract class firDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32, 32))))
	  val out = Decoupled(new Call(List(32)))
	})
}

class firDF(implicit p: Parameters) extends firDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1)))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  val mem_ctrl_cache = Module(new MemoryEngine(Size=200, ID = 0, NumRead = 2, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/fir/fir.txt")

  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c-1 = arith.constant -1 : index
  val int_const_0 = Module(new ConstFastNode(value = -1, ID = 0))

  //%c99 = arith.constant 99 : index
  val int_const_1 = Module(new ConstFastNode(value = 99, ID = 1))

  //%c1 = arith.constant 1 : index
  val int_const_2 = Module(new ConstFastNode(value = 1, ID = 2))

  //%c100 = arith.constant 100 : index
  val int_const_3 = Module(new ConstFastNode(value = 100, ID = 3))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 15, NumPhi = 0, BID = 0))



  /* ================================================================== *
   *                   Printing Operation nodes. 13                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%4 = dataflow.addr %arg1[%arg2] {memShape = [100]} : memref<100xi32>[index] -> i32
  val address_1 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 1)(ElementSize = 1, ArraySize = List()))

  //%5 = dataflow.load %4 : i32 -> i32
  val load_2 = Module(new Load(NumOuts = 1, ID = 2, RouteID = 0))

  //%6 = arith.muli %arg2, %c-1 : index
  val int_mul_3 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 3, opCode = "Mul")(sign = false, Debug = false))

  //%7 = arith.addi %6, %c99 : index
  val int_add_4 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 4, opCode = "Add")(sign = false, Debug = false))

  //%8 = dataflow.addr %arg0[%7] {memShape = [100]} : memref<100xi32>[index] -> i32
  val address_5 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 5)(ElementSize = 1, ArraySize = List()))

  //%9 = dataflow.load %8 : i32 -> i32
  val load_6 = Module(new Load(NumOuts = 1, ID = 6, RouteID = 1))

  //%10 = arith.muli %5, %9 : i32
  val int_mul_7 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 7, opCode = "Mul")(sign = false, Debug = false))

  //%11 = arith.addi %arg3, %10 : i32
  val int_add_8 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 8, opCode = "Add")(sign = false, Debug = false))

  //%12 = arith.addi %arg2, %c1 {Exe = "Loop"} : index
  val int_add_9 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 9, opCode = "Add")(sign = false, Debug = false))

  //%13 = arith.cmpi eq, %12, %c100 {Exe = "Loop"} : index
  val int_cmp_10 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 10, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %13, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_11 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 11))

  //func.return %0 : i32
  val return_12 = Module(new RetNode2(retTypes = List(32), ID = 12))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1, 1), NumOuts = List(1), NumCarry = List(3, 1), NumExits = 1, ID = 0))



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

  loop_0.io.loopBack(0) <> state_branch_11.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_11.io.TrueOutput(0)



  /* ================================================================== *
   *                   Loop dependencies.                               *
   * ================================================================== */



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

  loop_0.io.InLiveOut(0) <> int_add_8.io.Out(0)



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_12.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)



  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_9.io.Out(0)

  address_1.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  int_mul_3.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(1)

  int_add_9.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(2)

  loop_0.io.CarryDepenIn(1) <> int_add_8.io.Out(1)

  int_add_8.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field1")(0)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  int_mul_3.io.RightIO <> int_const_0.io.Out

  int_add_4.io.RightIO <> int_const_1.io.Out

  int_add_9.io.RightIO <> int_const_2.io.Out

  int_cmp_10.io.RightIO <> int_const_3.io.Out

  load_2.GepAddr <> address_1.io.Out(0)

  int_mul_7.io.LeftIO <> load_2.io.Out(0)

  int_add_4.io.LeftIO <> int_mul_3.io.Out(0)

  address_5.io.idx(0) <> int_add_4.io.Out(0)

  load_6.GepAddr <> address_5.io.Out(0)

  int_mul_7.io.RightIO <> load_6.io.Out(0)

  int_add_8.io.RightIO <> int_mul_7.io.Out(0)

  int_cmp_10.io.LeftIO <> int_add_9.io.Out(1)

  state_branch_11.io.CmpIO <> int_cmp_10.io.Out(0)

  mem_ctrl_cache.io.load_address(0) <> load_2.address_out

  load_2.data_in <> mem_ctrl_cache.io.load_data(0)

  mem_ctrl_cache.io.load_address(1) <> load_6.address_out

  load_6.data_in <> mem_ctrl_cache.io.load_data(1)



  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_0.io.enable <> exe_block_0.io.Out(0)

  int_const_1.io.enable <> exe_block_0.io.Out(1)

  int_const_2.io.enable <> exe_block_0.io.Out(2)

  int_const_3.io.enable <> exe_block_0.io.Out(3)

  address_1.io.enable <> exe_block_0.io.Out(4)

  load_2.io.enable <> exe_block_0.io.Out(5)

  int_mul_3.io.enable <> exe_block_0.io.Out(6)

  int_add_4.io.enable <> exe_block_0.io.Out(7)

  address_5.io.enable <> exe_block_0.io.Out(8)

  load_6.io.enable <> exe_block_0.io.Out(9)

  int_mul_7.io.enable <> exe_block_0.io.Out(10)

  int_add_8.io.enable <> exe_block_0.io.Out(11)

  int_add_9.io.enable <> exe_block_0.io.Out(12)

  int_cmp_10.io.enable <> exe_block_0.io.Out(13)

  state_branch_11.io.enable <> exe_block_0.io.Out(14)

  io.out <> return_12.io.Out

}

