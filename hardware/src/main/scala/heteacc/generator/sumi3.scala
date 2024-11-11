
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

abstract class sumDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32))))
	  val MemResp = Flipped(Valid(new MemResp))
	  val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List(32)))
	})
}

class sumDF(implicit p: Parameters) extends sumDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1 )))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 1, NumWrite = 0))

  io.MemReq <> mem_ctrl_cache.io.cache.MemReq
  mem_ctrl_cache.io.cache.MemResp <> io.MemResp



  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c0_i32 = arith.constant 0 : i32
  val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c0 = arith.constant 0 : index
  // val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

  //%c1 = arith.constant 1 : index
  val int_const_2 = Module(new ConstFastNode(value = 1, ID = 2))

  //%c1000 = arith.constant 1000 : index
  val int_const_3 = Module(new ConstFastNode(value = 100, ID = 3))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 12, NumPhi = 1, BID = 0))



  /* ================================================================== *
   *                   Printing Operation nodes. 12                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%4 = dataflow.merge %c0_i32 or %arg2 {Select = "Loop_Signal"} : i32
  val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 1, ID = 1, Res = false))

  //%5 = dataflow.merge %c0 or %arg1 {Select = "Loop_Signal"} : index
  // val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 2, Res = false))

  //%6 = dataflow.addr %arg0[%5] {memShape = [1000]} : memref<1000xi32>[index] -> i32
  val address_3 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 3)(ElementSize = 32, ArraySize = List()))

  //%7 = dataflow.load %6 : i32 -> i32
  val load_4 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 3, ID = 4, RouteID = 0))

  //%8 = arith.muli %7, %7 : i32
  val int_mul_5 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 5, opCode = "Mul")(sign = false, Debug = false))

  //%9 = arith.muli %8, %7 : i32
  val int_mul_6 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 6, opCode = "Mul")(sign = false, Debug = false))

  //%10 = arith.addi %4, %9 : i32
  val int_add_7 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 7, opCode = "Add")(sign = false, Debug = false))

  //%11 = arith.addi %5, %c1 {Exe = "Loop"} : index
  val int_add_8 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 8, opCode = "Add")(sign = false, Debug = false))

  //%12 = arith.cmpi eq, %11, %c1000 {Exe = "Loop"} : index
  val int_cmp_9 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 9, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %12, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_10 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 10))

  //func.return %0 : i32
  val return_11 = Module(new RetNode2(retTypes = List(32), ID = 11))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1), NumOuts = List(1), NumCarry = List(1, 2), NumExits = 1, ID = 0))



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

  loop_0.io.loopBack(0) <> state_branch_10.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_10.io.TrueOutput(0)



  /* ================================================================== *
   *                   Loop dependencies.                               *
   * ================================================================== */



  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

  loop_0.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)



  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  address_3.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> int_add_7.io.Out(0)



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_11.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)



  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_7.io.Out(1)

  merge_1.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  loop_0.io.CarryDepenIn(1) <> int_add_8.io.Out(1)

  // merge_2.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(0)

    address_3.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field1")(0)

  int_add_8.io.RightIO <> loop_0.io.CarryDepenOut.elements("field1")(1)

  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  merge_1.io.Mask <> exe_block_0.io.MaskBB(0)

  // merge_2.io.Mask <> exe_block_0.io.MaskBB(1)

  merge_1.io.InData(0) <> int_const_0.io.Out

  // merge_2.io.InData(0) <> int_const_1.io.Out

  int_add_8.io.LeftIO <> int_const_2.io.Out

  int_cmp_9.io.LeftIO <> int_add_8.io.Out(0)

  int_add_7.io.LeftIO <> merge_1.io.Out(0)

  // address_3.io.idx(0) <> merge_2.io.Out(0)

  // int_add_8.io.RightIO <> merge_2.io.Out(1)

  load_4.io.GepAddr <> address_3.io.Out(0)

  int_mul_5.io.LeftIO <> load_4.io.Out(0)

  int_mul_5.io.RightIO <> load_4.io.Out(1)

  int_mul_6.io.RightIO <> load_4.io.Out(2)

  int_mul_6.io.LeftIO <> int_mul_5.io.Out(0)

  int_add_7.io.RightIO <> int_mul_6.io.Out(0)

  int_cmp_9.io.RightIO <> int_const_3.io.Out

  state_branch_10.io.CmpIO <> int_cmp_9.io.Out(0)

  mem_ctrl_cache.io.rd.mem(0).MemReq <> load_4.io.MemReq

  load_4.io.MemResp <> mem_ctrl_cache.io.rd.mem(0).MemResp



  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_0.io.enable <> exe_block_0.io.Out(0)

  // int_const_1.io.enable <> exe_block_0.io.Out(1)

  int_const_2.io.enable <> exe_block_0.io.Out(2)

  int_const_3.io.enable <> exe_block_0.io.Out(3)

  merge_1.io.enable <> exe_block_0.io.Out(4)

  // merge_2.io.enable <> exe_block_0.io.Out(5)

  address_3.io.enable <> exe_block_0.io.Out(6)

  load_4.io.enable <> exe_block_0.io.Out(7)

  int_mul_5.io.enable <> exe_block_0.io.Out(8)

  int_mul_6.io.enable <> exe_block_0.io.Out(9)

  int_add_7.io.enable <> exe_block_0.io.Out(10)

  int_add_8.io.enable <> exe_block_0.io.Out(11)

  int_cmp_9.io.enable <> exe_block_0.io.Out(1)

  state_branch_10.io.enable <> exe_block_0.io.Out(5)

  io.out <> return_11.io.Out

}



