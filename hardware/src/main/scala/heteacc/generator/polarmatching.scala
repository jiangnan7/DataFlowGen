
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

abstract class polarmatchingDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32, 32))))
	  val MemResp = Flipped(Valid(new MemResp))
	  val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List()))
	})
}

class polarmatchingDF(implicit p: Parameters) extends polarmatchingDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1 )))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 1, NumWrite = 1))

  io.MemReq <> mem_ctrl_cache.io.cache.MemReq
  mem_ctrl_cache.io.cache.MemResp <> io.MemResp



  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c0 = arith.constant 0 : index
  // val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

  //%c0 = arith.constant 0 : index
  val int_const_2 = Module(new ConstFastNode(value = 0, ID = 2))

  //%c64_i32 = arith.constant 64 : i32
  val int_const_3 = Module(new ConstFastNode(value = 64, ID = 3))

  //%c96_i32 = arith.constant 96 : i32
  val int_const_4 = Module(new ConstFastNode(value = 96, ID = 4))

  //%c1 = arith.constant 1 : index
  val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

  //%c2 = arith.constant 2 : index
  val int_const_6 = Module(new ConstFastNode(value = 2, ID = 6))

  //%c2_i32 = arith.constant 2 : i32
  val int_const_7 = Module(new ConstFastNode(value = 2, ID = 7))

  //%c1 = arith.constant 1 : index
  val int_const_8 = Module(new ConstFastNode(value = 1, ID = 8))

  //%c64 = arith.constant 64 : index
  val int_const_9 = Module(new ConstFastNode(value = 64, ID = 9))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 20, NumPhi = 2, BID = 0))

  val exe_block_1 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 10, NumPhi = 0, BID = 1))

  /* ================================================================== *
   *                   Printing Operation nodes. 25                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%0 = dataflow.merge %c0 or %arg2 {Select = "Loop_Signal"} : index
  // val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 3, ID = 1, Res = false))

  //%1 = arith.index_cast %0 : index to i32
  val cast_2 = Module(new BitCastNode(NumOuts = 1, ID = 2))

  //%8 = dataflow.merge %c0_i32 or %arg4 {Select = "Loop_Signal"} : i32
  val merge_3 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 3, Res = false))

  //%9 = dataflow.merge %c0 or %arg3 {Select = "Loop_Signal"} : index
  val merge_4 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 4, Res = false))

  //%10 = arith.index_cast %9 : index to i32
  val cast_5 = Module(new BitCastNode(NumOuts = 1, ID = 5))

  //%11 = arith.muli %10, %c64_i32 : i32
  val int_mul_6 = Module(new ComputeNode(NumOuts = 1, ID = 6, opCode = "Mul")(sign = false, Debug = false))

  //%12 = arith.addi %1, %11 : i32
  val int_add_7 = Module(new ComputeNode(NumOuts = 2, ID = 7, opCode = "Add")(sign = false, Debug = false))

  //%13 = arith.cmpi slt, %12, %c96_i32 : i32
  val int_cmp_8 = Module(new ComputeNode(NumOuts = 1, ID = 8, opCode = "slt")(sign = false, Debug = false))

  //%14 = arith.index_cast %12 : i32 to index
  val cast_9 = Module(new BitCastNode(NumOuts = 1, ID = 9))

  //%15 = dataflow.addr %arg1[%14] {memShape = [96]} : memref<96xi32>[index] -> i32
  val address_10 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 10)(ElementSize = 32, ArraySize = List()))

  //%16 = dataflow.load %15 : i32 -> i32
  val load_11 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 11, RouteID = 0))

  //%17 = arith.addi %8, %16 : i32
  val int_add_12 = Module(new ComputeNode(NumOuts = 1, ID = 12, opCode = "Add")(sign = false, Debug = false))

  //%18 = dataflow.select %13, %17, %8 : i32
  val select_13 = Module(new SelectNode(NumOuts = 2, ID = 13))

  //%19 = arith.addi %9, %c1 {Exe = "Loop"} : index
  val int_add_14 = Module(new ComputeNode(NumOuts = 2, ID = 14, opCode = "Add")(sign = false, Debug = false))

  //%20 = arith.cmpi eq, %19, %c2 {Exe = "Loop"} : index
  val int_cmp_15 = Module(new ComputeNode(NumOuts = 1, ID = 15, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %20, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_16 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 16))

  //%3 = arith.shrsi %2, %c2_i32 : i32
  val int_shr_17 = Module(new ComputeNode(NumOuts = 1, ID = 17, opCode = "ashr")(sign = false, Debug = false))

  //%4 = dataflow.addr %arg0[%0] {memShape = [64]} : memref<64xi32>[index] -> i32
  val address_18 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 18)(ElementSize = 32, ArraySize = List()))

  //dataflow.store %3 %4 : i32 i32
  val store_19 = Module(new UnTypStoreCache(NumPredOps = 0, NumSuccOps = 0, ID = 19, RouteID = 1))

  //%5 = arith.addi %0, %c1 {Exe = "Loop"} : index
  val int_add_20 = Module(new ComputeNode(NumOuts = 2, ID = 20, opCode = "Add")(sign = false, Debug = false))

  //%6 = arith.cmpi eq, %5, %c64 {Exe = "Loop"} : index
  val int_cmp_21 = Module(new ComputeNode(NumOuts = 1, ID = 21, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %6, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_22 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 22))

  //func.return
  val return_23 = Module(new RetNode2(retTypes = List(), ID = 23))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1, 1), NumOuts = List(1), NumCarry = List(1, 1), NumExits = 1, ID = 0))

  val loop_1 = Module(new LoopBlockNode(NumIns = List(1,1), NumOuts = List(), NumCarry = List(3), NumExits = 1, ID = 1))



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

  loop_0.io.loopBack(0) <> state_branch_16.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_16.io.TrueOutput(0)

  loop_1.io.enable <> state_branch_0.io.Out(0)

  loop_1.io.loopBack(0) <> state_branch_22.io.FalseOutput(0)

  loop_1.io.loopFinish(0) <> state_branch_22.io.TrueOutput(0)

  store_19.io.Out(0).ready := true.B

  // state_branch_23.io.PredOp(0) <> store_19.io.SuccOp(0)



  /* ================================================================== *
   *                   Loop dependencies.                               *
   * ================================================================== */



  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

  loop_0.io.InLiveIn(0) <> cast_2.io.Out(0)

  loop_0.io.InLiveIn(1) <> loop_1.io.OutLiveIn.elements("field1")(0)//FineGrainedArgCall.io.Out.data.elements("field1")(0)

  // loop_0.io.InLiveIn(2) <> loop_1.io.OutLiveIn.elements("field0")(1)//FineGrainedArgCall.io.Out.data.elements("field0")(0)
  
  
  
  loop_1.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)


  loop_1.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)



  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  int_add_7.io.LeftIO <> loop_0.io.OutLiveIn.elements("field0")(0)

  address_10.io.baseAddress <> loop_0.io.OutLiveIn.elements("field1")(0)

  address_18.io.baseAddress <> loop_1.io.OutLiveIn.elements("field0")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> select_13.io.Out(0)

  loop_0.io.OutLiveOut.elements("field0")(0) <> int_shr_17.io.LeftIO

  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

   loop_1.io.loopExit(0) <>  return_23.io.In.enable
   state_branch_22.io.enable <>   loop_0.io.loopExit(0)

  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> select_13.io.Out(1)//loop_0.io.CarryDepenOut.elements("field0")(0)

  merge_3.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  loop_0.io.CarryDepenIn(1) <> int_add_14.io.Out(1)//loop_0.io.CarryDepenOut.elements("field1")(0)

  merge_4.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(0)

  loop_1.io.CarryDepenIn(0) <> int_add_20.io.Out(1)//loop_1.io.CarryDepenOut.elements("field0")(0)

  // merge_1.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  merge_3.io.Mask <> exe_block_0.io.MaskBB(0)

  merge_4.io.Mask <> exe_block_0.io.MaskBB(1)

  // merge_1.io.Mask <> exe_block_1.io.MaskBB(0)

  // merge_1.io.InData(0) <> int_const_0.io.Out

  merge_3.io.InData(0) <> int_const_1.io.Out

  merge_4.io.InData(0) <> int_const_2.io.Out

  int_mul_6.io.LeftIO <> int_const_3.io.Out

  int_cmp_8.io.LeftIO <> int_const_4.io.Out

  int_add_14.io.LeftIO <> int_const_5.io.Out

  int_cmp_15.io.LeftIO <> int_const_6.io.Out

  int_shr_17.io.RightIO <> int_const_7.io.Out

  int_add_20.io.LeftIO <> int_const_8.io.Out

  int_cmp_21.io.LeftIO <> int_const_9.io.Out

  cast_2.io.Input <> loop_1.io.CarryDepenOut.elements("field0")(0)//.io.Out(0)

  address_18.io.idx(0) <> loop_1.io.CarryDepenOut.elements("field0")(1)//merge_1.io.Out(1)

  int_add_20.io.RightIO <> loop_1.io.CarryDepenOut.elements("field0")(2)//merge_1.io.Out(2)

  int_add_12.io.LeftIO <> merge_3.io.Out(0)

  select_13.io.InData2 <> merge_3.io.Out(1)

  cast_5.io.Input <> merge_4.io.Out(0)

  int_add_14.io.RightIO <> merge_4.io.Out(1)

  int_mul_6.io.RightIO <> cast_5.io.Out(0)

  int_add_7.io.RightIO <> int_mul_6.io.Out(0)

  int_cmp_8.io.RightIO <> int_add_7.io.Out(0)

  cast_9.io.Input <> int_add_7.io.Out(1)

  select_13.io.Select <> int_cmp_8.io.Out(0)

  address_10.io.idx(0) <> cast_9.io.Out(0)

  load_11.io.GepAddr <> address_10.io.Out(0)

  int_add_12.io.RightIO <> load_11.io.Out(0)

  select_13.io.InData1 <> int_add_12.io.Out(0)

  int_cmp_15.io.RightIO <> int_add_14.io.Out(0)

  state_branch_16.io.CmpIO <> int_cmp_15.io.Out(0)

  store_19.io.inData <> int_shr_17.io.Out(0)

  store_19.io.GepAddr <> address_18.io.Out(0)

  int_cmp_21.io.RightIO <> int_add_20.io.Out(0)

  state_branch_22.io.CmpIO <> int_cmp_21.io.Out(0)

  mem_ctrl_cache.io.rd.mem(0).MemReq <> load_11.io.MemReq

  load_11.io.MemResp <> mem_ctrl_cache.io.rd.mem(0).MemResp

  mem_ctrl_cache.io.wr.mem(0).MemReq <> store_19.io.MemReq

  store_19.io.MemResp <> mem_ctrl_cache.io.wr.mem(0).MemResp



  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_1.io.enable <> exe_block_0.io.Out(0)

  int_const_2.io.enable <> exe_block_0.io.Out(1)

  int_const_3.io.enable <> exe_block_0.io.Out(2)

  int_const_4.io.enable <> exe_block_0.io.Out(3)

  int_const_5.io.enable <> exe_block_0.io.Out(4)

  int_const_6.io.enable <> exe_block_0.io.Out(5)

  merge_3.io.enable <> exe_block_0.io.Out(6)

  merge_4.io.enable <> exe_block_0.io.Out(7)

  cast_5.io.enable <> exe_block_0.io.Out(8)

  int_mul_6.io.enable <> exe_block_0.io.Out(9)

  int_add_7.io.enable <> exe_block_0.io.Out(10)

  int_cmp_8.io.enable <> exe_block_0.io.Out(11)

  cast_9.io.enable <> exe_block_0.io.Out(12)

  address_10.io.enable <> exe_block_0.io.Out(13)

  load_11.io.enable <> exe_block_0.io.Out(14)

  int_add_12.io.enable <> exe_block_0.io.Out(15)

  select_13.io.enable <> exe_block_0.io.Out(16)

  int_add_14.io.enable <> exe_block_0.io.Out(17)

  int_cmp_15.io.enable <> exe_block_0.io.Out(18)

  state_branch_16.io.enable <> exe_block_0.io.Out(19)

  // int_const_0.io.enable <> exe_block_1.io.Out(1)

  int_const_7.io.enable <> exe_block_1.io.Out(2)

  int_const_8.io.enable <> exe_block_1.io.Out(3)

  int_const_9.io.enable <> exe_block_1.io.Out(4)

  // merge_1.io.enable <> exe_block_1.io.Out(5)

  cast_2.io.enable <> exe_block_1.io.Out(6)

  int_shr_17.io.enable <> exe_block_1.io.Out(7)

  address_18.io.enable <> exe_block_1.io.Out(8)

  store_19.io.enable <> exe_block_1.io.Out(9)

  int_add_20.io.enable <> exe_block_1.io.Out(1)

  int_cmp_21.io.enable <> exe_block_1.io.Out(5)

  // state_branch_22.io.enable <> exe_block_1.io.Out(12)

  io.out <> return_23.io.Out


}



// import java.io.{File, FileWriter}

// object polarmatchingTop extends App {
//   implicit val p = new WithAccelConfig ++ new WithTestConfig
//   val verilogString = getVerilogString(new polarmatching())
//   val filePath = "RTL/polarmatching.v"
//   val writer = new PrintWriter(filePath)
//   try { 
//       writer.write(verilogString)
//   } finally {
//     writer.close()
//   }
// }