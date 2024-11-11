
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

abstract class gemmDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List(32, 32 ,32, 32, 32))))
	  val MemResp = Flipped(Valid(new MemResp))
	  val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List()))
	})
}

class gemmDF(implicit p: Parameters) extends gemmDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1, 2, 1, 1)))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 4, NumWrite = 2))

  io.MemReq <> mem_ctrl_cache.io.cache.MemReq
  mem_ctrl_cache.io.cache.MemResp <> io.MemResp



  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c0 = arith.constant 0 : index
  val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c0 = arith.constant 0 : index
  val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

  //%c0 = arith.constant 0 : index
  val int_const_2 = Module(new ConstFastNode(value = 0, ID = 2))

  //%c1 = arith.constant 1 : index
  val int_const_3 = Module(new ConstFastNode(value = 1, ID = 3))

  //%c32 = arith.constant 32 : index
  val int_const_4 = Module(new ConstFastNode(value = 32, ID = 4))

  //%c1 = arith.constant 1 : index
  val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

  //%c32 = arith.constant 32 : index
  val int_const_6 = Module(new ConstFastNode(value = 32, ID = 6))

  //%c1 = arith.constant 1 : index
  val int_const_7 = Module(new ConstFastNode(value = 1, ID = 7))

  //%c32 = arith.constant 32 : index
  val int_const_8 = Module(new ConstFastNode(value = 32, ID = 8))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 17, NumPhi = 1, BID = 0))

  val exe_block_1 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 13, NumPhi = 1, BID = 1))

  val exe_block_2 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 8, NumPhi = 1, BID = 2))



  /* ================================================================== *
   *                   Printing Operation nodes. 30                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%0 = dataflow.merge %c0 or %arg5 {Select = "Loop_Signal"} : index
  val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 3, ID = 1, Res = false))

  //%3 = dataflow.merge %c0 or %arg6 {Select = "Loop_Signal"} : index
  val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 4, ID = 2, Res = false))

  //%4 = dataflow.addr %arg2[%0, %3] {memShape = [32, 32]} : memref<32x32xi32>[index, index] -> i32
  val address_3 = Module(new GepNode(NumIns = 2, NumOuts = 1, ID = 3)(ElementSize = 32, ArraySize = List()))

  //%5 = dataflow.load %4 : i32 -> i32
  val load_4 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 4, RouteID = 0))

  //%6 = arith.muli %5, %arg1 : i32
  val int_mul_5 = Module(new ComputeNode(NumOuts = 1, ID = 5, opCode = "Mul")(sign = false, Debug = false))

  //%7 = dataflow.addr %arg2[%0, %3] {memShape = [32, 32]} : memref<32x32xi32>[index, index] -> i32
  val address_6 = Module(new GepNode(NumIns = 2, NumOuts = 1, ID = 6)(ElementSize = 32, ArraySize = List()))

  //dataflow.store %6 %7 : i32 i32
  val store_7 = Module(new UnTypStoreCache(NumPredOps = 0, NumSuccOps = 1, ID = 7, RouteID = 4))

  //%10 = dataflow.merge %c0 or %arg7 {Select = "Loop_Signal"} : index
  val merge_8 = Module(new MergeNode(NumInputs = 2, NumOutputs = 3, ID = 8, Res = false))

  //%11 = dataflow.addr %arg3[%0, %10] {memShape = [32, 32]} : memref<32x32xi32>[index, index] -> i32
  val address_9 = Module(new GepNode(NumIns = 2, NumOuts = 1, ID = 9)(ElementSize = 32, ArraySize = List()))

  //%12 = dataflow.load %11 : i32 -> i32
  val load_10 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 10, RouteID = 1))

  //%13 = arith.muli %arg0, %12 : i32
  val int_mul_11 = Module(new ComputeNode(NumOuts = 1, ID = 11, opCode = "Mul")(sign = false, Debug = false))

  //%14 = dataflow.addr %arg4[%10, %3] {memShape = [32, 32]} : memref<32x32xi32>[index, index] -> i32
  val address_12 = Module(new GepNode(NumIns = 2, NumOuts = 1, ID = 12)(ElementSize = 32, ArraySize = List()))

  //%15 = dataflow.load %14 : i32 -> i32
  val load_13 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 13, RouteID = 2))

  //%16 = arith.muli %13, %15 : i32
  val int_mul_14 = Module(new ComputeNode(NumOuts = 1, ID = 14, opCode = "Mul")(sign = false, Debug = false))

  //%17 = dataflow.addr %arg2[%0, %3] {memShape = [32, 32]} : memref<32x32xi32>[index, index] -> i32
  val address_15 = Module(new GepNode(NumIns = 2, NumOuts = 1, ID = 15)(ElementSize = 32, ArraySize = List()))

  //%18 = dataflow.load %17 : i32 -> i32
  val load_16 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 16, RouteID = 3))

  //%19 = arith.addi %18, %16 : i32
  val int_add_17 = Module(new ComputeNode(NumOuts = 1, ID = 17, opCode = "Add")(sign = false, Debug = false))

  //%20 = dataflow.addr %arg2[%0, %3] {memShape = [32, 32]} : memref<32x32xi32>[index, index] -> i32
  val address_18 = Module(new GepNode(NumIns = 2, NumOuts = 1, ID = 18)(ElementSize = 32, ArraySize = List()))

  //dataflow.store %19 %20 : i32 i32
  val store_19 = Module(new UnTypStoreCache(NumPredOps = 0, NumSuccOps = 0, ID = 19, RouteID = 5))

  //%21 = arith.addi %10, %c1 {Exe = "Loop"} : index
  val int_add_20 = Module(new ComputeNode(NumOuts = 2, ID = 20, opCode = "Add")(sign = false, Debug = false))

//   //%22 = arith.cmpi eq, %21, %c32 {Exe = "Loop"} : index
//   val int_cmp_21 = Module(new ComputeNode(NumOuts = 1, ID = 21, opCode = "eq")(sign = false, Debug = false))

//   //dataflow.state %22, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//   val state_branch_22 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 1, ID = 22))
  

  val state_branch_23 = Module(new CompareBranchNode(ID = 22, opCode = "eq"))


  //%8 = arith.addi %3, %c1 {Exe = "Loop"} : index
  val int_add_23 = Module(new ComputeNode(NumOuts = 2, ID = 23, opCode = "Add")(sign = false, Debug = false))

  //%9 = arith.cmpi eq, %8, %c32 {Exe = "Loop"} : index
  val int_cmp_24 = Module(new ComputeNode(NumOuts = 1, ID = 24, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %9, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_25 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 1, ID = 25))

  //%1 = arith.addi %0, %c1 {Exe = "Loop"} : index
  val int_add_26 = Module(new ComputeNode(NumOuts = 2, ID = 26, opCode = "Add")(sign = false, Debug = false))

  //%2 = arith.cmpi eq, %1, %c32 {Exe = "Loop"} : index
  val int_cmp_27 = Module(new ComputeNode(NumOuts = 1, ID = 27, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %2, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_28 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 28))

  //func.return
  val return_29 = Module(new RetNode2(retTypes = List(), ID = 29))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1, 3, 1, 1, 3, 2), NumOuts = List(), NumCarry = List(1), NumExits = 1, ID = 0))

  val loop_1 = Module(new LoopBlockNode(NumIns = List(2, 2, 1), NumOuts = List(), NumCarry = List(1), NumExits = 1, ID = 1))

  val loop_2 = Module(new LoopBlockNode(NumIns = List(), NumOuts = List(), NumCarry = List(1), NumExits = 1, ID = 2))

  loop_0.io.loopExit(0) <>  DontCare
  loop_1.io.loopExit(0) <>  DontCare
  loop_2.io.loopExit(0) <>  return_29.io.In.enable

  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back

  exe_block_1.io.predicateIn(0) <> loop_1.io.activate_loop_start

  exe_block_1.io.predicateIn(1) <> loop_1.io.activate_loop_back

  exe_block_2.io.predicateIn(0) <> loop_2.io.activate_loop_start

  exe_block_2.io.predicateIn(1) <> loop_2.io.activate_loop_back



  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> exe_block_1.io.Out(0)

//   loop_0.io.loopBack(0) <> state_branch_22.io.FalseOutput(0)

//   loop_0.io.loopFinish(0) <> state_branch_22.io.TrueOutput(0)

  loop_0.io.loopBack(0) <> state_branch_23.io.Out(1)

  loop_0.io.loopFinish(0) <> state_branch_23.io.Out(0)

  loop_1.io.enable <> exe_block_2.io.Out(0)

  loop_1.io.loopBack(0) <> state_branch_25.io.FalseOutput(0)

  loop_1.io.loopFinish(0) <> state_branch_25.io.TrueOutput(0)

  loop_2.io.enable <> state_branch_0.io.Out(0)

  loop_2.io.loopBack(0) <> state_branch_28.io.FalseOutput(0)

  loop_2.io.loopFinish(0) <> state_branch_28.io.TrueOutput(0)

  store_7.io.Out(0).ready := true.B

  state_branch_25.io.PredOp(0) <> store_7.io.SuccOp(0)

  store_19.io.Out(0).ready := true.B

//   state_branch_22.io.PredOp(0) <> store_19.io.SuccOp(0)



  /* ================================================================== *
   *                   Loop dependencies.                               *
   * ================================================================== */



  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

   loop_0.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field3")(0)

  loop_0.io.InLiveIn(1) <> merge_1.io.Out(0)

  loop_0.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

  loop_0.io.InLiveIn(3) <> FineGrainedArgCall.io.Out.data.elements("field4")(0)

  loop_0.io.InLiveIn(4) <> merge_2.io.Out(0)

  loop_0.io.InLiveIn(5) <> FineGrainedArgCall.io.Out.data.elements("field2")(1)

  loop_1.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)

  loop_1.io.InLiveIn(1) <> merge_1.io.Out(1)

  loop_1.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)



  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  address_9.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  address_9.io.idx(0) <> loop_0.io.OutLiveIn.elements("field1")(0)

  address_15.io.idx(0) <> loop_0.io.OutLiveIn.elements("field1")(1)

  address_18.io.idx(0) <> loop_0.io.OutLiveIn.elements("field1")(2)

  int_mul_11.io.LeftIO <> loop_0.io.OutLiveIn.elements("field2")(0)

  address_12.io.baseAddress <> loop_0.io.OutLiveIn.elements("field3")(0)

  address_12.io.idx(1) <> loop_0.io.OutLiveIn.elements("field4")(0)

  address_15.io.idx(1) <> loop_0.io.OutLiveIn.elements("field4")(1)

  address_18.io.idx(1) <> loop_0.io.OutLiveIn.elements("field4")(2)

  address_15.io.baseAddress <> loop_0.io.OutLiveIn.elements("field5")(0)

  address_18.io.baseAddress <> loop_0.io.OutLiveIn.elements("field5")(1)

  address_3.io.baseAddress <> loop_1.io.OutLiveIn.elements("field0")(0)

  address_6.io.baseAddress <> loop_1.io.OutLiveIn.elements("field0")(1)

  address_3.io.idx(0) <> loop_1.io.OutLiveIn.elements("field1")(0)

  address_6.io.idx(0) <> loop_1.io.OutLiveIn.elements("field1")(1)

  int_mul_5.io.RightIO <> loop_1.io.OutLiveIn.elements("field2")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */



  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_20.io.Out(1)

  merge_8.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  loop_1.io.CarryDepenIn(0) <> int_add_23.io.Out(1)

  merge_2.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)

  loop_2.io.CarryDepenIn(0) <> int_add_26.io.Out(1)


  merge_1.io.InData(1) <> loop_2.io.CarryDepenOut.elements("field0")(0)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  merge_8.io.Mask <> exe_block_0.io.MaskBB(0)

  merge_2.io.Mask <> exe_block_1.io.MaskBB(0)

  merge_1.io.Mask <> exe_block_2.io.MaskBB(0)

  merge_1.io.InData(0) <> int_const_0.io.Out

  merge_2.io.InData(0) <> int_const_1.io.Out

  merge_8.io.InData(0) <> int_const_2.io.Out

  int_add_20.io.LeftIO <> int_const_3.io.Out

//   int_cmp_21.io.LeftIO <> int_const_4.io.Out

  state_branch_23.io.LeftIO <> int_const_4.io.Out

  int_add_23.io.LeftIO <> int_const_5.io.Out

  int_cmp_24.io.LeftIO <> int_const_6.io.Out

  int_add_26.io.LeftIO <> int_const_7.io.Out

  int_cmp_27.io.LeftIO <> int_const_8.io.Out

  int_add_26.io.RightIO <> merge_1.io.Out(2)

  address_3.io.idx(1) <> merge_2.io.Out(1)

  address_6.io.idx(1) <> merge_2.io.Out(2)

  int_add_23.io.RightIO <> merge_2.io.Out(3)

  load_4.io.GepAddr <> address_3.io.Out(0)

  int_mul_5.io.LeftIO <> load_4.io.Out(0)

  store_7.io.inData <> int_mul_5.io.Out(0)

  store_7.io.GepAddr <> address_6.io.Out(0)

  address_9.io.idx(1) <> merge_8.io.Out(0)

  address_12.io.idx(0) <> merge_8.io.Out(1)

  int_add_20.io.RightIO <> merge_8.io.Out(2)

  load_10.io.GepAddr <> address_9.io.Out(0)

  int_mul_11.io.RightIO <> load_10.io.Out(0)

  int_mul_14.io.LeftIO <> int_mul_11.io.Out(0)

  load_13.io.GepAddr <> address_12.io.Out(0)

  int_mul_14.io.RightIO <> load_13.io.Out(0)

  int_add_17.io.RightIO <> int_mul_14.io.Out(0)

  load_16.io.GepAddr <> address_15.io.Out(0)

  int_add_17.io.LeftIO <> load_16.io.Out(0)

  store_19.io.inData <> int_add_17.io.Out(0)

  store_19.io.GepAddr <> address_18.io.Out(0)

//   int_cmp_21.io.RightIO <> int_add_20.io.Out(0)
  
  state_branch_23.io.RightIO <> int_add_20.io.Out(0)

//   state_branch_22.io.CmpIO <> int_cmp_21.io.Out(0)

  int_cmp_24.io.RightIO <> int_add_23.io.Out(0)

  state_branch_25.io.CmpIO <> int_cmp_24.io.Out(0)

  int_cmp_27.io.RightIO <> int_add_26.io.Out(0)

  state_branch_28.io.CmpIO <> int_cmp_27.io.Out(0)

  mem_ctrl_cache.io.rd.mem(0).MemReq <> load_4.io.MemReq

  load_4.io.MemResp <> mem_ctrl_cache.io.rd.mem(0).MemResp

  mem_ctrl_cache.io.rd.mem(1).MemReq <> load_10.io.MemReq

  load_10.io.MemResp <> mem_ctrl_cache.io.rd.mem(1).MemResp

  mem_ctrl_cache.io.rd.mem(2).MemReq <> load_13.io.MemReq

  load_13.io.MemResp <> mem_ctrl_cache.io.rd.mem(2).MemResp

  mem_ctrl_cache.io.rd.mem(3).MemReq <> load_16.io.MemReq

  load_16.io.MemResp <> mem_ctrl_cache.io.rd.mem(3).MemResp

  mem_ctrl_cache.io.wr.mem(0).MemReq <> store_7.io.MemReq

  store_7.io.MemResp <> mem_ctrl_cache.io.wr.mem(0).MemResp

  mem_ctrl_cache.io.wr.mem(1).MemReq <> store_19.io.MemReq

  store_19.io.MemResp <> mem_ctrl_cache.io.wr.mem(1).MemResp



  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_2.io.enable <> exe_block_0.io.Out(0)

  int_const_3.io.enable <> exe_block_0.io.Out(1)

  int_const_4.io.enable <> exe_block_0.io.Out(2)

  merge_8.io.enable <> exe_block_0.io.Out(3)

  address_9.io.enable <> exe_block_0.io.Out(4)

  load_10.io.enable <> exe_block_0.io.Out(5)

  int_mul_11.io.enable <> exe_block_0.io.Out(6)

  address_12.io.enable <> exe_block_0.io.Out(7)

  load_13.io.enable <> exe_block_0.io.Out(8)

  int_mul_14.io.enable <> exe_block_0.io.Out(9)

  address_15.io.enable <> exe_block_0.io.Out(10)

  load_16.io.enable <> exe_block_0.io.Out(11)

  int_add_17.io.enable <> exe_block_0.io.Out(12)

  address_18.io.enable <> exe_block_0.io.Out(13)

  store_19.io.enable <> exe_block_0.io.Out(14)

  int_add_20.io.enable <> exe_block_0.io.Out(15)

//   int_cmp_21.io.enable <> exe_block_0.io.Out(16)

//   state_branch_22.io.enable <> exe_block_0.io.Out(17)

state_branch_23.io.enable <> exe_block_0.io.Out(16)

  int_const_1.io.enable <> exe_block_1.io.Out(1)

  int_const_5.io.enable <> exe_block_1.io.Out(2)

  int_const_6.io.enable <> exe_block_1.io.Out(3)

  merge_2.io.enable <> exe_block_1.io.Out(4)

  address_3.io.enable <> exe_block_1.io.Out(5)

  load_4.io.enable <> exe_block_1.io.Out(6)

  int_mul_5.io.enable <> exe_block_1.io.Out(7)

  address_6.io.enable <> exe_block_1.io.Out(8)

  store_7.io.enable <> exe_block_1.io.Out(9)

  int_add_23.io.enable <> exe_block_1.io.Out(10)

  int_cmp_24.io.enable <> exe_block_1.io.Out(11)

  state_branch_25.io.enable <> exe_block_1.io.Out(12)

  int_const_0.io.enable <> exe_block_2.io.Out(1)

  int_const_7.io.enable <> exe_block_2.io.Out(2)

  int_const_8.io.enable <> exe_block_2.io.Out(3)

  merge_1.io.enable <> exe_block_2.io.Out(4)

  int_add_26.io.enable <> exe_block_2.io.Out(5)

  int_cmp_27.io.enable <> exe_block_2.io.Out(6)

  state_branch_28.io.enable <> exe_block_2.io.Out(7)

  io.out <> return_29.io.Out

}

