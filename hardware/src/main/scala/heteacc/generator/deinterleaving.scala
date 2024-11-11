
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

abstract class deinterleavingDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32, 32, 32))))
	  val MemResp = Flipped(Valid(new MemResp))
	  val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List()))
	})
}

class deinterleavingDF(implicit p: Parameters) extends deinterleavingDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1, 1 )))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 3, NumWrite = 2))

  io.MemReq <> mem_ctrl_cache.io.cache.MemReq
  mem_ctrl_cache.io.cache.MemResp <> io.MemResp



  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c0 = arith.constant 0 : index
  val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_1 = Module(new ConstFastNode(value = 1, ID = 1))

  //%c-1 = arith.constant -1 : index
  val int_const_2 = Module(new ConstFastNode(value = -1, ID = 2))

  //%c0 = arith.constant 0 : index
  val int_const_3 = Module(new ConstFastNode(value = 0, ID = 3))

  //%c-1 = arith.constant -1 : index
  val int_const_4 = Module(new ConstFastNode(value = -1, ID = 4))

  //%c13 = arith.constant 13 : index
  val int_const_5 = Module(new ConstFastNode(value = 13, ID = 5))

  //%c0 = arith.constant 0 : index
  val int_const_6 = Module(new ConstFastNode(value = 0, ID = 6))

  //%c7 = arith.constant 7 : index
  val int_const_7 = Module(new ConstFastNode(value = 7, ID = 7))

  //%c0 = arith.constant 0 : index
  val int_const_8 = Module(new ConstFastNode(value = 0, ID = 8))

  //%c0 = arith.constant 0 : index
  val int_const_9 = Module(new ConstFastNode(value = 0, ID = 9))

  //%c14_i32 = arith.constant 14 : i32
  val int_const_10 = Module(new ConstFastNode(value = 14, ID = 10))

  //%c-1_i32 = arith.constant -1 : i32
  val int_const_11 = Module(new ConstFastNode(value = -1, ID = 11))

  //%c2_i32 = arith.constant 2 : i32
  val int_const_12 = Module(new ConstFastNode(value = 2, ID = 12))

  //%c0 = arith.constant 0 : index
  val int_const_13 = Module(new ConstFastNode(value = 0, ID = 13))

  //%c0 = arith.constant 0 : index
  val int_const_14 = Module(new ConstFastNode(value = 0, ID = 14))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_15 = Module(new ConstFastNode(value = 1, ID = 15))

  //%c0 = arith.constant 0 : index
  val int_const_16 = Module(new ConstFastNode(value = 0, ID = 16))

  //%c1 = arith.constant 1 : index
  val int_const_17 = Module(new ConstFastNode(value = 1, ID = 17))

  //%c9 = arith.constant 9 : index
  val int_const_18 = Module(new ConstFastNode(value = 9, ID = 18))

  //%c1 = arith.constant 1 : index
  val int_const_19 = Module(new ConstFastNode(value = 1, ID = 19))

  //%c14 = arith.constant 14 : index
  val int_const_20 = Module(new ConstFastNode(value = 14, ID = 20))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new ExecutionBlockNode(NumInputs = 1, NumOuts = 26, BID = 0))

  val exe_block_1 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 25, NumPhi = 1, BID = 1))

  val exe_block_2 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 12, NumPhi = 1, BID = 2))



  /* ================================================================== *
   *                   Printing Operation nodes. 42                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%0 = dataflow.merge %c0 or %arg3 {Select = "Loop_Signal"} : index
  val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 3, ID = 1, Res = false))

  //%1 = arith.index_cast %0 : index to i32
  val cast_2 = Module(new BitCastNode(NumOuts = 2, ID = 2))

  //%2 = arith.cmpi slt, %1, %c1_i32 : i32
  val int_cmp_3 = Module(new ComputeNode(NumOuts = 1, ID = 3, opCode = "slt")(sign = false, Debug = false))

  //%3 = arith.muli %0, %c-1 : index
  val int_mul_4 = Module(new ComputeNode(NumOuts = 1, ID = 4, opCode = "Mul")(sign = false, Debug = false))

  //%6 = dataflow.merge %c0 or %arg4 {Select = "Loop_Signal"} : index
  val merge_5 = Module(new MergeNode(NumInputs = 2, NumOutputs = 3, ID = 5, Res = false))

  //%7 = arith.index_cast %6 : index to i32
  val cast_6 = Module(new BitCastNode(NumOuts = 3, ID = 6))

  //%8 = arith.muli %6, %c-1 : index
  val int_mul_7 = Module(new ComputeNode(NumOuts = 2, ID = 7, opCode = "Mul")(sign = false, Debug = false))

  //%9 = arith.addi %3, %8 : index
  val int_add_8 = Module(new ComputeNode(NumOuts = 1, ID = 8, opCode = "Add")(sign = false, Debug = false))

  //%10 = arith.addi %9, %c13 : index
  val int_add_9 = Module(new ComputeNode(NumOuts = 1, ID = 9, opCode = "Add")(sign = false, Debug = false))

  //%11 = arith.cmpi sge, %10, %c0 : index
  val int_cmp_10 = Module(new ComputeNode(NumOuts = 1, ID = 10, opCode = "gte")(sign = false, Debug = false))

  //%12 = arith.addi %8, %c7 : index
  val int_add_11 = Module(new ComputeNode(NumOuts = 2, ID = 11, opCode = "Add")(sign = false, Debug = false))

  //%13 = arith.cmpi sge, %12, %c0 : index
  val int_cmp_12 = Module(new ComputeNode(NumOuts = 1, ID = 12, opCode = "gte")(sign = false, Debug = false))

  //%14 = arith.cmpi slt, %12, %c0 : index
  val int_cmp_13 = Module(new ComputeNode(NumOuts = 1, ID = 13, opCode = "slt")(sign = false, Debug = false))

  //%15 = arith.andi %14, %2 : i1
  val int_andi_14 = Module(new ComputeNode(NumOuts = 1, ID = 14, opCode = "and")(sign = false, Debug = false))

  //%16 = arith.ori %13, %15 : i1
  val int_ori_15 = Module(new ComputeNode(NumOuts = 1, ID = 15, opCode = "or")(sign = false, Debug = false))

  //%17 = arith.andi %11, %16 : i1
  val int_andi_16 = Module(new ComputeNode(NumOuts = 1, ID = 16, opCode = "and")(sign = false, Debug = false))

  //dataflow.state %17, "if_then" or "null" : i1
  val state_branch_17 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 17))

  //%20 = arith.muli %7, %c14_i32 : i32
  val int_mul_18 = Module(new ComputeNode(NumOuts = 1, ID = 18, opCode = "Mul")(sign = false, Debug = false))

  //%21 = arith.addi %7, %c-1_i32 : i32
  val int_add_19 = Module(new ComputeNode(NumOuts = 1, ID = 19, opCode = "Add")(sign = false, Debug = false))

  //%22 = arith.muli %7, %21 : i32
  val int_mul_20 = Module(new ComputeNode(NumOuts = 1, ID = 20, opCode = "Mul")(sign = false, Debug = false))

  //%23 = arith.divsi %22, %c2_i32 : i32
  val int_divsi21 = Module(new ComputeNode(NumOuts = 1, ID = 21, opCode = "udiv")(sign = false, Debug = false))

  //%24 = arith.subi %20, %23 : i32
  val int_sub_22 = Module(new ComputeNode(NumOuts = 1, ID = 22, opCode = "Sub")(sign = false, Debug = false))

  //%25 = arith.addi %24, %1 : i32
  val int_add_23 = Module(new ComputeNode(NumOuts = 1, ID = 23, opCode = "Add")(sign = false, Debug = false))

  //%26 = arith.index_cast %25 : i32 to index
  val cast_24 = Module(new BitCastNode(NumOuts = 1, ID = 24))

  //%27 = dataflow.addr %arg2[%c0] {memShape = [1]} : memref<1xi32>[index] -> i32
  val address_25 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 25)(ElementSize = 32, ArraySize = List()))

  //%28 = dataflow.load %27 : i32 -> i32
  val load_26 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 26, RouteID = 0))

  //%29 = arith.index_cast %28 : i32 to index
  val cast_27 = Module(new BitCastNode(NumOuts = 3, ID = 27))

  //%30 = dataflow.addr %arg0[%29] {memShape = [96]} : memref<96xi32>[index] -> i32
  val address_28 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 28)(ElementSize = 32, ArraySize = List()))

  //%31 = dataflow.load %30 : i32 -> i32
  val load_29 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 29, RouteID = 1))

  //%32 = dataflow.addr %arg1[%26] {memShape = [96]} : memref<96xi32>[index] -> i32
  val address_30 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 30)(ElementSize = 32, ArraySize = List()))

  //dataflow.store %31 %32 : i32 i32
  val store_31 = Module(new UnTypStoreCache(NumPredOps = 0, NumSuccOps = 0, ID = 31, RouteID = 3))

  //%33 = dataflow.addr %arg2[%c0] {memShape = [1]} : memref<1xi32>[index] -> i32
  val address_32 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 32)(ElementSize = 32, ArraySize = List()))

  //%34 = dataflow.load %33 : i32 -> i32
  val load_33 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 33, RouteID = 2))

  //%35 = arith.addi %34, %c1_i32 : i32
  val int_add_34 = Module(new ComputeNode(NumOuts = 1, ID = 34, opCode = "Add")(sign = false, Debug = false))

  //%36 = dataflow.addr %arg2[%c0] {memShape = [1]} : memref<1xi32>[index] -> i32
  val address_35 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 35)(ElementSize = 32, ArraySize = List()))

  //dataflow.store %35 %36 : i32 i32
  val store_36 = Module(new UnTypStoreCache(NumPredOps = 0, NumSuccOps = 0, ID = 36, RouteID = 4))

  //%18 = arith.addi %6, %c1 {Exe = "Loop"} : index
  val int_add_37 = Module(new ComputeNode(NumOuts = 2, ID = 37, opCode = "Add")(sign = false, Debug = false))

  //%19 = arith.cmpi eq, %18, %c9 {Exe = "Loop"} : index
  val int_cmp_38 = Module(new ComputeNode(NumOuts = 1, ID = 38, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %19, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_39 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 39))

  //%4 = arith.addi %0, %c1 {Exe = "Loop"} : index
  val int_add_40 = Module(new ComputeNode(NumOuts = 2, ID = 40, opCode = "Add")(sign = false, Debug = false))

  //%5 = arith.cmpi eq, %4, %c14 {Exe = "Loop"} : index
  val int_cmp_41 = Module(new ComputeNode(NumOuts = 1, ID = 41, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %5, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_42 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 42))

  //func.return
  val return_43 = Module(new RetNode2(retTypes = List(), ID = 43))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1, 1, 1, 3, 1, 1), NumOuts = List(), NumCarry = List(1), NumExits = 1, ID = 0))

  val loop_1 = Module(new LoopBlockNode(NumIns = List(1,1,1), NumOuts = List(), NumCarry = List(1), NumExits = 1, ID = 1))



  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  state_branch_17.io.TrueOutput(0) <> exe_block_0.io.predicateIn(0)
  
  state_branch_17.io.FalseOutput(0) <> DontCare

  // state_branch_39.io.enable <> 

  exe_block_1.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_1.io.predicateIn(1) <> loop_0.io.activate_loop_back

  exe_block_2.io.predicateIn(0) <> loop_1.io.activate_loop_start

  exe_block_2.io.predicateIn(1) <> loop_1.io.activate_loop_back



  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> exe_block_2.io.Out(0)

  loop_0.io.loopBack(0) <> state_branch_39.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_39.io.TrueOutput(0)

  loop_1.io.enable <> state_branch_0.io.Out(0)

  loop_1.io.loopBack(0) <> state_branch_42.io.FalseOutput(0)

  loop_1.io.loopFinish(0) <> state_branch_42.io.TrueOutput(0)


  store_31.io.Out(0).ready := true.B

  store_36.io.Out(0).ready := true.B



  /* ================================================================== *
   *                   Loop dependencies.                               *
   * ================================================================== */



  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

  loop_0.io.InLiveIn(0) <> int_mul_4.io.Out(0)

  loop_0.io.InLiveIn(1) <> int_cmp_3.io.Out(0)

  loop_0.io.InLiveIn(2) <> cast_2.io.Out(0)

  loop_0.io.InLiveIn(3) <> loop_1.io.OutLiveIn.elements("field2")(0)//FineGrainedArgCall.io.Out.data.elements("field2")(0)

  loop_0.io.InLiveIn(4) <> loop_1.io.OutLiveIn.elements("field0")(0)//FineGrainedArgCall.io.Out.data.elements("field0")(0)

  loop_0.io.InLiveIn(5) <> loop_1.io.OutLiveIn.elements("field1")(0)//FineGrainedArgCall.io.Out.data.elements("field1")(0)

  


  loop_1.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)

  loop_1.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

  loop_1.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)


  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  int_add_8.io.LeftIO <> loop_0.io.OutLiveIn.elements("field0")(0)

  int_andi_14.io.RightIO <> loop_0.io.OutLiveIn.elements("field1")(0)

  int_add_23.io.RightIO <> loop_0.io.OutLiveIn.elements("field2")(0)

  address_25.io.idx(0) <> loop_0.io.OutLiveIn.elements("field3")(0)

  address_35.io.idx(0) <> loop_0.io.OutLiveIn.elements("field3")(1)

  address_35.io.idx(0) <> loop_0.io.OutLiveIn.elements("field3")(2)

  address_28.io.baseAddress <> loop_0.io.OutLiveIn.elements("field4")(0)

  address_30.io.baseAddress <> loop_0.io.OutLiveIn.elements("field5")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */



  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_37.io.Out(1)//loop_0.io.CarryDepenOut.elements("field0")(0)

  merge_5.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  loop_1.io.CarryDepenIn(0) <> int_add_40.io.Out(1)//loop_1.io.CarryDepenOut.elements("field0")(0)

  merge_1.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)


   loop_1.io.loopExit(0) <>  return_43.io.In.enable
   state_branch_42.io.enable <>   loop_0.io.loopExit(0)
  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  merge_5.io.Mask <> exe_block_1.io.MaskBB(0)

  merge_1.io.Mask <> exe_block_2.io.MaskBB(0)

  merge_1.io.InData(0) <> int_const_0.io.Out

  int_cmp_3.io.LeftIO <> int_const_1.io.Out

  int_mul_4.io.LeftIO <> int_const_2.io.Out

  merge_5.io.InData(0) <> int_const_3.io.Out

  int_mul_7.io.LeftIO <> int_const_4.io.Out

  int_add_9.io.LeftIO <> int_const_5.io.Out

  int_cmp_10.io.LeftIO <> int_const_6.io.Out

  int_add_11.io.LeftIO <> int_const_7.io.Out

  int_cmp_12.io.LeftIO <> int_const_8.io.Out

  int_cmp_13.io.LeftIO <> int_const_9.io.Out

  int_mul_18.io.LeftIO <> int_const_10.io.Out

  int_add_19.io.LeftIO <> int_const_11.io.Out

  int_divsi21.io.LeftIO <> int_const_12.io.Out

  address_25.io.baseAddress <> int_const_13.io.Out

  address_32.io.baseAddress <> int_const_14.io.Out

  int_add_34.io.LeftIO <> int_const_15.io.Out

  address_35.io.baseAddress <> int_const_16.io.Out

  int_add_37.io.LeftIO <> int_const_17.io.Out

  int_cmp_38.io.LeftIO <> int_const_18.io.Out

  int_add_40.io.LeftIO <> int_const_19.io.Out

  int_cmp_41.io.LeftIO <> int_const_20.io.Out

  cast_2.io.Input <> merge_1.io.Out(0)

  int_mul_4.io.RightIO <> merge_1.io.Out(1)

  int_add_40.io.RightIO <> merge_1.io.Out(2)

  int_cmp_3.io.RightIO <> cast_2.io.Out(1)

  cast_6.io.Input <> merge_5.io.Out(0)

  int_mul_7.io.RightIO <> merge_5.io.Out(1)

  int_add_37.io.RightIO <> merge_5.io.Out(2)

  int_mul_18.io.RightIO <> cast_6.io.Out(0)

  int_add_19.io.RightIO <> cast_6.io.Out(1)

  int_mul_20.io.LeftIO <> cast_6.io.Out(2)

  int_add_8.io.RightIO <> int_mul_7.io.Out(0)

  int_add_11.io.RightIO <> int_mul_7.io.Out(1)

  int_add_9.io.RightIO <> int_add_8.io.Out(0)

  int_cmp_10.io.RightIO <> int_add_9.io.Out(0)

  int_andi_16.io.LeftIO <> int_cmp_10.io.Out(0)

  int_cmp_12.io.RightIO <> int_add_11.io.Out(0)

  int_cmp_13.io.RightIO <> int_add_11.io.Out(1)

  int_ori_15.io.LeftIO <> int_cmp_12.io.Out(0)

  int_andi_14.io.LeftIO <> int_cmp_13.io.Out(0)

  int_ori_15.io.RightIO <> int_andi_14.io.Out(0)

  int_andi_16.io.RightIO <> int_ori_15.io.Out(0)

  state_branch_17.io.CmpIO <> int_andi_16.io.Out(0)

  int_sub_22.io.LeftIO <> int_mul_18.io.Out(0)

  int_mul_20.io.RightIO <> int_add_19.io.Out(0)

  int_divsi21.io.RightIO <> int_mul_20.io.Out(0)

  int_sub_22.io.RightIO <> int_divsi21.io.Out(0)

  int_add_23.io.LeftIO <> int_sub_22.io.Out(0)

  cast_24.io.Input <> int_add_23.io.Out(0)

  address_30.io.idx(0) <> cast_24.io.Out(0)

  load_26.io.GepAddr <> address_25.io.Out(0)

  cast_27.io.Input <> load_26.io.Out(0)

  address_28.io.idx(0) <> cast_27.io.Out(0)
  address_32.io.idx(0) <> cast_27.io.Out(1)
  address_35.io.idx(0) <> cast_27.io.Out(2)
  load_29.io.GepAddr <> address_28.io.Out(0)

  store_31.io.inData <> load_29.io.Out(0)

  store_31.io.GepAddr <> address_30.io.Out(0)

  load_33.io.GepAddr <> address_32.io.Out(0)

  int_add_34.io.RightIO <> load_33.io.Out(0)

  store_36.io.inData <> int_add_34.io.Out(0)

  store_36.io.GepAddr <> address_35.io.Out(0)

  int_cmp_38.io.RightIO <> int_add_37.io.Out(0)

  state_branch_39.io.CmpIO <> int_cmp_38.io.Out(0)

  int_cmp_41.io.RightIO <> int_add_40.io.Out(0)

  state_branch_42.io.CmpIO <> int_cmp_41.io.Out(0)

  mem_ctrl_cache.io.rd.mem(0).MemReq <> load_26.io.MemReq

  load_26.io.MemResp <> mem_ctrl_cache.io.rd.mem(0).MemResp

  mem_ctrl_cache.io.rd.mem(1).MemReq <> load_29.io.MemReq

  load_29.io.MemResp <> mem_ctrl_cache.io.rd.mem(1).MemResp

  mem_ctrl_cache.io.rd.mem(2).MemReq <> load_33.io.MemReq

  load_33.io.MemResp <> mem_ctrl_cache.io.rd.mem(2).MemResp

  mem_ctrl_cache.io.wr.mem(0).MemReq <> store_31.io.MemReq

  store_31.io.MemResp <> mem_ctrl_cache.io.wr.mem(0).MemResp

  mem_ctrl_cache.io.wr.mem(1).MemReq <> store_36.io.MemReq

  store_36.io.MemResp <> mem_ctrl_cache.io.wr.mem(1).MemResp




 
  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_10.io.enable <> exe_block_0.io.Out(0)

  int_const_11.io.enable <> exe_block_0.io.Out(1)

  int_const_12.io.enable <> exe_block_0.io.Out(2)

  int_const_13.io.enable <> exe_block_0.io.Out(3)

  int_const_14.io.enable <> exe_block_0.io.Out(4)

  int_const_15.io.enable <> exe_block_0.io.Out(5)

  int_const_16.io.enable <> exe_block_0.io.Out(6)

  int_mul_18.io.enable <> exe_block_0.io.Out(7)

  int_add_19.io.enable <> exe_block_0.io.Out(8)

  int_mul_20.io.enable <> exe_block_0.io.Out(9)

  int_divsi21.io.enable <> exe_block_0.io.Out(10)

  int_sub_22.io.enable <> exe_block_0.io.Out(11)

  int_add_23.io.enable <> exe_block_0.io.Out(12)

  cast_24.io.enable <> exe_block_0.io.Out(13)

  address_25.io.enable <> exe_block_0.io.Out(14)

  load_26.io.enable <> exe_block_0.io.Out(15)

  cast_27.io.enable <> exe_block_0.io.Out(16)

  address_28.io.enable <> exe_block_0.io.Out(17)

  load_29.io.enable <> exe_block_0.io.Out(18)

  address_30.io.enable <> exe_block_0.io.Out(19)

  store_31.io.enable <> exe_block_0.io.Out(20)

  address_32.io.enable <> exe_block_0.io.Out(21)

  load_33.io.enable <> exe_block_0.io.Out(22)

  int_add_34.io.enable <> exe_block_0.io.Out(23)

  address_35.io.enable <> exe_block_0.io.Out(24)

  store_36.io.enable <> exe_block_0.io.Out(25)

  int_const_3.io.enable <> exe_block_1.io.Out(0)

  int_const_4.io.enable <> exe_block_1.io.Out(1)

  int_const_5.io.enable <> exe_block_1.io.Out(2)

  int_const_6.io.enable <> exe_block_1.io.Out(3)

  int_const_7.io.enable <> exe_block_1.io.Out(4)

  int_const_8.io.enable <> exe_block_1.io.Out(5)

  int_const_9.io.enable <> exe_block_1.io.Out(6)

  int_const_17.io.enable <> exe_block_1.io.Out(7)

  int_const_18.io.enable <> exe_block_1.io.Out(8)

  merge_5.io.enable <> exe_block_1.io.Out(9)

  cast_6.io.enable <> exe_block_1.io.Out(10)

  int_mul_7.io.enable <> exe_block_1.io.Out(11)

  int_add_8.io.enable <> exe_block_1.io.Out(12)

  int_add_9.io.enable <> exe_block_1.io.Out(13)

  int_cmp_10.io.enable <> exe_block_1.io.Out(14)

  int_add_11.io.enable <> exe_block_1.io.Out(15)

  int_cmp_12.io.enable <> exe_block_1.io.Out(16)

  int_cmp_13.io.enable <> exe_block_1.io.Out(17)

  int_andi_14.io.enable <> exe_block_1.io.Out(18)

  int_ori_15.io.enable <> exe_block_1.io.Out(19)

  int_andi_16.io.enable <> exe_block_1.io.Out(20)

  state_branch_17.io.enable <> exe_block_1.io.Out(21)

  int_add_37.io.enable <> exe_block_1.io.Out(22)

  int_cmp_38.io.enable <> exe_block_1.io.Out(23)

  state_branch_39.io.enable <> exe_block_1.io.Out(24)

  int_const_0.io.enable <> exe_block_2.io.Out(1)

  int_const_1.io.enable <> exe_block_2.io.Out(2)

  int_const_2.io.enable <> exe_block_2.io.Out(3)

  int_const_19.io.enable <> exe_block_2.io.Out(4)

  int_const_20.io.enable <> exe_block_2.io.Out(5)

  merge_1.io.enable <> exe_block_2.io.Out(6)

  cast_2.io.enable <> exe_block_2.io.Out(7)

  int_cmp_3.io.enable <> exe_block_2.io.Out(8)

  int_mul_4.io.enable <> exe_block_2.io.Out(9)

  int_add_40.io.enable <> exe_block_2.io.Out(10)

  int_cmp_41.io.enable <> exe_block_2.io.Out(11)

  // state_branch_42.io.enable <> exe_block_2.io.Out(12)

  io.out <> return_43.io.Out
}


