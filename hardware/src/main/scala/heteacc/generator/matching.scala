
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

abstract class matchingDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32, 32))))
	  // val MemResp = Flipped(Valid(new MemResp))
	  // val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List()))
	})
}

class matchingDF(implicit p: Parameters) extends matchingDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1 )))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  // val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 2, NumWrite = 1))

  // io.MemReq <> mem_ctrl_cache.io.cache.MemReq
  // mem_ctrl_cache.io.cache.MemResp <> io.MemResp

  val mem_ctrl_cache = Module(new MemoryEngine(Size=1720, ID = 0, NumRead = 2, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/matching/fint.txt")
  
  val mem_ctrl_cache_store = Module(new MemoryEngine(Size=660, ID = 0, NumRead = 0, NumWrite = 1))
  mem_ctrl_cache_store.initMem("dataset/matching/w.txt")

  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c0 = arith.constant 0 : index
  // val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c360_i32 = arith.constant 360 : i32
  val int_const_1 = Module(new ConstFastNode(value = 360, ID = 1))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_2 = Module(new ConstFastNode(value = 0, ID = 2))

  //%c0 = arith.constant 0 : index
  val int_const_3 = Module(new ConstFastNode(value = 0, ID = 3))

  //%c650_i32 = arith.constant 650 : i32
  val int_const_4 = Module(new ConstFastNode(value = 650, ID = 4))

  //%c-1 = arith.constant -1 : index
  val int_const_5 = Module(new ConstFastNode(value = -1, ID = 5))

  //%c349 = arith.constant 349 : index
  val int_const_6 = Module(new ConstFastNode(value = 349, ID = 6))

  //%c0 = arith.constant 0 : index
  val int_const_7 = Module(new ConstFastNode(value = 0, ID = 7))

  //%c650 = arith.constant 650 : index
  val int_const_8 = Module(new ConstFastNode(value = 650, ID = 8))

  //%c-360 = arith.constant -360 : index
  val int_const_9 = Module(new ConstFastNode(value = -360, ID = 9))

  //%c0 = arith.constant 0 : index
  val int_const_10 = Module(new ConstFastNode(value = 0, ID = 10))

  //%c-10_i32 = arith.constant -10 : i32
  val int_const_11 = Module(new ConstFastNode(value = -10, ID = 11))

  //%c1792_i32 = arith.constant 1792 : i32
  val int_const_12 = Module(new ConstFastNode(value = 1792, ID = 12))

  //%false = arith.constant false
  val int_const_13 = Module(new ConstFastNode(value = 0, ID = 13))

  //%c650 = arith.constant 650 : index
  val int_const_14 = Module(new ConstFastNode(value = 650, ID = 14))

  //%c-10 = arith.constant -10 : index
  val int_const_15 = Module(new ConstFastNode(value = -10, ID = 15))

  //%c-350 = arith.constant -350 : index
  val int_const_16 = Module(new ConstFastNode(value = -350, ID = 16))

  //%c0 = arith.constant 0 : index
  val int_const_17 = Module(new ConstFastNode(value = 0, ID = 17))

  //%c129_i32 = arith.constant 129 : i32
  val int_const_18 = Module(new ConstFastNode(value = 129, ID = 18))

  //%c1 = arith.constant 1 : index
  val int_const_19 = Module(new ConstFastNode(value = 1, ID = 19))

  //%c3 = arith.constant 3 : index
  val int_const_20 = Module(new ConstFastNode(value = 3, ID = 20))

  //%c2_i32 = arith.constant 2 : i32
  val int_const_21 = Module(new ConstFastNode(value = 2, ID = 21))

  //%c1 = arith.constant 1 : index
  val int_const_22 = Module(new ConstFastNode(value = 1, ID = 22))

  //%c660 = arith.constant 660 : index
  val int_const_23 = Module(new ConstFastNode(value = 660, ID = 23))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new ExecutionBlockNode(NumInputs = 1, NumOuts = 6, BID = 0))

  val exe_block_1 = Module(new ExecutionBlockNode(NumInputs = 1, NumOuts = 8,  BID = 1))

  val exe_block_2 = Module(new ExecutionBlockNode(NumInputs = 1, NumOuts = 7,  BID = 2))

  val exe_block_3 = Module(new ExecutionBlockNode(NumInputs = 1, NumOuts = 12, BID = 3))

  val exe_block_4 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 21, NumPhi = 2, BID = 4))

  val exe_block_5 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 12, NumPhi = 0, BID = 5))



  /* ================================================================== *
   *                   Printing Operation nodes. 46                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%0 = dataflow.merge %c0 or %arg2 {Select = "Loop_Signal"} : index
  // val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 4, ID = 1, Res = false))

  //%1 = arith.index_cast %0 : index to i32
  val cast_2 = Module(new BitCastNode(NumOuts = 2, ID = 2))

  //%2 = arith.cmpi slt, %1, %c360_i32 : i32
  val int_cmp_3 = Module(new ComputeNode(NumOuts = 1, ID = 3, opCode = "sgt")(sign = false, Debug = false))

  //%9 = dataflow.merge %c0_i32 or %arg4 {Select = "Loop_Signal"} : i32
  val merge_4 = Module(new MergeNode(NumInputs = 2, NumOutputs = 4, ID = 4, Res = false))

  //%10 = dataflow.merge %c0 or %arg3 {Select = "Loop_Signal"} : index
  val merge_5 = Module(new MergeNode(NumInputs = 2, NumOutputs = 4, ID = 5, Res = false))

  //%11 = arith.index_cast %10 : index to i32
  val cast_6 = Module(new BitCastNode(NumOuts = 1, ID = 6))

  //%12 = arith.muli %11, %c650_i32 : i32
  val int_mul_7 = Module(new ComputeNode(NumOuts = 1, ID = 7, opCode = "Mul")(sign = false, Debug = false))

  //%13 = arith.addi %1, %12 : i32
  val int_add_8 = Module(new ComputeNode(NumOuts = 1, ID = 8, opCode = "Add")(sign = false, Debug = false))

  //%14 = arith.muli %0, %c-1 : index
  val int_mul_9 = Module(new ComputeNode(NumOuts = 1, ID = 9, opCode = "Mul")(sign = false, Debug = false))

  //%15 = arith.addi %14, %c349 : index
  val int_add_10 = Module(new ComputeNode(NumOuts = 1, ID = 10, opCode = "Add")(sign = false, Debug = false))

  //%16 = arith.cmpi sge, %15, %c0 : index
  val int_cmp_11 = Module(new ComputeNode(NumOuts = 2, ID = 11, opCode = "gte")(sign = false, Debug = false))

  //dataflow.state %16, "if_then" or "if_else" : i1
  val state_branch_12 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 12))

  //%22 = arith.muli %10, %c650 : index
  val int_mul_13 = Module(new ComputeNode(NumOuts = 1, ID = 13, opCode = "Mul")(sign = false, Debug = false))

  //%23 = arith.addi %0, %22 : index
  val int_add_14 = Module(new ComputeNode(NumOuts = 1, ID = 14, opCode = "Add")(sign = false, Debug = false))

  //%24 = dataflow.addr %arg1[%23] {memShape = [1792]} : memref<1792xi32>[index] -> i32
  val address_15 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 15)(ElementSize = 32, ArraySize = List()))

  //%25 = dataflow.load %24 : i32 -> i32
  val load_16 = Module(new Load(NumOuts = 1, ID = 16, RouteID = 0))

  //%26 = arith.addi %9, %25 : i32
  val int_add_17 = Module(new ComputeNode(NumOuts = 1, ID = 17, opCode = "Add")(sign = false, Debug = false))

  //%22 = arith.addi %0, %c-360 : index
  val int_add_18 = Module(new ComputeNode(NumOuts = 1, ID = 18, opCode = "Add")(sign = false, Debug = false))

  //%23 = arith.cmpi sge, %22, %c0 : index
  val int_cmp_19 = Module(new ComputeNode(NumOuts = 1, ID = 19, opCode = "gte")(sign = false, Debug = false))

  //%24 = arith.addi %13, %c-10_i32 : i32
  val int_add_20 = Module(new ComputeNode(NumOuts = 1, ID = 20, opCode = "Add")(sign = false, Debug = false))

  //%25 = arith.cmpi slt, %24, %c1792_i32 : i32
  val int_cmp_21 = Module(new ComputeNode(NumOuts = 1, ID = 21, opCode = "sgt")(sign = false, Debug = false))

  //%26 = dataflow.select %23, %25, %false : i1
  val select_22 = Module(new SelectNodeWithoutState(NumOuts = 2, ID = 22))

  //dataflow.state %26, "if_then" or "if_else" : i1
  val state_branch_23 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 23))

  //%30 = arith.muli %10, %c650 : index
  val int_mul_24 = Module(new ComputeNode(NumOuts = 1, ID = 24, opCode = "Mul")(sign = false, Debug = false))

  //%31 = arith.addi %0, %30 : index
  val int_add_25 = Module(new ComputeNode(NumOuts = 1, ID = 25, opCode = "Add")(sign = false, Debug = false))

  //%32 = arith.addi %31, %c-10 : index
  val int_add_26 = Module(new ComputeNode(NumOuts = 1, ID = 26, opCode = "Add")(sign = false, Debug = false))

  //%33 = dataflow.addr %arg1[%32] {memShape = [1792]} : memref<1792xi32>[index] -> i32
  val address_27 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 27)(ElementSize = 32, ArraySize = List()))

  //%34 = dataflow.load %33 : i32 -> i32
  val load_28 = Module(new Load(NumOuts = 1, ID = 28, RouteID = 1))

  //%35 = arith.addi %9, %34 : i32
  val int_add_29 = Module(new ComputeNode(NumOuts = 1, ID = 29, opCode = "Add")(sign = false, Debug = false))

  //%30 = arith.addi %0, %c-350 : index
  val int_add_30 = Module(new ComputeNode(NumOuts = 1, ID = 30, opCode = "Add")(sign = false, Debug = false))

  //%31 = arith.cmpi sge, %30, %c0 : index
  val int_cmp_31 = Module(new ComputeNode(NumOuts = 1, ID = 31, opCode = "lte")(sign = false, Debug = false))

  //%32 = arith.select %2, %c129_i32, %9 : i32
  val select_32 = Module(new SelectNodeWithoutState(NumOuts = 1, ID = 32))

  //%33 = dataflow.select %31, %32, %9 : i32
  val select_33 = Module(new SelectNodeWithoutState(NumOuts = 1, ID = 33))

  //%28 = dataflow.select %26, %27, %27 {Data = "IF-THEN-ELSE", Select = "Data"} : i32
  val select_34 = Module(new SelectNodeWithoutState(NumOuts = 1, ID = 34))

  //%18 = dataflow.select %16, %17, %17 {Data = "IF-THEN-ELSE", Select = "Data"} : i32
  val select_35 = Module(new SelectNodeWithoutState(NumOuts = 2, ID = 35))

  //%19 = arith.addi %10, %c1 {Exe = "Loop"} : index
  val int_add_36 = Module(new ComputeNode(NumOuts = 2, ID = 36, opCode = "Add")(sign = false, Debug = false))

  //%20 = arith.cmpi eq, %19, %c3 {Exe = "Loop"} : index
  val int_cmp_37 = Module(new ComputeNode(NumOuts = 1, ID = 37, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %20, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_38 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 38))

  //%4 = arith.shrsi %3, %c2_i32 : i32 ashr
  val int_shr_39 = Module(new ComputeNode(NumOuts = 1, ID = 39, opCode = "ashr")(sign = false, Debug = false))

  //%5 = dataflow.addr %arg0[%0] {memShape = [660]} : memref<660xi32>[index] -> i32
  val address_40 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 40)(ElementSize = 32, ArraySize = List()))

  //dataflow.store %4 %5 : i32 i32
  val store_41 = Module(new Store(NumOuts = 1, ID = 41, RouteID = 2))

  //%6 = arith.addi %0, %c1 {Exe = "Loop"} : index
  val int_add_42 = Module(new ComputeNode(NumOuts = 2, ID = 42, opCode = "Add")(sign = false, Debug = false))

  //%7 = arith.cmpi eq, %6, %c660 {Exe = "Loop"} : index
  val int_cmp_43 = Module(new ComputeNode(NumOuts = 1, ID = 43, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %7, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_44 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 44))

  //func.return
  val return_45 = Module(new RetNode2(retTypes = List(), ID = 45))




  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1, 5, 2, 1), NumOuts = List(1), NumCarry = List(1, 1), NumExits = 1, ID = 0))

  val loop_1 = Module(new LoopBlockNode(NumIns = List(1,1), NumOuts = List( ), NumCarry = List(4), NumExits = 1, ID = 1))



  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  exe_block_0.io.predicateIn(0) <> state_branch_12.io.TrueOutput(0)

  exe_block_1.io.predicateIn(0) <> state_branch_23.io.TrueOutput(0)

  exe_block_2.io.predicateIn(0) <> state_branch_23.io.FalseOutput(0)

  exe_block_3.io.predicateIn(0) <> state_branch_12.io.FalseOutput(0)

  exe_block_4.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_4.io.predicateIn(1) <> loop_0.io.activate_loop_back

  exe_block_5.io.predicateIn(0) <> loop_1.io.activate_loop_start

  exe_block_5.io.predicateIn(1) <> loop_1.io.activate_loop_back



  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> exe_block_5.io.Out(0)

  loop_0.io.loopBack(0) <> state_branch_38.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_38.io.TrueOutput(0)

  loop_1.io.enable <> state_branch_0.io.Out(0)

  loop_1.io.loopBack(0) <> state_branch_44.io.FalseOutput(0)

  loop_1.io.loopFinish(0) <> state_branch_44.io.TrueOutput(0)

  store_41.io.Out(0).ready := true.B

  // state_branch_44.io.PredOp(0) <> store_41.io.SuccOp(0)



  /* ================================================================== *
   *                   Loop dependencies.                               *
   * ================================================================== */

  loop_1.io.loopExit(0) <>  return_45.io.In.enable

  loop_0.io.OutLiveOut.elements("field0")(0) <> int_shr_39.io.RightIO
  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

  loop_0.io.InLiveIn(0) <> cast_2.io.Out(0)

  loop_0.io.InLiveIn(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)//merge_1.io.Out(0)


  loop_0.io.InLiveIn(2) <>  loop_1.io.OutLiveIn.elements("field1")(0)

  loop_0.io.InLiveIn(3) <> int_cmp_3.io.Out(0)

  loop_1.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

  loop_1.io.InLiveIn(1) <>  FineGrainedArgCall.io.Out.data.elements("field1")(0) 

  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  int_add_8.io.LeftIO <> loop_0.io.OutLiveIn.elements("field0")(0)

  int_mul_9.io.RightIO <> loop_0.io.OutLiveIn.elements("field1")(0)

  int_add_14.io.LeftIO <> loop_0.io.OutLiveIn.elements("field1")(1)

  int_add_18.io.RightIO <> loop_0.io.OutLiveIn.elements("field1")(2)

  int_add_25.io.LeftIO <> loop_0.io.OutLiveIn.elements("field1")(3)

  int_add_30.io.RightIO <> loop_0.io.OutLiveIn.elements("field1")(4)

  address_15.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(0)

  address_27.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(1)

  select_32.io.InData1 <> loop_0.io.OutLiveIn.elements("field3")(0)

  address_40.io.baseAddress <> loop_1.io.OutLiveIn.elements("field0")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> select_35.io.Out(0)



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */



  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <>  select_35.io.Out(1)

  merge_4.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  loop_0.io.CarryDepenIn(1) <>  int_add_36.io.Out(1)

  merge_5.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(0)

  loop_1.io.CarryDepenIn(0) <>  int_add_42.io.Out(1)

  // merge_1.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  
  merge_4.io.Mask <> exe_block_4.io.MaskBB(0)

  merge_5.io.Mask <> exe_block_4.io.MaskBB(1)

  // merge_1.io.Mask <> exe_block_5.io.MaskBB(0)

  // merge_1.io.InData(0) <> int_const_0.io.Out

  int_cmp_3.io.LeftIO <> int_const_1.io.Out

  merge_4.io.InData(0) <> int_const_2.io.Out

  merge_5.io.InData(0) <> int_const_3.io.Out

  int_mul_7.io.LeftIO <> int_const_4.io.Out

  int_mul_9.io.LeftIO <> int_const_5.io.Out

  int_add_10.io.LeftIO <> int_const_6.io.Out

  int_cmp_11.io.LeftIO <> int_const_7.io.Out

  int_mul_13.io.LeftIO <> int_const_8.io.Out

  int_add_18.io.LeftIO <> int_const_9.io.Out

  int_cmp_19.io.LeftIO <> int_const_10.io.Out

  int_add_20.io.LeftIO <> int_const_11.io.Out

  int_cmp_21.io.LeftIO <> int_const_12.io.Out

  select_22.io.Select <> int_const_13.io.Out

  int_mul_24.io.LeftIO <> int_const_14.io.Out

  int_add_26.io.LeftIO <> int_const_15.io.Out

  int_add_30.io.LeftIO <> int_const_16.io.Out

  int_cmp_31.io.LeftIO <> int_const_17.io.Out

  select_32.io.Select <> int_const_18.io.Out

  int_add_36.io.LeftIO <> int_const_19.io.Out

  int_cmp_37.io.LeftIO <> int_const_20.io.Out

  int_shr_39.io.LeftIO <> int_const_21.io.Out

  int_add_42.io.LeftIO <> int_const_22.io.Out

  int_cmp_43.io.LeftIO <> int_const_23.io.Out

  cast_2.io.Input <> loop_1.io.CarryDepenOut.elements("field0")(1)//merge_1.io.Out(1)

  address_40.io.idx(0) <> loop_1.io.CarryDepenOut.elements("field0")(2)//merge_1.io.Out(2)

  int_add_42.io.RightIO <> loop_1.io.CarryDepenOut.elements("field0")(3)//merge_1.io.Out(3)

  int_cmp_3.io.RightIO <> cast_2.io.Out(1)

  int_add_17.io.LeftIO <> merge_4.io.Out(0)

  int_add_29.io.LeftIO <> merge_4.io.Out(1)

  select_32.io.InData2 <> merge_4.io.Out(2)

  select_33.io.InData2 <> merge_4.io.Out(3)

  cast_6.io.Input <> merge_5.io.Out(0)

  int_mul_13.io.RightIO <> merge_5.io.Out(1)

  int_mul_24.io.RightIO <> merge_5.io.Out(2)

  int_add_36.io.RightIO <> merge_5.io.Out(3)

  int_mul_7.io.RightIO <> cast_6.io.Out(0)

  int_add_8.io.RightIO <> int_mul_7.io.Out(0)

  int_add_20.io.RightIO <> int_add_8.io.Out(0)

  int_add_10.io.RightIO <> int_mul_9.io.Out(0)

  int_cmp_11.io.RightIO <> int_add_10.io.Out(0)

  select_35.io.Select <> int_cmp_11.io.Out(0)

  state_branch_12.io.CmpIO <> int_cmp_11.io.Out(1)

  int_add_14.io.RightIO <> int_mul_13.io.Out(0)

  address_15.io.idx(0) <> int_add_14.io.Out(0)

  load_16.GepAddr <> address_15.io.Out(0)

  int_add_17.io.RightIO <> load_16.io.Out(0)

  select_35.io.InData1 <> int_add_17.io.Out(0)

  int_cmp_19.io.RightIO <> int_add_18.io.Out(0)

  select_22.io.InData1 <> int_cmp_19.io.Out(0)

  int_cmp_21.io.RightIO <> int_add_20.io.Out(0)

  select_22.io.InData2 <> int_cmp_21.io.Out(0)

  select_34.io.Select <> select_22.io.Out(0)

  state_branch_23.io.CmpIO <> select_22.io.Out(1)

  int_add_25.io.RightIO <> int_mul_24.io.Out(0)

  int_add_26.io.RightIO <> int_add_25.io.Out(0)

  address_27.io.idx(0) <> int_add_26.io.Out(0)

  load_28.GepAddr <> address_27.io.Out(0)

  int_add_29.io.RightIO <> load_28.io.Out(0)

  select_34.io.InData1 <> int_add_29.io.Out(0)

  int_cmp_31.io.RightIO <> int_add_30.io.Out(0)

  select_33.io.Select <> int_cmp_31.io.Out(0)

  select_33.io.InData1 <> select_32.io.Out(0)

  select_34.io.InData2 <> select_33.io.Out(0)

  select_35.io.InData2 <> select_34.io.Out(0)

  int_cmp_37.io.RightIO <> int_add_36.io.Out(0)

  state_branch_38.io.CmpIO <> int_cmp_37.io.Out(0)

  store_41.inData <> int_shr_39.io.Out(0)

  store_41.GepAddr <> address_40.io.Out(0)

  int_cmp_43.io.RightIO <> int_add_42.io.Out(0)

  state_branch_44.io.CmpIO <> int_cmp_43.io.Out(0)

  // mem_ctrl_cache.io.rd.mem(0).MemReq <> load_16.io.MemReq

  // load_16.io.MemResp <> mem_ctrl_cache.io.rd.mem(0).MemResp

  // mem_ctrl_cache.io.rd.mem(1).MemReq <> load_28.io.MemReq

  // load_28.io.MemResp <> mem_ctrl_cache.io.rd.mem(1).MemResp

  // mem_ctrl_cache.io.wr.mem(0).MemReq <> store_41.io.MemReq

  // store_41.io.MemResp <> mem_ctrl_cache.io.wr.mem(0).MemResp


  
  mem_ctrl_cache.io.load_address(0) <> load_16.address_out

  load_16.data_in <> mem_ctrl_cache.io.load_data(0)


  mem_ctrl_cache.io.load_address(1) <> load_28.address_out

  load_28.data_in <> mem_ctrl_cache.io.load_data(1)


  mem_ctrl_cache_store.io.store_address(0) <> store_41.address_out

  store_41.io.Out(0) <> mem_ctrl_cache_store.io.store_data(0)



  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */
  state_branch_44.io.enable <>   loop_0.io.loopExit(0)

  int_const_8.io.enable <> exe_block_0.io.Out(0)

  int_mul_13.io.enable <> exe_block_0.io.Out(1)

  int_add_14.io.enable <> exe_block_0.io.Out(2)

  address_15.io.enable <> exe_block_0.io.Out(3)

  load_16.io.enable <> exe_block_0.io.Out(4)

  int_add_17.io.enable <> exe_block_0.io.Out(5)

  int_const_14.io.enable <> exe_block_1.io.Out(0)

  int_const_15.io.enable <> exe_block_1.io.Out(1)

  int_mul_24.io.enable <> exe_block_1.io.Out(2)

  int_add_25.io.enable <> exe_block_1.io.Out(3)

  int_add_26.io.enable <> exe_block_1.io.Out(4)

  address_27.io.enable <> exe_block_1.io.Out(5)

  load_28.io.enable <> exe_block_1.io.Out(6)

  int_add_29.io.enable <> exe_block_1.io.Out(7)

  int_const_16.io.enable <> exe_block_2.io.Out(0)

  int_const_17.io.enable <> exe_block_2.io.Out(1)

  int_const_18.io.enable <> exe_block_2.io.Out(2)

  int_add_30.io.enable <> exe_block_2.io.Out(3)

  int_cmp_31.io.enable <> exe_block_2.io.Out(4)

  select_32.io.enable <> exe_block_2.io.Out(5)

  select_33.io.enable <> exe_block_2.io.Out(6)

  int_const_9.io.enable <> exe_block_3.io.Out(0)

  int_const_10.io.enable <> exe_block_3.io.Out(1)

  int_const_11.io.enable <> exe_block_3.io.Out(2)

  int_const_12.io.enable <> exe_block_3.io.Out(3)

  int_const_13.io.enable <> exe_block_3.io.Out(4)

  int_add_18.io.enable <> exe_block_3.io.Out(5)

  int_cmp_19.io.enable <> exe_block_3.io.Out(6)

  int_add_20.io.enable <> exe_block_3.io.Out(7)

  int_cmp_21.io.enable <> exe_block_3.io.Out(8)

  select_22.io.enable <> exe_block_3.io.Out(9)

  state_branch_23.io.enable <> exe_block_3.io.Out(10)

  select_34.io.enable <> exe_block_3.io.Out(11)

  int_const_2.io.enable <> exe_block_4.io.Out(0)

  int_const_3.io.enable <> exe_block_4.io.Out(1)

  int_const_4.io.enable <> exe_block_4.io.Out(2)

  int_const_5.io.enable <> exe_block_4.io.Out(3)

  int_const_6.io.enable <> exe_block_4.io.Out(4)

  int_const_7.io.enable <> exe_block_4.io.Out(5)

  int_const_19.io.enable <> exe_block_4.io.Out(6)

  int_const_20.io.enable <> exe_block_4.io.Out(7)

  merge_4.io.enable <> exe_block_4.io.Out(8)

  merge_5.io.enable <> exe_block_4.io.Out(9)

  cast_6.io.enable <> exe_block_4.io.Out(10)

  int_mul_7.io.enable <> exe_block_4.io.Out(11)

  int_add_8.io.enable <> exe_block_4.io.Out(12)

  int_mul_9.io.enable <> exe_block_4.io.Out(13)

  int_add_10.io.enable <> exe_block_4.io.Out(14)

  int_cmp_11.io.enable <> exe_block_4.io.Out(15)

  state_branch_12.io.enable <> exe_block_4.io.Out(16)

  select_35.io.enable <> exe_block_4.io.Out(17)

  int_add_36.io.enable <> exe_block_4.io.Out(18)

  int_cmp_37.io.enable <> exe_block_4.io.Out(19)

  state_branch_38.io.enable <> exe_block_4.io.Out(20)

  // int_const_0.io.enable <> exe_block_5.io.Out(1)

  int_const_1.io.enable <> exe_block_5.io.Out(2)

  int_const_21.io.enable <> exe_block_5.io.Out(3)

  int_const_22.io.enable <> exe_block_5.io.Out(4)

  int_const_23.io.enable <> exe_block_5.io.Out(5)

  // merge_1.io.enable <> exe_block_5.io.Out(6)

  cast_2.io.enable <> exe_block_5.io.Out(7)

  int_cmp_3.io.enable <> exe_block_5.io.Out(8)

  int_shr_39.io.enable <> exe_block_5.io.Out(9)

  address_40.io.enable <> exe_block_5.io.Out(10)

  store_41.io.enable <> exe_block_5.io.Out(11)

  int_add_42.io.enable <> exe_block_5.io.Out(1)

  int_cmp_43.io.enable <> exe_block_5.io.Out(6)

  // state_branch_44.io.enable <> exe_block_5.io.Out(14)

  io.out <> return_45.io.Out

}

