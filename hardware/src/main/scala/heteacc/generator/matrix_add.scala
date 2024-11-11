
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

abstract class matrix_addDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32, 32, 32, 32))))
	  // val MemResp = Flipped(Valid(new MemResp))
	  // val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List(32)))
	})
}

class matrix_addDF(implicit p: Parameters) extends matrix_addDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1, 1, 1 )))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  val mem_ctrl_cache = Module(new MemoryEngine(Size=200, ID = 0, NumRead = 4, NumWrite = 0))
  
  // val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 4, NumWrite = 0))

  mem_ctrl_cache.initMem("dataset/matirx_add/in.txt")
  // io <> mem_ctrl_cache.io.cache
  // mem_ctrl_cache.io.cache.MemResp <> io.MemResp



  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c0_i32 = arith.constant 0 : i32
  val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c0 = arith.constant 0 : index
  val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_2 = Module(new ConstFastNode(value = 0, ID = 2))

  //%c1 = arith.constant 1 : index
  val int_const_3 = Module(new ConstFastNode(value = 1, ID = 3))

  //%c50 = arith.constant 50 : index
  val int_const_4 = Module(new ConstFastNode(value = 50, ID = 4))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_5 = Module(new ConstFastNode(value = 0, ID = 5))

  //%c0 = arith.constant 0 : index
  val int_const_6 = Module(new ConstFastNode(value = 0, ID = 6))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_7 = Module(new ConstFastNode(value = 0, ID = 7))

  //%c1 = arith.constant 1 : index
  val int_const_8 = Module(new ConstFastNode(value = 1, ID = 8))

  //%c50 = arith.constant 50 : index
  val int_const_9 = Module(new ConstFastNode(value = 50, ID = 9))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_10 = Module(new ConstFastNode(value = 0, ID = 10))

  //%c0 = arith.constant 0 : index
  val int_const_11 = Module(new ConstFastNode(value = 0, ID = 11))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_12 = Module(new ConstFastNode(value = 0, ID = 12))

  //%c1 = arith.constant 1 : index
  val int_const_13 = Module(new ConstFastNode(value = 1, ID = 13))

  //%c50 = arith.constant 50 : index
  val int_const_14 = Module(new ConstFastNode(value = 50, ID = 14))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_15 = Module(new ConstFastNode(value = 0, ID = 15))

  //%c0 = arith.constant 0 : index
  val int_const_16 = Module(new ConstFastNode(value = 0, ID = 16))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_17 = Module(new ConstFastNode(value = 0, ID = 17))

  //%c1 = arith.constant 1 : index
  val int_const_18 = Module(new ConstFastNode(value = 1, ID = 18))

  //%c50 = arith.constant 50 : index
  val int_const_19 = Module(new ConstFastNode(value = 50, ID = 19))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 15, NumPhi = 2, BID = 0))

  val exe_block_1 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 15, NumPhi = 2, BID = 1))

  val exe_block_2 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 15, NumPhi = 2, BID = 2))

  val exe_block_3 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 15, NumPhi = 2, BID = 3))



  /* ================================================================== *
   *                   Printing Operation nodes. 45                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(NumOuts=4, ID = 0))

  //%8 = dataflow.merge %c0_i32 or %arg5 {Select = "Loop_Signal"} : i32
  val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 1, Res = false))

  //%9 = dataflow.merge %c0 or %arg4 {Select = "Loop_Signal"} : index
  val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 2, Res = false))

  //%10 = dataflow.addr %arg0[%9] {memShape = [50]} : memref<50xi32>[index] -> i32
  val address_3 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 3)(ElementSize = 1, ArraySize = List()))

  //%11 = dataflow.load %10 : i32 -> i32
  val load_4 = Module(new Load(NumOuts = 2, ID = 4, RouteID = 0))

  //%12 = arith.cmpi ne, %11, %c0_i32 : i32
  val int_cmp_5 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 5, opCode = "ne")(sign = false, Debug = false))

  //%13 = arith.addi %8, %11 : i32
  val int_add_6 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 6, opCode = "Add")(sign = false, Debug = false))

  //%14 = dataflow.select %12, %13, %8 : i32
  val select_7 = Module(new SelectNodeWithoutState(NumOuts = 2, ID = 7))

  //%15 = arith.addi %9, %c1 {Exe = "Loop"} : index
  val int_add_8 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 8, opCode = "Add")(sign = false, Debug = false))

  //%16 = arith.cmpi eq, %15, %c50 {Exe = "Loop"} : index
  val int_cmp_9 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 9, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %16, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_10 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 10))

  //%8 = dataflow.merge %c0_i32 or %arg5 {Select = "Loop_Signal"} : i32
  val merge_11 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 11, Res = false))

  //%9 = dataflow.merge %c0 or %arg4 {Select = "Loop_Signal"} : index
  val merge_12 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 12, Res = false))

  //%10 = dataflow.addr %arg1[%9] {memShape = [50]} : memref<50xi32>[index] -> i32
  val address_13 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 13)(ElementSize = 1, ArraySize = List()))

  //%11 = dataflow.load %10 : i32 -> i32
  val load_14 = Module(new Load(NumOuts = 2, ID = 14, RouteID = 1))

  //%12 = arith.cmpi ne, %11, %c0_i32 : i32
  val int_cmp_15 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 15, opCode = "ne")(sign = false, Debug = false))

  //%13 = arith.addi %8, %11 : i32
  val int_add_16 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 16, opCode = "Add")(sign = false, Debug = false))

  //%14 = dataflow.select %12, %13, %8 : i32
  val select_17 = Module(new SelectNodeWithoutState(NumOuts = 2, ID = 17))

  //%15 = arith.addi %9, %c1 {Exe = "Loop"} : index
  val int_add_18 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 18, opCode = "Add")(sign = false, Debug = false))

  //%16 = arith.cmpi eq, %15, %c50 {Exe = "Loop"} : index
  val int_cmp_19 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 19, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %16, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_20 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 20))

  //%8 = dataflow.merge %c0_i32 or %arg5 {Select = "Loop_Signal"} : i32
  val merge_21 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 21, Res = false))

  //%9 = dataflow.merge %c0 or %arg4 {Select = "Loop_Signal"} : index
  val merge_22 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 22, Res = false))

  //%10 = dataflow.addr %arg2[%9] {memShape = [50]} : memref<50xi32>[index] -> i32
  val address_23 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 23)(ElementSize = 1, ArraySize = List()))

  //%11 = dataflow.load %10 : i32 -> i32
  val load_24 = Module(new Load(NumOuts = 2, ID = 24, RouteID = 2))

  //%12 = arith.cmpi ne, %11, %c0_i32 : i32
  val int_cmp_25 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 25, opCode = "ne")(sign = false, Debug = false))

  //%13 = arith.addi %8, %11 : i32
  val int_add_26 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 26, opCode = "Add")(sign = false, Debug = false))

  //%14 = dataflow.select %12, %13, %8 : i32
  val select_27 = Module(new SelectNodeWithoutState(NumOuts = 2, ID = 27))

  //%15 = arith.addi %9, %c1 {Exe = "Loop"} : index
  val int_add_28 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 28, opCode = "Add")(sign = false, Debug = false))

  //%16 = arith.cmpi eq, %15, %c50 {Exe = "Loop"} : index
  val int_cmp_29 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 29, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %16, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_30 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 30))

  //%8 = dataflow.merge %c0_i32 or %arg5 {Select = "Loop_Signal"} : i32
  val merge_31 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 31, Res = false))

  //%9 = dataflow.merge %c0 or %arg4 {Select = "Loop_Signal"} : index
  val merge_32 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 32, Res = false))

  //%10 = dataflow.addr %arg3[%9] {memShape = [50]} : memref<50xi32>[index] -> i32
  val address_33 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 33)(ElementSize = 1, ArraySize = List()))

  //%11 = dataflow.load %10 : i32 -> i32
  val load_34 = Module(new Load(NumOuts = 2, ID = 34, RouteID = 3))

  //%12 = arith.cmpi ne, %11, %c0_i32 : i32
  val int_cmp_35 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 35, opCode = "ne")(sign = false, Debug = false))

  //%13 = arith.addi %8, %11 : i32
  val int_add_36 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 36, opCode = "Add")(sign = false, Debug = false))

  //%14 = dataflow.select %12, %13, %8 : i32
  val select_37 = Module(new SelectNodeWithoutState(NumOuts = 2, ID = 37))

  //%15 = arith.addi %9, %c1 {Exe = "Loop"} : index
  val int_add_38 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 38, opCode = "Add")(sign = false, Debug = false))

  //%16 = arith.cmpi eq, %15, %c50 {Exe = "Loop"} : index
  val int_cmp_39 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 39, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %16, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_40 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 40))

  //%6 = arith.addi %1, %2 : i32
  val int_add_41 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 41, opCode = "Add")(sign = false, Debug = false))

  //%7 = arith.addi %6, %3 : i32
  val int_add_42 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 42, opCode = "Add")(sign = false, Debug = false))

  //%8 = arith.addi %7, %4 : i32
  val int_add_43 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 43, opCode = "Add")(sign = false, Debug = false))

  //func.return %0 : i32
  val return_44 = Module(new RetNode2(retTypes = List(32), ID = 44))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1), NumOuts = List(1), NumCarry = List(1, 1), NumExits = 1, ID = 0))

  val loop_1 = Module(new LoopBlockNode(NumIns = List(1), NumOuts = List(1), NumCarry = List(1, 1), NumExits = 1, ID = 1))

  val loop_2 = Module(new LoopBlockNode(NumIns = List(1), NumOuts = List(1), NumCarry = List(1, 1), NumExits = 1, ID = 2))

  val loop_3 = Module(new LoopBlockNode(NumIns = List(1), NumOuts = List(1), NumCarry = List(1, 1), NumExits = 1, ID = 3))



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

  exe_block_3.io.predicateIn(0) <> loop_3.io.activate_loop_start

  exe_block_3.io.predicateIn(1) <> loop_3.io.activate_loop_back



  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> state_branch_0.io.Out(0)

  loop_0.io.loopBack(0) <> state_branch_10.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_10.io.TrueOutput(0)

  loop_1.io.enable <> state_branch_0.io.Out(1)

  loop_1.io.loopBack(0) <> state_branch_20.io.FalseOutput(0)

  loop_1.io.loopFinish(0) <> state_branch_20.io.TrueOutput(0)

  loop_2.io.enable <> state_branch_0.io.Out(2)

  loop_2.io.loopBack(0) <> state_branch_30.io.FalseOutput(0)

  loop_2.io.loopFinish(0) <> state_branch_30.io.TrueOutput(0)

  loop_3.io.enable <> state_branch_0.io.Out(3)

  loop_3.io.loopBack(0) <> state_branch_40.io.FalseOutput(0)

  loop_3.io.loopFinish(0) <> state_branch_40.io.TrueOutput(0)



  /* ================================================================== *
   *                   Loop dependencies.                               *
   * ================================================================== */



  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

  loop_0.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

  loop_1.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)

  loop_2.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)

  loop_3.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field3")(0)



  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  address_3.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  address_13.io.baseAddress <> loop_1.io.OutLiveIn.elements("field0")(0)

  address_23.io.baseAddress <> loop_2.io.OutLiveIn.elements("field0")(0)

  address_33.io.baseAddress <> loop_3.io.OutLiveIn.elements("field0")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> select_7.io.Out(0)

  loop_1.io.InLiveOut(0) <> select_17.io.Out(0)

  loop_2.io.InLiveOut(0) <> select_27.io.Out(0)

  loop_3.io.InLiveOut(0) <> select_37.io.Out(0)



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  

  loop_0.io.loopExit(0) <> int_add_41.io.enable
  
  loop_1.io.loopExit(0) <> int_add_42.io.enable

  loop_2.io.loopExit(0) <> int_add_43.io.enable

  loop_3.io.loopExit(0) <> return_44.io.In.enable


  int_add_41.io.LeftIO <> loop_0.io.OutLiveOut.elements("field0")(0)
  int_add_41.io.RightIO <>loop_1.io.OutLiveOut.elements("field0")(0)

  int_add_42.io.LeftIO <> loop_2.io.OutLiveOut.elements("field0")(0)
  int_add_42.io.RightIO <>int_add_41.io.Out(0)
  
  int_add_43.io.LeftIO <> loop_3.io.OutLiveOut.elements("field0")(0)
  int_add_43.io.RightIO <>int_add_42.io.Out(0)

  return_44.io.In.data("field0") <> int_add_43.io.Out(0)
  

  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_8.io.Out(1)

  merge_2.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  loop_0.io.CarryDepenIn(1) <> select_7.io.Out(1)

  merge_1.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(0)


  loop_1.io.CarryDepenIn(0) <> int_add_18.io.Out(1)

  merge_12.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)

  loop_1.io.CarryDepenIn(1) <> select_17.io.Out(1)

  merge_11.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field1")(0)


  loop_2.io.CarryDepenIn(0) <> int_add_28.io.Out(1)

  merge_22.io.InData(1) <> loop_2.io.CarryDepenOut.elements("field0")(0)

  loop_2.io.CarryDepenIn(1) <> select_27.io.Out(1)

  merge_21.io.InData(1) <> loop_2.io.CarryDepenOut.elements("field1")(0)


  loop_3.io.CarryDepenIn(0) <> int_add_38.io.Out(1)

  merge_32.io.InData(1) <> loop_3.io.CarryDepenOut.elements("field0")(0)

  loop_3.io.CarryDepenIn(1) <> select_37.io.Out(1)

  merge_31.io.InData(1) <> loop_3.io.CarryDepenOut.elements("field1")(0)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  merge_1.io.Mask <> exe_block_0.io.MaskBB(0)

  merge_2.io.Mask <> exe_block_0.io.MaskBB(1)

  merge_11.io.Mask <> exe_block_1.io.MaskBB(0)

  merge_12.io.Mask <> exe_block_1.io.MaskBB(1)

  merge_21.io.Mask <> exe_block_2.io.MaskBB(0)

  merge_22.io.Mask <> exe_block_2.io.MaskBB(1)

  merge_31.io.Mask <> exe_block_3.io.MaskBB(0)

  merge_32.io.Mask <> exe_block_3.io.MaskBB(1)

  merge_1.io.InData(0) <> int_const_0.io.Out

  merge_2.io.InData(0) <> int_const_1.io.Out

  int_cmp_5.io.LeftIO <> int_const_2.io.Out

  int_add_8.io.LeftIO <> int_const_3.io.Out

  int_cmp_9.io.LeftIO <> int_const_4.io.Out

  merge_11.io.InData(0) <> int_const_5.io.Out

  merge_12.io.InData(0) <> int_const_6.io.Out

  int_cmp_15.io.LeftIO <> int_const_7.io.Out

  int_add_18.io.LeftIO <> int_const_8.io.Out

  int_cmp_19.io.LeftIO <> int_const_9.io.Out

  merge_21.io.InData(0) <> int_const_10.io.Out

  merge_22.io.InData(0) <> int_const_11.io.Out

  int_cmp_25.io.LeftIO <> int_const_12.io.Out

  int_add_28.io.LeftIO <> int_const_13.io.Out

  int_cmp_29.io.LeftIO <> int_const_14.io.Out

  merge_31.io.InData(0) <> int_const_15.io.Out

  merge_32.io.InData(0) <> int_const_16.io.Out

  int_cmp_35.io.LeftIO <> int_const_17.io.Out

  int_add_38.io.LeftIO <> int_const_18.io.Out

  int_cmp_39.io.LeftIO <> int_const_19.io.Out

  int_add_6.io.LeftIO <> merge_1.io.Out(0)

  select_7.io.InData2 <> merge_1.io.Out(1)

  address_3.io.idx(0) <> merge_2.io.Out(0)

  int_add_8.io.RightIO <> merge_2.io.Out(1)

  load_4.GepAddr <> address_3.io.Out(0)

  int_cmp_5.io.RightIO <> load_4.io.Out(0)

  int_add_6.io.RightIO <> load_4.io.Out(1)

  select_7.io.Select <> int_cmp_5.io.Out(0)

  select_7.io.InData1 <> int_add_6.io.Out(0)

  int_cmp_9.io.RightIO <> int_add_8.io.Out(0)

  state_branch_10.io.CmpIO <> int_cmp_9.io.Out(0)

  int_add_16.io.LeftIO <> merge_11.io.Out(0)

  select_17.io.InData2 <> merge_11.io.Out(1)

  address_13.io.idx(0) <> merge_12.io.Out(0)

  int_add_18.io.RightIO <> merge_12.io.Out(1)

  load_14.GepAddr <> address_13.io.Out(0)

  int_cmp_15.io.RightIO <> load_14.io.Out(0)

  int_add_16.io.RightIO <> load_14.io.Out(1)

  select_17.io.Select <> int_cmp_15.io.Out(0)

  select_17.io.InData1 <> int_add_16.io.Out(0)

  int_cmp_19.io.RightIO <> int_add_18.io.Out(0)

  state_branch_20.io.CmpIO <> int_cmp_19.io.Out(0)

  int_add_26.io.LeftIO <> merge_21.io.Out(0)

  select_27.io.InData2 <> merge_21.io.Out(1)

  address_23.io.idx(0) <> merge_22.io.Out(0)

  int_add_28.io.RightIO <> merge_22.io.Out(1)

  load_24.GepAddr <> address_23.io.Out(0)

  int_cmp_25.io.RightIO <> load_24.io.Out(0)

  int_add_26.io.RightIO <> load_24.io.Out(1)

  select_27.io.Select <> int_cmp_25.io.Out(0)

  select_27.io.InData1 <> int_add_26.io.Out(0)

  int_cmp_29.io.RightIO <> int_add_28.io.Out(0)

  state_branch_30.io.CmpIO <> int_cmp_29.io.Out(0)

  int_add_36.io.LeftIO <> merge_31.io.Out(0)

  select_37.io.InData2 <> merge_31.io.Out(1)

  address_33.io.idx(0) <> merge_32.io.Out(0)

  int_add_38.io.RightIO <> merge_32.io.Out(1)

  load_34.GepAddr <> address_33.io.Out(0)

  int_cmp_35.io.RightIO <> load_34.io.Out(0)

  int_add_36.io.RightIO <> load_34.io.Out(1)

  select_37.io.Select <> int_cmp_35.io.Out(0)

  select_37.io.InData1 <> int_add_36.io.Out(0)

  int_cmp_39.io.RightIO <> int_add_38.io.Out(0)

  state_branch_40.io.CmpIO <> int_cmp_39.io.Out(0)

  

  mem_ctrl_cache.io.load_address(0) <> load_4.address_out

  load_4.data_in <> mem_ctrl_cache.io.load_data(0)

  mem_ctrl_cache.io.load_address(1) <> load_14.address_out

  load_14.data_in <> mem_ctrl_cache.io.load_data(1)

  mem_ctrl_cache.io.load_address(2) <> load_24.address_out

  load_24.data_in <> mem_ctrl_cache.io.load_data(2)

  mem_ctrl_cache.io.load_address(3) <> load_34.address_out

  load_34.data_in <> mem_ctrl_cache.io.load_data(3)



  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_0.io.enable <> exe_block_0.io.Out(0)

  int_const_1.io.enable <> exe_block_0.io.Out(1)

  int_const_2.io.enable <> exe_block_0.io.Out(2)

  int_const_3.io.enable <> exe_block_0.io.Out(3)

  int_const_4.io.enable <> exe_block_0.io.Out(4)

  merge_1.io.enable <> exe_block_0.io.Out(5)

  merge_2.io.enable <> exe_block_0.io.Out(6)

  address_3.io.enable <> exe_block_0.io.Out(7)

  load_4.io.enable <> exe_block_0.io.Out(8)

  int_cmp_5.io.enable <> exe_block_0.io.Out(9)

  int_add_6.io.enable <> exe_block_0.io.Out(10)

  select_7.io.enable <> exe_block_0.io.Out(11)

  int_add_8.io.enable <> exe_block_0.io.Out(12)

  int_cmp_9.io.enable <> exe_block_0.io.Out(13)

  state_branch_10.io.enable <> exe_block_0.io.Out(14)

  int_const_5.io.enable <> exe_block_1.io.Out(0)

  int_const_6.io.enable <> exe_block_1.io.Out(1)

  int_const_7.io.enable <> exe_block_1.io.Out(2)

  int_const_8.io.enable <> exe_block_1.io.Out(3)

  int_const_9.io.enable <> exe_block_1.io.Out(4)

  merge_11.io.enable <> exe_block_1.io.Out(5)

  merge_12.io.enable <> exe_block_1.io.Out(6)

  address_13.io.enable <> exe_block_1.io.Out(7)

  load_14.io.enable <> exe_block_1.io.Out(8)

  int_cmp_15.io.enable <> exe_block_1.io.Out(9)

  int_add_16.io.enable <> exe_block_1.io.Out(10)

  select_17.io.enable <> exe_block_1.io.Out(11)

  int_add_18.io.enable <> exe_block_1.io.Out(12)

  int_cmp_19.io.enable <> exe_block_1.io.Out(13)

  state_branch_20.io.enable <> exe_block_1.io.Out(14)

  int_const_10.io.enable <> exe_block_2.io.Out(0)

  int_const_11.io.enable <> exe_block_2.io.Out(1)

  int_const_12.io.enable <> exe_block_2.io.Out(2)

  int_const_13.io.enable <> exe_block_2.io.Out(3)

  int_const_14.io.enable <> exe_block_2.io.Out(4)

  merge_21.io.enable <> exe_block_2.io.Out(5)

  merge_22.io.enable <> exe_block_2.io.Out(6)

  address_23.io.enable <> exe_block_2.io.Out(7)

  load_24.io.enable <> exe_block_2.io.Out(8)

  int_cmp_25.io.enable <> exe_block_2.io.Out(9)

  int_add_26.io.enable <> exe_block_2.io.Out(10)

  select_27.io.enable <> exe_block_2.io.Out(11)

  int_add_28.io.enable <> exe_block_2.io.Out(12)

  int_cmp_29.io.enable <> exe_block_2.io.Out(13)

  state_branch_30.io.enable <> exe_block_2.io.Out(14)

  int_const_15.io.enable <> exe_block_3.io.Out(0)

  int_const_16.io.enable <> exe_block_3.io.Out(1)

  int_const_17.io.enable <> exe_block_3.io.Out(2)

  int_const_18.io.enable <> exe_block_3.io.Out(3)

  int_const_19.io.enable <> exe_block_3.io.Out(4)

  merge_31.io.enable <> exe_block_3.io.Out(5)

  merge_32.io.enable <> exe_block_3.io.Out(6)

  address_33.io.enable <> exe_block_3.io.Out(7)

  load_34.io.enable <> exe_block_3.io.Out(8)

  int_cmp_35.io.enable <> exe_block_3.io.Out(9)

  int_add_36.io.enable <> exe_block_3.io.Out(10)

  select_37.io.enable <> exe_block_3.io.Out(11)

  int_add_38.io.enable <> exe_block_3.io.Out(12)

  int_cmp_39.io.enable <> exe_block_3.io.Out(13)

  state_branch_40.io.enable <> exe_block_3.io.Out(14)

  io.out <> return_44.io.Out

}


