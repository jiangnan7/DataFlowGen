
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

abstract class aeloss_pull_floatDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 64, 64, 64, 64))))
	  // val MemResp = Flipped(Valid(new MemResp))
	  // val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List(64)))
	})
}

class aeloss_pull_floatDF(implicit p: Parameters) extends aeloss_pull_floatDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1, 1, 1 )))
  FineGrainedArgCall.io.In <> io.in

 val mem_ctrl_cache1 = Module(new MemoryEngine(Size=3072, ID = 0, NumRead = 3, NumWrite = 0))
  mem_ctrl_cache1.initMem("dataset/aeloss_pull/in.txt")

  // val mem_ctrl_cache1 = Module(new MemoryEngine(Size=1024, ID = 0, NumRead = 1, NumWrite = 0))
  // mem_ctrl_cache1.initMem("dataset/aeloss_pull/in0.txt")

  // val mem_ctrl_cache2 = Module(new MemoryEngine(Size=1024, ID = 1, NumRead = 1, NumWrite = 0))
  // mem_ctrl_cache2.initMem("dataset/aeloss_pull/in1.txt")


  // val mem_ctrl_cache3 = Module(new MemoryEngine(Size=1024, ID = 2, NumRead = 1, NumWrite = 0))
  // mem_ctrl_cache3.initMem("dataset/aeloss_pull/in3.txt")
  
  val mem_ctrl_cache_store = Module(new MemoryEngine(Size=1024, ID = 3, NumRead = 0, NumWrite = 1))
  mem_ctrl_cache_store.initMem("dataset/aeloss_pull/out.txt")



  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%cst = arith.constant 0.000000e+00 : f64
  val float_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c0 = arith.constant 0 : index
  // val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

  //%cst_1 = arith.constant 5.000000e-01 : f64
  val float_const_2 = Module(new ConstFastNode(value = 2, ID = 2))

  //%cst_0 = arith.constant 0.097713504000000007 : f64
  val float_const_3 = Module(new ConstFastNode(value = 1, ID = 3))

  //%c1 = arith.constant 1 : index
  val int_const_4 = Module(new ConstFastNode(value = 1, ID = 4))

  //%c1024 = arith.constant 1024 : index
  val int_const_5 = Module(new ConstFastNode(value = 1024, ID = 5))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 28, NumPhi = 1, BID = 0))



  /* ================================================================== *
   *                   Printing Operation nodes. 26                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%4 = dataflow.merge %cst or %arg5 {Select = "Loop_Signal"} : f64
  val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 1, Res = false))

  //%5 = dataflow.merge %c0 or %arg4 {Select = "Loop_Signal"} : index
  // val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 5, ID = 2, Res = false))

  //%6 = dataflow.addr %arg0[%5] {memShape = [1024]} : memref<1024xf64>[index] -> i32
  val address_3 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 3)(ElementSize = 32, ArraySize = List()))

  //%7 = dataflow.load %6 : i32 -> f64
  val load_4 = Module(new Load( NumOuts = 2, ID = 4, RouteID = 0))

  //%8 = dataflow.addr %arg1[%5] {memShape = [1024]} : memref<1024xf64>[index] -> i32
  val address_5 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 5)(ElementSize = 32, ArraySize = List()))

  //%9 = dataflow.load %8 : i32 -> f64
  val load_6 = Module(new Load( NumOuts = 2, ID = 6, RouteID = 1))

  //%10 = arith.addf %7, %9 : f64
  val float_add_7 = Module(new FPComputeNode(NumOuts = 1, ID = 7, opCode = "Add")(t = FType.D))

  //%11 = arith.mulf %10, %cst_1 : f64
  val float_mul_8 = Module(new FPComputeNode(NumOuts = 3, ID = 8, opCode = "Mul")(t = FType.D))

  //%12 = dataflow.addr %arg2[%5] {memShape = [1024]} : memref<1024xf64>[index] -> i32
  val address_9 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 9)(ElementSize = 32, ArraySize = List()))

  //dataflow.store %11 %12 : f64 i32
  val store_10 = Module(new Store(NumOuts = 1, ID = 10, RouteID = 3))

  //%13 = arith.subf %7, %11 : f64
  val float_sub_11 = Module(new FPComputeNode(NumOuts = 2, ID = 11, opCode = "Sub")(t = FType.D))

  //%14 = arith.subf %9, %11 : f64
  val float_sub_12 = Module(new FPComputeNode(NumOuts = 2, ID = 12, opCode = "Sub")(t = FType.D))

  //%15 = arith.mulf %13, %13 : f64
  val float_mul_13 = Module(new FPComputeNode(NumOuts = 1, ID = 13, opCode = "Mul")(t = FType.D))

  //%16 = arith.mulf %14, %14 : f64
  val float_mul_14 = Module(new FPComputeNode(NumOuts = 1, ID = 14, opCode = "Mul")(t = FType.D))

  //%17 = arith.addf %15, %16 : f64
  val float_add_15 = Module(new FPComputeNode(NumOuts = 1, ID = 15, opCode = "Add")(t = FType.D))

  //%18 = arith.mulf %17, %cst_0 : f64
  val float_mul_16 = Module(new FPComputeNode(NumOuts = 1, ID = 16, opCode = "Mul")(t = FType.D))

  //%19 = dataflow.addr %arg3[%5] {memShape = [1024]} : memref<1024xi32>[index] -> i32
  val address_17 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 17)(ElementSize = 32, ArraySize = List()))

  //%20 = dataflow.load %19 : i32 -> i32
  val load_18 = Module(new Load( NumOuts = 1, ID = 18, RouteID = 2))

  //%21 = arith.trunci %20 : i32 to i1
  val trunc_19 = Module(new BitCastNode(NumOuts = 1, ID = 19))

  //%22 = arith.addf %4, %18 : f64
  val float_add_20 = Module(new FPComputeNode(NumOuts = 1, ID = 20, opCode = "Add")(t = FType.D))

  //%23 = dataflow.select %21, %22, %4 : f64
  val select_21 = Module(new SelectNode(NumOuts = 2, ID = 21))

  //%24 = arith.addi %5, %c1 {Exe = "Loop"} : index
  val int_add_22 = Module(new ComputeNode(NumOuts = 2, ID = 22, opCode = "Add")(sign = false, Debug = false))

  //%25 = arith.cmpi eq, %24, %c1024 {Exe = "Loop"} : index
  val int_cmp_23 = Module(new ComputeNode(NumOuts = 1, ID = 23, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %25, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_24 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 24))

  //func.return %0 : f64
  val return_25 = Module(new RetNode2(retTypes = List(32), ID = 25))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1, 1, 1, 1), NumOuts = List( 1), NumCarry = List(1, 5), NumExits = 1, ID = 0))




  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_0.io.loopExit(0) <> return_25.io.In.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back



  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> state_branch_0.io.Out(0)

  loop_0.io.loopBack(0) <> state_branch_24.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_24.io.TrueOutput(0)

  store_10.io.Out(0).ready := true.B

  // state_branch_24.io.PredOp(0) <> store_10.io.SuccOp(0)



  /* ================================================================== *
   *                   Loop dependencies.                               *
   * ================================================================== */



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

  address_3.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  address_5.io.baseAddress <> loop_0.io.OutLiveIn.elements("field1")(0)

  address_9.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(0)

  address_17.io.baseAddress <> loop_0.io.OutLiveIn.elements("field3")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> select_21.io.Out(0)



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_25.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)

  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> select_21.io.Out(1)//loop_0.io.CarryDepenOut.elements("field0")(0)

  merge_1.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  loop_0.io.CarryDepenIn(1) <> int_add_22.io.Out(1)
  // merge_1.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(1)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  merge_1.io.Mask <> exe_block_0.io.MaskBB(0)

  // merge_2.io.Mask <> exe_block_0.io.MaskBB(1)

  merge_1.io.InData(0) <> float_const_0.io.Out

  // merge_2.io.InData(0) <> int_const_1.io.Out

  float_mul_8.io.LeftIO <> float_const_2.io.Out

  float_mul_16.io.LeftIO <> float_const_3.io.Out

  int_add_22.io.LeftIO <> int_const_4.io.Out

  int_cmp_23.io.LeftIO <> int_const_5.io.Out

  float_add_20.io.LeftIO <> merge_1.io.Out(0)

  select_21.io.InData2 <> merge_1.io.Out(1)

   address_3.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field1")(0)//merge_2.io.Out(0)

  address_5.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field1")(1)//merge_2.io.Out(1)

  address_9.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field1")(2)//merge_2.io.Out(2)

  address_17.io.idx(0) <>loop_0.io.CarryDepenOut.elements("field1")(3) //merge_2.io.Out(3)


  int_add_22.io.RightIO <> loop_0.io.CarryDepenOut.elements("field1")(4)

  load_4.GepAddr <> address_3.io.Out(0)

  float_add_7.io.LeftIO <> load_4.io.Out(0)

  float_sub_11.io.LeftIO <> load_4.io.Out(1)

  load_6.GepAddr <> address_5.io.Out(0)

  float_add_7.io.RightIO <> load_6.io.Out(0)

  float_sub_12.io.LeftIO <> load_6.io.Out(1)

  float_mul_8.io.RightIO <> float_add_7.io.Out(0)

  store_10.inData <> float_mul_8.io.Out(0)

  float_sub_11.io.RightIO <> float_mul_8.io.Out(1)

  float_sub_12.io.RightIO <> float_mul_8.io.Out(2)

  store_10.GepAddr <> address_9.io.Out(0)

  float_mul_13.io.LeftIO <> float_sub_11.io.Out(0)

  float_mul_13.io.RightIO <> float_sub_11.io.Out(1)

  float_mul_14.io.LeftIO <> float_sub_12.io.Out(0)

  float_mul_14.io.RightIO <> float_sub_12.io.Out(1)

  float_add_15.io.LeftIO <> float_mul_13.io.Out(0)

  float_add_15.io.RightIO <> float_mul_14.io.Out(0)

  float_mul_16.io.RightIO <> float_add_15.io.Out(0)

  float_add_20.io.RightIO <> float_mul_16.io.Out(0)

  load_18.GepAddr <> address_17.io.Out(0)

  trunc_19.io.Input <> load_18.io.Out(0)

  select_21.io.Select <> trunc_19.io.Out(0)

  select_21.io.InData1 <> float_add_20.io.Out(0)

  int_cmp_23.io.RightIO <> int_add_22.io.Out(0)

  state_branch_24.io.CmpIO <> int_cmp_23.io.Out(0)

 
 
 mem_ctrl_cache1.io.load_address(0) <> load_4.address_out

  load_4.data_in <> mem_ctrl_cache1.io.load_data(0)


  mem_ctrl_cache1.io.load_address(1) <> load_6.address_out

  load_6.data_in <> mem_ctrl_cache1.io.load_data(1)
  
  mem_ctrl_cache1.io.load_address(2) <> load_18.address_out

  load_18.data_in <> mem_ctrl_cache1.io.load_data(2)


  mem_ctrl_cache_store.io.store_address(0) <> store_10.address_out

  store_10.io.Out(0) <> mem_ctrl_cache_store.io.store_data(0)



  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  float_const_0.io.enable <> exe_block_0.io.Out(0)

  // int_const_1.io.enable <> exe_block_0.io.Out(1)

  float_const_2.io.enable <> exe_block_0.io.Out(2)

  float_const_3.io.enable <> exe_block_0.io.Out(3)

  int_const_4.io.enable <> exe_block_0.io.Out(4)

  int_const_5.io.enable <> exe_block_0.io.Out(5)

  merge_1.io.enable <> exe_block_0.io.Out(6)

  // merge_2.io.enable <> exe_block_0.io.Out(7)

  address_3.io.enable <> exe_block_0.io.Out(8)

  load_4.io.enable <> exe_block_0.io.Out(9)

  address_5.io.enable <> exe_block_0.io.Out(10)

  load_6.io.enable <> exe_block_0.io.Out(11)

  float_add_7.io.enable <> exe_block_0.io.Out(12)

  float_mul_8.io.enable <> exe_block_0.io.Out(13)

  address_9.io.enable <> exe_block_0.io.Out(14)

  store_10.io.enable <> exe_block_0.io.Out(15)

  float_sub_11.io.enable <> exe_block_0.io.Out(16)

  float_sub_12.io.enable <> exe_block_0.io.Out(17)

  float_mul_13.io.enable <> exe_block_0.io.Out(18)

  float_mul_14.io.enable <> exe_block_0.io.Out(19)

  float_add_15.io.enable <> exe_block_0.io.Out(20)

  float_mul_16.io.enable <> exe_block_0.io.Out(21)

  address_17.io.enable <> exe_block_0.io.Out(22)

  load_18.io.enable <> exe_block_0.io.Out(23)

  trunc_19.io.enable <> exe_block_0.io.Out(24)

  float_add_20.io.enable <> exe_block_0.io.Out(25)

  select_21.io.enable <> exe_block_0.io.Out(26)

  int_add_22.io.enable <> exe_block_0.io.Out(27)

  int_cmp_23.io.enable <> exe_block_0.io.Out(1)

  state_branch_24.io.enable <> exe_block_0.io.Out(7)

  io.out <> return_25.io.Out

}

