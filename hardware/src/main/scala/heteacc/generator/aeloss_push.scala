
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

abstract class aeloss_pushDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32, 32))))
	//   val MemResp = Flipped(Valid(new MemResp))
	//   val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List(32)))
	})
}

class aeloss_pushDF(implicit p: Parameters) extends aeloss_pushDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1 )))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  val mem_ctrl_cache = Module(new MemoryEngine(Size=2048, ID = 0, NumRead = 4, NumWrite = 0))
  // val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 1, NumWrite = 0))
  // io.MemReq <> mem_ctrl_cache.io.cache.MemReq
  // mem_ctrl_cache.io.cache.MemResp <> io.MemResp
  mem_ctrl_cache.initMem("dataset/aeloss_push/data.txt")


  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c0_i32 = arith.constant 0 : i32
  val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c0 = arith.constant 0 : index
  val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

  //%c0 = arith.constant 0 : index
  // val int_const_2 = Module(new ConstFastNode(value = 0, ID = 2))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_3 = Module(new ConstFastNode(value = 0, ID = 3))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_4 = Module(new ConstFastNode(value = 1, ID = 4))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_6 = Module(new ConstFastNode(value = 0, ID = 6))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_7 = Module(new ConstFastNode(value = 0, ID = 7))

  //%c94_i32 = arith.constant 94 : i32
  val int_const_8 = Module(new ConstFastNode(value = 94, ID = 8))

  //%c1 = arith.constant 1 : index
  val int_const_9 = Module(new ConstFastNode(value = 1, ID = 9))

  //%c1023 = arith.constant 1023 : index
  val int_const_10 = Module(new ConstFastNode(value = 1023, ID = 10))

  //%c1 = arith.constant 1 : index
  val int_const_11 = Module(new ConstFastNode(value = 1, ID = 11))

  //%c1023 = arith.constant 1023 : index
  val int_const_12 = Module(new ConstFastNode(value = 97, ID = 12))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 32, NumPhi = 1, BID = 0))

  val exe_block_1 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 14, NumPhi = 2, BID = 1))



  /* ================================================================== *
   *                   Printing Operation nodes. 37                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%4 = dataflow.merge %c0_i32 or %arg3 {Select = "Loop_Signal"} : i32
  val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 1, Res = false))

  //%5 = dataflow.merge %c0 or %arg2 {Select = "Loop_Signal"} : index
  val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 4, ID = 2, Res = false))

  //%6 = arith.index_cast %5 : index to i32
  val cast_3 = Module(new BitCastNode(NumOuts = 1, ID = 3))

  //%7 = dataflow.addr %arg1[%5] {memShape = [1024]} : memref<1024xi32>[index] -> i32
  val address_4 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 4)(ElementSize = 1, ArraySize = List()))

  //%8 = dataflow.load %7 : i32 -> i32
  val load_5 = Module(new Load( NumOuts = 1, ID = 5, RouteID = 0))

  //%9 = arith.trunci %8 : i32 to i1
  val trunc_6 = Module(new BitCastNode(NumOuts = 1, ID = 6))

  //%15 = dataflow.merge %4 or %arg5 {Select = "Loop_Signal"} : i32
  val merge_7 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 7, Res = false))

  //%16 = dataflow.merge %c0 or %arg4 {Select = "Loop_Signal"} : index
  // val merge_8 = Module(new MergeNode(NumInputs = 2, NumOutputs = 4, ID = 8, Res = false))

  //%17 = arith.index_cast %16 : index to i32
  val cast_9 = Module(new BitCastNode(NumOuts = 1, ID = 9))

  //%18 = dataflow.addr %arg0[%5] {memShape = [1024]} : memref<1024xi32>[index] -> i32
  val address_10 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 10)(ElementSize = 1, ArraySize = List()))

  //%19 = dataflow.load %18 : i32 -> i32
  val load_11 = Module(new Load(NumOuts = 1, ID = 11, RouteID = 1))

  //%20 = dataflow.addr %arg0[%16] {memShape = [1024]} : memref<1024xi32>[index] -> i32
  val address_12 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 12)(ElementSize = 1, ArraySize = List()))

  //%21 = dataflow.load %20 : i32 -> i32
  val load_13 = Module(new Load( NumOuts = 1, ID = 13, RouteID = 2))

  //%22 = arith.subi %19, %21 : i32
  val int_sub_14 = Module(new ComputeNodeWithoutState(NumOuts = 3, ID = 14, opCode = "Sub")(sign = false, Debug = false))

  //%23 = arith.cmpi ugt, %22, %c0_i32 : i32
  val int_cmp_15 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 15, opCode = "ugt")(sign = false, Debug = false))

  //%24 = arith.subi %c1_i32, %22 : i32
  val int_sub_16 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 16, opCode = "Sub")(sign = false, Debug = false))

  //%25 = arith.addi %22, %c1_i32 : i32
  val int_add_17 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 17, opCode = "Add")(sign = false, Debug = false))

  //%26 = arith.select %23, %24, %25 : i32
  val select_18 = Module(new SelectNode(NumOuts = 2, ID = 18))

  //%27 = arith.cmpi ugt, %26, %c0_i32 : i32
  val int_cmp_19 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 19, opCode = "ugt")(sign = false, Debug = false))

  //%28 = arith.select %27, %26, %c0_i32 : i32
  val select_20 = Module(new SelectNode(NumOuts = 1, ID = 20))

  //%29 = dataflow.addr %arg1[%16] {memShape = [1024]} : memref<1024xi32>[index] -> i32
  val address_21 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 21)(ElementSize = 1, ArraySize = List()))

  //%30 = dataflow.load %29 : i32 -> i32
  val load_22 = Module(new Load( NumOuts = 1, ID = 22, RouteID = 3))

  //%31 = arith.cmpi ne, %6, %17 : i32
  val int_cmp_23 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 23, opCode = "ne")(sign = false, Debug = false))

  //%32 = arith.trunci %30 : i32 to i1
  val trunc_24 = Module(new BitCastNode(NumOuts = 1, ID = 24))

  //%33 = arith.andi %32, %31 : i1
  val int_andi_25 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 25, opCode = "and")(sign = false, Debug = false))

  //%34 = arith.divsi %28, %c94_i32 : i32
  val int_divsi26 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 26, opCode = "udiv")(sign = false, Debug = false))

  //%35 = arith.addi %15, %34 : i32
  val int_add_27 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 27, opCode = "Add")(sign = false, Debug = false))

  //%36 = dataflow.select %33, %35, %15 : i32
  val select_28 = Module(new SelectNode(NumOuts = 2, ID = 28))

  //%37 = arith.addi %16, %c1 {Exe = "Loop"} : index
  val int_add_29 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 29, opCode = "Add")(sign = false, Debug = false))

  //%38 = arith.cmpi eq, %37, %c1023 {Exe = "Loop"} : index
  val int_cmp_30 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 30, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %38, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_31 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 31))

  //%11 = dataflow.select %9, %10, %4 : i32
  val select_32 = Module(new SelectNode(NumOuts = 2, ID = 32))

  //%12 = arith.addi %5, %c1 {Exe = "Loop"} : index
  val int_add_33 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 33, opCode = "Add")(sign = false, Debug = false))

  //%13 = arith.cmpi eq, %12, %c1023 {Exe = "Loop"} : index
  val int_cmp_34 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 34, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %13, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_35 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 35))

  //func.return %0 : i32
  val return_36 = Module(new RetNode2(retTypes = List(32), ID = 36))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(2, 1, 1, 1, 1), NumOuts = List(1), NumCarry = List(4, 1), NumExits = 1, ID = 0))

  val loop_1 = Module(new LoopBlockNode(NumIns = List(1, 2), NumOuts = List(1), NumCarry = List(1, 1), NumExits = 1, ID = 1))



  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_1.io.loopExit(0) <> return_36.io.In.enable
  
  loop_0.io.loopExit(0) <> state_branch_35.io.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back

  exe_block_1.io.predicateIn(0) <> loop_1.io.activate_loop_start

  exe_block_1.io.predicateIn(1) <> loop_1.io.activate_loop_back



  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> exe_block_1.io.Out(0)

  loop_0.io.loopBack(0) <> state_branch_31.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_31.io.TrueOutput(0)

  loop_1.io.enable <> state_branch_0.io.Out(0)

  loop_1.io.loopBack(0) <> state_branch_35.io.FalseOutput(0)

  loop_1.io.loopFinish(0) <> state_branch_35.io.TrueOutput(0)



  /* ================================================================== *
   *                   Loop dependencies.                               *
   * ================================================================== */



  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

  loop_0.io.InLiveIn(0) <> loop_1.io.OutLiveIn.elements("field0")(0)//FineGrainedArgCall.io.Out.data.elements("field0")(0)

  loop_0.io.InLiveIn(1) <> merge_2.io.Out(0)

  loop_0.io.InLiveIn(2) <> loop_1.io.OutLiveIn.elements("field1")(1)//FineGrainedArgCall.io.Out.data.elements("field1")(0)

  loop_0.io.InLiveIn(3) <> cast_3.io.Out(0)


  loop_0.io.InLiveIn(4) <> merge_1.io.Out(0)

  merge_7.io.InData(0) <>  loop_0.io.OutLiveIn.elements("field4")(0)//merge_1.io.Out(0)

  loop_1.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)
  
  loop_1.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)


  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  address_10.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  address_12.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(1)

  address_10.io.idx(0) <> loop_0.io.OutLiveIn.elements("field1")(0)

  address_21.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(0)

  int_cmp_23.io.LeftIO <> loop_0.io.OutLiveIn.elements("field3")(0)

  address_4.io.baseAddress <> loop_1.io.OutLiveIn.elements("field1")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> select_28.io.Out(0)

  loop_1.io.InLiveOut(0) <> select_32.io.Out(0)

  loop_0.io.OutLiveOut.elements("field0")(0) <> select_32.io.InData1

  loop_1.io.OutLiveOut.elements("field0")(0) <> return_36.io.In.data("field0")



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */



  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_29.io.Out(1)//loop_0.io.CarryDepenOut.elements("field0")(0)

  // merge_8.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0) 

  loop_0.io.CarryDepenIn(1) <> select_28.io.Out(1)//loop_0.io.CarryDepenOut.elements("field1")(0)

  merge_7.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(0)

  loop_1.io.CarryDepenIn(0) <> int_add_33.io.Out(1)//loop_1.io.CarryDepenOut.elements("field0")(0)

  merge_2.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)

  loop_1.io.CarryDepenIn(1) <> select_32.io.Out(1)//loop_1.io.CarryDepenOut.elements("field1")(0)

  merge_1.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field1")(0)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  merge_7.io.Mask <> exe_block_0.io.MaskBB(0)

  // merge_8.io.Mask <> exe_block_0.io.MaskBB(1)

  merge_1.io.Mask <> exe_block_1.io.MaskBB(0)

  merge_2.io.Mask <> exe_block_1.io.MaskBB(1)

  merge_1.io.InData(0) <> int_const_0.io.Out

  merge_2.io.InData(0) <> int_const_1.io.Out

  // merge_8.io.InData(0) <> int_const_2.io.Out

  int_cmp_15.io.LeftIO <> int_sub_14.io.Out(0)

  int_sub_16.io.LeftIO <> int_const_4.io.Out

  int_add_17.io.LeftIO <> int_const_5.io.Out

  int_cmp_19.io.LeftIO <> select_18.io.Out(0)

  select_20.io.Select <> int_const_7.io.Out

  int_divsi26.io.LeftIO <> select_20.io.Out(0)

  int_add_29.io.LeftIO <> int_const_9.io.Out

  int_cmp_30.io.LeftIO <> int_const_10.io.Out

  int_add_33.io.LeftIO <> int_const_11.io.Out

  int_cmp_34.io.LeftIO <> int_const_12.io.Out

  // merge_7.io.InData(0) <> merge_1.io.Out(0)

  cast_3.io.Input <> merge_2.io.Out(1)

  address_4.io.idx(0) <> merge_2.io.Out(2)

  int_add_33.io.RightIO <> merge_2.io.Out(3)

  load_5.GepAddr <> address_4.io.Out(0)

  trunc_6.io.Input <> load_5.io.Out(0)

  select_32.io.Select <> trunc_6.io.Out(0)



  select_32.io.InData2 <> merge_1.io.Out(1)//loop_0.io.OutLiveIn.elements("field4")(1)//merge_1.io.Out(1)
//////////////////////////////////////

  int_add_27.io.LeftIO <> merge_7.io.Out(0)

  select_28.io.InData2 <> merge_7.io.Out(1)

  cast_9.io.Input <> loop_0.io.CarryDepenOut.elements("field0")(0)//merge_8.io.Out(0)

  address_12.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(1)//merge_8.io.Out(1)

  address_21.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(2)//merge_8.io.Out(2)

  int_add_29.io.RightIO <> loop_0.io.CarryDepenOut.elements("field0")(3)//merge_8.io.Out(3)

  int_cmp_23.io.RightIO <> cast_9.io.Out(0)

  load_11.GepAddr <> address_10.io.Out(0)

  int_sub_14.io.LeftIO <> load_11.io.Out(0)

  load_13.GepAddr <> address_12.io.Out(0)

  int_sub_14.io.RightIO <> load_13.io.Out(0)

  int_cmp_15.io.RightIO <> int_const_3.io.Out

  int_sub_16.io.RightIO <> int_sub_14.io.Out(1)

  int_add_17.io.RightIO <> int_sub_14.io.Out(2)

  select_18.io.Select <> int_cmp_15.io.Out(0)

  select_18.io.InData1 <> int_sub_16.io.Out(0)

  select_18.io.InData2 <> int_add_17.io.Out(0)

  int_cmp_19.io.RightIO <> int_const_6.io.Out 

  select_20.io.InData2 <> select_18.io.Out(1)

  select_20.io.InData1 <> int_cmp_19.io.Out(0)

  int_divsi26.io.RightIO <> int_const_8.io.Out

  load_22.GepAddr <> address_21.io.Out(0)

  trunc_24.io.Input <> load_22.io.Out(0)

  int_andi_25.io.RightIO <> int_cmp_23.io.Out(0)

  int_andi_25.io.LeftIO <> trunc_24.io.Out(0)

  select_28.io.Select <> int_andi_25.io.Out(0)

  int_add_27.io.RightIO <> int_divsi26.io.Out(0)

  select_28.io.InData1 <> int_add_27.io.Out(0)

  int_cmp_30.io.RightIO <> int_add_29.io.Out(0)

  state_branch_31.io.CmpIO <> int_cmp_30.io.Out(0)

  int_cmp_34.io.RightIO <> int_add_33.io.Out(0)

  state_branch_35.io.CmpIO <> int_cmp_34.io.Out(0)
  
  mem_ctrl_cache.io.load_address(0) <> load_5.address_out

  load_5.data_in <> mem_ctrl_cache.io.load_data(0)

  mem_ctrl_cache.io.load_address(1) <> load_11.address_out

  load_11.data_in <> mem_ctrl_cache.io.load_data(1)

  mem_ctrl_cache.io.load_address(2) <> load_13.address_out

  load_13.data_in <> mem_ctrl_cache.io.load_data(2)

  mem_ctrl_cache.io.load_address(3) <> load_22.address_out

  load_22.data_in <> mem_ctrl_cache.io.load_data(3)



//   mem_ctrl_cache.io.rd.mem(0).MemReq <> load_5.io.MemReq

//   load_5.io.MemResp <> mem_ctrl_cache.io.rd.mem(0).MemResp

//   mem_ctrl_cache.io.rd.mem(1).MemReq <> load_11.io.MemReq

//   load_11.io.MemResp <> mem_ctrl_cache.io.rd.mem(1).MemResp

//   mem_ctrl_cache.io.rd.mem(2).MemReq <> load_13.io.MemReq

//   load_13.io.MemResp <> mem_ctrl_cache.io.rd.mem(2).MemResp

//   mem_ctrl_cache.io.rd.mem(3).MemReq <> load_22.io.MemReq

//   load_22.io.MemResp <> mem_ctrl_cache.io.rd.mem(3).MemResp



  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  // int_const_2.io.enable <> exe_block_0.io.Out(0)

  int_const_3.io.enable <> exe_block_0.io.Out(1)

  int_const_4.io.enable <> exe_block_0.io.Out(2)

  int_const_5.io.enable <> exe_block_0.io.Out(3)

  int_const_6.io.enable <> exe_block_0.io.Out(4)

  int_const_7.io.enable <> exe_block_0.io.Out(5)

  int_const_8.io.enable <> exe_block_0.io.Out(6)

  int_const_9.io.enable <> exe_block_0.io.Out(7)

  int_const_10.io.enable <> exe_block_0.io.Out(8)

  merge_7.io.enable <> exe_block_0.io.Out(9)

  // merge_8.io.enable <> exe_block_0.io.Out(10)

  cast_9.io.enable <> exe_block_0.io.Out(11)

  address_10.io.enable <> exe_block_0.io.Out(12)

  load_11.io.enable <> exe_block_0.io.Out(13)

  address_12.io.enable <> exe_block_0.io.Out(14)

  load_13.io.enable <> exe_block_0.io.Out(15)

  int_sub_14.io.enable <> exe_block_0.io.Out(16)

  int_cmp_15.io.enable <> exe_block_0.io.Out(17)

  int_sub_16.io.enable <> exe_block_0.io.Out(18)

  int_add_17.io.enable <> exe_block_0.io.Out(19)

  select_18.io.enable <> exe_block_0.io.Out(20)

  int_cmp_19.io.enable <> exe_block_0.io.Out(21)

  select_20.io.enable <> exe_block_0.io.Out(22)

  address_21.io.enable <> exe_block_0.io.Out(23)

  load_22.io.enable <> exe_block_0.io.Out(24)

  int_cmp_23.io.enable <> exe_block_0.io.Out(25)

  trunc_24.io.enable <> exe_block_0.io.Out(26)

  int_andi_25.io.enable <> exe_block_0.io.Out(27)

  int_divsi26.io.enable <> exe_block_0.io.Out(28)

  int_add_27.io.enable <> exe_block_0.io.Out(29)

  select_28.io.enable <> exe_block_0.io.Out(30)

  int_add_29.io.enable <> exe_block_0.io.Out(31)

  int_cmp_30.io.enable <> exe_block_0.io.Out(0)

  state_branch_31.io.enable <> exe_block_0.io.Out(10)

  int_const_0.io.enable <> exe_block_1.io.Out(1)

  int_const_1.io.enable <> exe_block_1.io.Out(2)

  int_const_11.io.enable <> exe_block_1.io.Out(3)

  int_const_12.io.enable <> exe_block_1.io.Out(4)

  merge_1.io.enable <> exe_block_1.io.Out(5)

  merge_2.io.enable <> exe_block_1.io.Out(6)

  cast_3.io.enable <> exe_block_1.io.Out(7)

  address_4.io.enable <> exe_block_1.io.Out(8)

  load_5.io.enable <> exe_block_1.io.Out(9)

  trunc_6.io.enable <> exe_block_1.io.Out(10)

  select_32.io.enable <> exe_block_1.io.Out(11)

  int_add_33.io.enable <> exe_block_1.io.Out(12)

  int_cmp_34.io.enable <> exe_block_1.io.Out(13)

  // state_branch_35.io.enable <> exe_block_1.io.Out(14)

  io.out <> return_36.io.Out

}

