
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

abstract class dropout_goldDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32, 32))))
	  // val MemResp = Flipped(Valid(new MemResp))
	  // val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List(32)))
	})
}

class dropout_goldDF(implicit p: Parameters) extends dropout_goldDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1 )))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  // val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 2, NumWrite = 0))

  // io.MemReq <> mem_ctrl_cache.io.cache.MemReq
  // mem_ctrl_cache.io.cache.MemResp <> io.MemResp


  val mem_ctrl_cache = Module(new MemoryEngine(Size=1152, ID = 0, NumRead = 2, NumWrite = 0))
  
  mem_ctrl_cache.initMem("dataset/memory/drop.txt")
  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c0_i32 = arith.constant 0 : i32
  val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

  //%c0 = arith.constant 0 : index
  // val int_const_2 = Module(new ConstFastNode(value = 0, ID = 2))

  //%c7 = arith.constant 7 : index
  val int_const_3 = Module(new ConstFastNode(value = 7, ID = 3))

  //%c0 = arith.constant 0 : index
  val int_const_4 = Module(new ConstFastNode(value = 0, ID = 4))

  //%c7 = arith.constant 7 : index
  val int_const_5 = Module(new ConstFastNode(value = 7, ID = 5))

  //%c0 = arith.constant 0 : index
  val int_const_6 = Module(new ConstFastNode(value = 0, ID = 6))

  //%c3_i32 = arith.constant 3 : i32
  val int_const_7 = Module(new ConstFastNode(value = 3, ID = 7))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_8 = Module(new ConstFastNode(value = 1, ID = 8))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_9 = Module(new ConstFastNode(value = 0, ID = 9))

  //%c2_i32 = arith.constant 2 : i32
  val int_const_10 = Module(new ConstFastNode(value = 2, ID = 10))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_11 = Module(new ConstFastNode(value = 0, ID = 11))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_12 = Module(new ConstFastNode(value = 1, ID = 12))

  //%c1 = arith.constant 1 : index
  val int_const_13 = Module(new ConstFastNode(value = 1, ID = 13))

  //%c1024 = arith.constant 1024 : index
  val int_const_14 = Module(new ConstFastNode(value = 1024, ID = 14))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 38, NumPhi = 2, BID = 0))



  /* ================================================================== *
   *                   Printing Operation nodes. 27                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%4 = dataflow.merge %c0_i32 or %arg3 {Select = "Loop_Signal"} : i32
  val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 1, ID = 1, Res = false))

  //%5 = dataflow.merge %c0_i32 or %arg4 {Select = "Loop_Signal"} : i32
  val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 1, ID = 2, Res = false))

  //%6 = dataflow.merge %c0 or %arg2 {Select = "Loop_Signal"} : index
  // val merge_3 = Module(new MergeNode(NumInputs = 2, NumOutputs = 4, ID = 3, Res = false))

  //%7 = arith.index_cast %6 : index to i32
  val cast_4 = Module(new BitCastNode(NumOuts = 1, ID = 4))

  //%8 = arith.andi %6, %c7 : index
  val int_addi_5 = Module(new ComputeNodeWithoutState(NumOuts = 3, ID = 5, opCode = "and")(sign = false, Debug = false))

  //%9 = arith.cmpi slt, %8, %c0 : index
  val int_cmp_6 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 6, opCode = "slt")(sign = false, Debug = false))

  //%10 = arith.addi %8, %c7 : index
  val int_add_7 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 7, opCode = "Add")(sign = false, Debug = false))

  //%11 = arith.select %9, %10, %8 : index
  val select_8 = Module(new SelectNodeWithoutState(NumOuts = 1, ID = 8))

  //%12 = arith.cmpi eq, %11, %c0 : index
  val int_cmp_9 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 9, opCode = "eq")(sign = false, Debug = false))

  //%13 = arith.shrui %7, %c3_i32 : i32
  val int_lshr_10 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 10, opCode = "lshr")(sign = false, Debug = false))

  //%14 = arith.index_cast %13 : i32 to index
  val cast_11 = Module(new BitCastNode(NumOuts = 1, ID = 11))

  //%15 = dataflow.addr %arg1[%14] {memShape = [128]} : memref<128xi32>[index] -> i32
  val address_12 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 12)(ElementSize = 1, ArraySize = List()))

  //%16 = dataflow.load %15 : i32 -> i32
  // val load_13 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 13, RouteID = 0))
  val load_13 = Module(new Load(NumOuts = 1, ID = 4, RouteID = 0))

  //%17 = dataflow.select %12, %16, %5 : i32
  val select_14 = Module(new SelectNodeWithoutState(NumOuts = 2, ID = 14))

  //%18 = arith.andi %17, %c1_i32 : i32
  val int_addi_15 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 15, opCode = "and")(sign = false, Debug = false))

  //%19 = arith.cmpi ne, %18, %c0_i32 : i32
  val int_cmp_16 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 16, opCode = "ne")(sign = false, Debug = false))

  //%20 = dataflow.addr %arg0[%6] {memShape = [1024]} : memref<1024xi32>[index] -> i32
  val address_17 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 17)(ElementSize = 1, ArraySize = List()))

  //%21 = dataflow.load %20 : i32 -> i32
  // val load_18 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 18, RouteID = 1))
  val load_18 = Module(new Load(NumOuts = 1, ID = 4, RouteID = 0))

  //%22 = arith.muli %21, %c2_i32 : i32
  val int_mul_19 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 19, opCode = "Mul")(sign = false, Debug = false))

  //%23 = dataflow.select %19, %22, %c0_i32 : i32
  val select_20 = Module(new SelectNodeWithoutState(NumOuts = 1, ID = 20))

  //%24 = arith.addi %4, %23 : i32
  val int_add_21 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 21, opCode = "Add")(sign = false, Debug = false))

  //%25 = arith.shrsi %17, %c1_i32 : i32
  val int_shr_22 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 22, opCode = "ashr")(sign = false, Debug = false))

  //%26 = arith.addi %6, %c1 {Exe = "Loop"} : index
  val int_add_23 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 23, opCode = "Add")(sign = false, Debug = false))

  //%27 = arith.cmpi eq, %26, %c1024 {Exe = "Loop"} : index
  val int_cmp_24 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 24, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %27, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_25 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 25))

  //func.return %0 : i32
  val return_26 = Module(new RetNode2(retTypes = List(32), ID = 26))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1, 1), NumOuts = List(1), NumCarry = List(1, 1, 4), NumExits = 1, ID = 0))



  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_0.io.loopExit(0) <> return_26.io.In.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back



  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> state_branch_0.io.Out(0)

  loop_0.io.loopBack(0) <> state_branch_25.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_25.io.TrueOutput(0)



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

  address_12.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  address_17.io.baseAddress <> loop_0.io.OutLiveIn.elements("field1")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> int_add_21.io.Out(0)

  

  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_26.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)



  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_21.io.Out(1)//loop_0.io.CarryDepenOut.elements("field0")(0)

  merge_1.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  loop_0.io.CarryDepenIn(1) <> int_shr_22.io.Out(0)

  merge_2.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(0)//loop_0.io.CarryDepenOut.elements("field1")(0)

  loop_0.io.CarryDepenIn(2) <> int_add_23.io.Out(1)

  // merge_3.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field2")(0)//loop_0.io.CarryDepenOut.elements("field2")(0)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  merge_1.io.Mask <> exe_block_0.io.MaskBB(0)

  merge_2.io.Mask <> exe_block_0.io.MaskBB(1)

  // merge_3.io.Mask <> exe_block_0.io.MaskBB(2)

  merge_1.io.InData(0) <> int_const_0.io.Out

  merge_2.io.InData(0) <> int_const_1.io.Out

  // merge_3.io.InData(0) <> int_const_2.io.Out

  int_addi_5.io.LeftIO <> int_const_3.io.Out

  int_cmp_6.io.LeftIO <> int_addi_5.io.Out(0)

  int_add_7.io.LeftIO <> int_const_5.io.Out

  int_cmp_9.io.LeftIO <> int_const_6.io.Out

  int_lshr_10.io.LeftIO <> cast_4.io.Out(0)

  int_addi_15.io.LeftIO <> int_const_8.io.Out

  int_cmp_16.io.LeftIO <> int_const_9.io.Out

  int_mul_19.io.LeftIO <> int_const_10.io.Out

  select_20.io.Select <> int_const_11.io.Out

  int_shr_22.io.LeftIO <> select_14.io.Out(1)

  int_add_23.io.LeftIO <> int_const_13.io.Out

  int_cmp_24.io.LeftIO <> int_const_14.io.Out

  int_add_21.io.LeftIO <> merge_1.io.Out(0)

  select_14.io.InData2 <> merge_2.io.Out(0)

  cast_4.io.Input <> loop_0.io.CarryDepenOut.elements("field2")(0)//merge_3.io.Out(0)

  int_addi_5.io.RightIO <> loop_0.io.CarryDepenOut.elements("field2")(1)//merge_3.io.Out(1)

  address_17.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field2")(2)//merge_3.io.Out(2)

  int_add_23.io.RightIO <> loop_0.io.CarryDepenOut.elements("field2")(3)//merge_3.io.Out(3)

  int_lshr_10.io.RightIO <> int_const_7.io.Out

  int_cmp_6.io.RightIO <> int_const_4.io.Out

  int_add_7.io.RightIO <> int_addi_5.io.Out(1)

  select_8.io.InData2 <> int_addi_5.io.Out(2)

  select_8.io.Select <> int_cmp_6.io.Out(0)

  select_8.io.InData1 <> int_add_7.io.Out(0)

  int_cmp_9.io.RightIO <> select_8.io.Out(0)

  select_14.io.Select <> int_cmp_9.io.Out(0)

  cast_11.io.Input <> int_lshr_10.io.Out(0)

  address_12.io.idx(0) <> cast_11.io.Out(0)

  load_13.GepAddr <> address_12.io.Out(0)

  select_14.io.InData1 <> load_13.io.Out(0)

  int_addi_15.io.RightIO <> select_14.io.Out(0)

  int_shr_22.io.RightIO <> int_const_12.io.Out

  int_cmp_16.io.RightIO <> int_addi_15.io.Out(0)

  select_20.io.InData1 <> int_cmp_16.io.Out(0)

  load_18.GepAddr <> address_17.io.Out(0)

  int_mul_19.io.RightIO <> load_18.io.Out(0)

  select_20.io.InData2 <> int_mul_19.io.Out(0)

  int_add_21.io.RightIO <> select_20.io.Out(0)

  int_cmp_24.io.RightIO <> int_add_23.io.Out(0)

  state_branch_25.io.CmpIO <> int_cmp_24.io.Out(0)

  // mem_ctrl_cache.io.rd.mem(0).MemReq <> load_13.io.MemReq

  // load_13.io.MemResp <> mem_ctrl_cache.io.rd.mem(0).MemResp

  // mem_ctrl_cache.io.rd.mem(1).MemReq <> load_18.io.MemReq

  // load_18.io.MemResp <> mem_ctrl_cache.io.rd.mem(1).MemResp



    mem_ctrl_cache.io.load_address(0) <> load_13.address_out

  load_13.data_in <> mem_ctrl_cache.io.load_data(0)

    mem_ctrl_cache.io.load_address(1) <> load_18.address_out

  load_18.data_in <> mem_ctrl_cache.io.load_data(1)
  


  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_0.io.enable <> exe_block_0.io.Out(0)

  int_const_1.io.enable <> exe_block_0.io.Out(1)

  // int_const_2.io.enable <> exe_block_0.io.Out(2)

  int_const_3.io.enable <> exe_block_0.io.Out(3)

  int_const_4.io.enable <> exe_block_0.io.Out(4)

  int_const_5.io.enable <> exe_block_0.io.Out(5)

  int_const_6.io.enable <> exe_block_0.io.Out(6)

  int_const_7.io.enable <> exe_block_0.io.Out(7)

  int_const_8.io.enable <> exe_block_0.io.Out(8)

  int_const_9.io.enable <> exe_block_0.io.Out(9)

  int_const_10.io.enable <> exe_block_0.io.Out(10)

  int_const_11.io.enable <> exe_block_0.io.Out(11)

  int_const_12.io.enable <> exe_block_0.io.Out(12)

  int_const_13.io.enable <> exe_block_0.io.Out(13)

  int_const_14.io.enable <> exe_block_0.io.Out(14)

  merge_1.io.enable <> exe_block_0.io.Out(15)

  merge_2.io.enable <> exe_block_0.io.Out(16)

  // merge_3.io.enable <> exe_block_0.io.Out(17)

  cast_4.io.enable <> exe_block_0.io.Out(18)

  int_addi_5.io.enable <> exe_block_0.io.Out(19)

  int_cmp_6.io.enable <> exe_block_0.io.Out(20)

  int_add_7.io.enable <> exe_block_0.io.Out(21)

  select_8.io.enable <> exe_block_0.io.Out(22)

  int_cmp_9.io.enable <> exe_block_0.io.Out(23)

  int_lshr_10.io.enable <> exe_block_0.io.Out(24)

  cast_11.io.enable <> exe_block_0.io.Out(25)

  address_12.io.enable <> exe_block_0.io.Out(26)

  load_13.io.enable <> exe_block_0.io.Out(27)

  select_14.io.enable <> exe_block_0.io.Out(28)

  int_addi_15.io.enable <> exe_block_0.io.Out(29)

  int_cmp_16.io.enable <> exe_block_0.io.Out(30)

  address_17.io.enable <> exe_block_0.io.Out(31)

  load_18.io.enable <> exe_block_0.io.Out(32)

  int_mul_19.io.enable <> exe_block_0.io.Out(33)

  select_20.io.enable <> exe_block_0.io.Out(34)

  int_add_21.io.enable <> exe_block_0.io.Out(35)

  int_shr_22.io.enable <> exe_block_0.io.Out(36)

  int_add_23.io.enable <> exe_block_0.io.Out(37)

  int_cmp_24.io.enable <> exe_block_0.io.Out(2)

  state_branch_25.io.enable <> exe_block_0.io.Out(17)

  io.out <> return_26.io.Out

}

