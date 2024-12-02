
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

abstract class dropoutDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32, 32))))
	  val out = Decoupled(new Call(List(32)))
	})
}

class dropoutDF(implicit p: Parameters) extends dropoutDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1)))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  val mem_ctrl_cache = Module(new MemoryEngine(Size=1152, ID = 0, NumRead = 2, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/dropout/dropout.txt")

  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c7 = arith.constant 7 : index
  val int_const_0 = Module(new ConstFastNode(value = 7, ID = 0))

  //%c0 = arith.constant 0 : index
  val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

  //%c7 = arith.constant 7 : index
  val int_const_2 = Module(new ConstFastNode(value = 7, ID = 2))

  //%c0 = arith.constant 0 : index
  val int_const_3 = Module(new ConstFastNode(value = 0, ID = 3))

  //%c3_i32 = arith.constant 3 : i32
  val int_const_4 = Module(new ConstFastNode(value = 3, ID = 4))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_6 = Module(new ConstFastNode(value = 0, ID = 6))

  //%c2_i32 = arith.constant 2 : i32
  val int_const_7 = Module(new ConstFastNode(value = 2, ID = 7))

  //%c0_i32 = arith.constant 0 : i32
  val int_const_8 = Module(new ConstFastNode(value = 0, ID = 8))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_9 = Module(new ConstFastNode(value = 1, ID = 9))

  //%c1 = arith.constant 1 : index
  val int_const_10 = Module(new ConstFastNode(value = 1, ID = 10))

  //%c1024 = arith.constant 1024 : index
  val int_const_11 = Module(new ConstFastNode(value = 1024, ID = 11))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 32, NumPhi = 0, BID = 0))



  /* ================================================================== *
   *                   Printing Operation nodes. 24                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%4 = arith.index_cast %arg2 : index to i32
  // val cast_1 = Module(new BitCastNode(NumOuts = 1, ID = 1))

  //%5 = arith.andi %arg2, %c7 : index
  val int_andi_2 = Module(new ComputeNodeWithoutState(NumOuts = 3, ID = 2, opCode = "And")(sign = false, Debug = false))

  //%6 = arith.cmpi slt, %5, %c0 : index
  val int_cmp_3 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 3, opCode = "slt")(sign = false, Debug = false))

  //%7 = arith.addi %5, %c7 : index
  val int_add_4 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 4, opCode = "Add")(sign = false, Debug = false))

  //%8 = arith.select %6, %7, %5 : index
  val select_5 = Module(new SelectNode(NumOuts = 1, ID = 5))

  //%9 = arith.cmpi eq, %8, %c0 : index
  val int_cmp_6 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 6, opCode = "eq")(sign = false, Debug = false))

  //%10 = arith.shrui %4, %c3_i32 : i32
  val int_lshr_7 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 7, opCode = "lshr")(sign = false, Debug = false))

  //%11 = arith.index_cast %10 : i32 to index
  // val cast_8 = Module(new BitCastNode(NumOuts = 1, ID = 8))

  //%12 = dataflow.addr %arg1[%11] {memShape = [128]} : memref<128xi32>[index] -> i32
  val address_9 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 9)(ElementSize = 1, ArraySize = List()))

  //%13 = dataflow.load %12 : i32 -> i32
  val load_10 = Module(new Load(NumOuts = 1, ID = 10, RouteID = 0))

  //%14 = dataflow.select %9, %13, %arg4 : i32
  val select_11 = Module(new SelectNode(NumOuts = 2, ID = 11))

  //%15 = arith.andi %14, %c1_i32 : i32
  val int_andi_12 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 12, opCode = "and")(sign = false, Debug = false))

  //%16 = arith.cmpi ne, %15, %c0_i32 : i32
  val int_cmp_13 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 13, opCode = "ne")(sign = false, Debug = false))

  //%17 = dataflow.addr %arg0[%arg2] {memShape = [1024]} : memref<1024xi32>[index] -> i32
  val address_14 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 14)(ElementSize = 1, ArraySize = List()))

  //%18 = dataflow.load %17 : i32 -> i32
  val load_15 = Module(new Load(NumOuts = 1, ID = 15, RouteID = 1))

  //%19 = arith.muli %18, %c2_i32 : i32
  val int_mul_16 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 16, opCode = "Mul")(sign = false, Debug = false))

  //%20 = dataflow.select %16, %19, %c0_i32 : i32
  val select_17 = Module(new SelectNode(NumOuts = 1, ID = 17))

  //%21 = arith.addi %arg3, %20 : i32
  val int_add_18 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 18, opCode = "Add")(sign = false, Debug = false))

  //%22 = arith.shrsi %14, %c1_i32 : i32
  val int_shr_19 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 19, opCode = "ashr")(sign = false, Debug = false))

  //%23 = arith.addi %arg2, %c1 {Exe = "Loop"} : index
  val int_add_20 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 20, opCode = "Add")(sign = false, Debug = false))

  //%24 = arith.cmpi eq, %23, %c1024 {Exe = "Loop"} : index
  val int_cmp_21 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 21, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %24, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_22 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 22))

  //func.return %0 : i32
  val return_23 = Module(new RetNode2(retTypes = List(32), ID = 23))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1, 1), NumOuts = List(1), NumCarry = List(4, 1, 1), NumExits = 1, ID = 0))



  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_0.io.loopExit(0) <> return_23.io.In.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back



  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> state_branch_0.io.Out(0)

  loop_0.io.loopBack(0) <> state_branch_22.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_22.io.TrueOutput(0)



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

  address_9.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  address_14.io.baseAddress <> loop_0.io.OutLiveIn.elements("field1")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> int_add_18.io.Out(0)



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_23.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)



  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_20.io.Out(0)

  // cast_1.io.Input <> loop_0.io.CarryDepenOut.elements("field0")(0)

  int_andi_2.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(1)

  address_14.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(2)

  int_add_20.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(3)

  loop_0.io.CarryDepenIn(1) <> int_shr_19.io.Out(0)

  select_11.io.InData2 <> loop_0.io.CarryDepenOut.elements("field1")(0)

  loop_0.io.CarryDepenIn(2) <> int_add_18.io.Out(1)

  int_add_18.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field2")(0)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  int_andi_2.io.RightIO <> int_const_0.io.Out

  int_cmp_3.io.RightIO <> int_const_1.io.Out

  int_add_4.io.RightIO <> int_const_2.io.Out

  int_cmp_6.io.RightIO <> int_const_3.io.Out

  int_lshr_7.io.RightIO <> int_const_4.io.Out

  int_andi_12.io.RightIO <> int_const_5.io.Out

  int_cmp_13.io.RightIO <> int_const_6.io.Out

  int_mul_16.io.RightIO <> int_const_7.io.Out

  select_17.io.InData2 <> int_const_8.io.Out

  int_shr_19.io.RightIO <> int_const_9.io.Out

  int_add_20.io.RightIO <> int_const_10.io.Out

  int_cmp_21.io.RightIO <> int_const_11.io.Out

  int_lshr_7.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(0)//cast_1.io.Out(0)

  int_cmp_3.io.LeftIO <> int_andi_2.io.Out(0)

  int_add_4.io.LeftIO <> int_andi_2.io.Out(1)

  select_5.io.InData2 <> int_andi_2.io.Out(2)

  select_5.io.Select <> int_cmp_3.io.Out(0)

  select_5.io.InData1 <> int_add_4.io.Out(0)

  int_cmp_6.io.LeftIO <> select_5.io.Out(0)

  select_11.io.Select <> int_cmp_6.io.Out(0)

  // cast_8.io.Input <> int_lshr_7.io.Out(0)

  address_9.io.idx(0) <> int_lshr_7.io.Out(0)//cast_8.io.Out(0)

  load_10.GepAddr <> address_9.io.Out(0)

  select_11.io.InData1 <> load_10.io.Out(0)

  int_andi_12.io.LeftIO <> select_11.io.Out(0)

  int_shr_19.io.LeftIO <> select_11.io.Out(1)

  int_cmp_13.io.LeftIO <> int_andi_12.io.Out(0)

  select_17.io.Select <> int_cmp_13.io.Out(0)

  load_15.GepAddr <> address_14.io.Out(0)

  int_mul_16.io.LeftIO <> load_15.io.Out(0)

  select_17.io.InData1 <> int_mul_16.io.Out(0)

  int_add_18.io.RightIO <> select_17.io.Out(0)

  int_cmp_21.io.LeftIO <> int_add_20.io.Out(1)

  state_branch_22.io.CmpIO <> int_cmp_21.io.Out(0)

  mem_ctrl_cache.io.load_address(0) <> load_10.address_out

  load_10.data_in <> mem_ctrl_cache.io.load_data(0)

  mem_ctrl_cache.io.load_address(1) <> load_15.address_out

  load_15.data_in <> mem_ctrl_cache.io.load_data(1)



  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_0.io.enable <> exe_block_0.io.Out(0)

  int_const_1.io.enable <> exe_block_0.io.Out(1)

  int_const_2.io.enable <> exe_block_0.io.Out(2)

  int_const_3.io.enable <> exe_block_0.io.Out(3)

  int_const_4.io.enable <> exe_block_0.io.Out(4)

  int_const_5.io.enable <> exe_block_0.io.Out(5)

  int_const_6.io.enable <> exe_block_0.io.Out(6)

  int_const_7.io.enable <> exe_block_0.io.Out(7)

  int_const_8.io.enable <> exe_block_0.io.Out(8)

  int_const_9.io.enable <> exe_block_0.io.Out(9)

  int_const_10.io.enable <> exe_block_0.io.Out(10)

  int_const_11.io.enable <> exe_block_0.io.Out(11)

  // cast_1.io.enable <> exe_block_0.io.Out(12)

  int_andi_2.io.enable <> exe_block_0.io.Out(13)

  int_cmp_3.io.enable <> exe_block_0.io.Out(14)

  int_add_4.io.enable <> exe_block_0.io.Out(15)

  select_5.io.enable <> exe_block_0.io.Out(16)

  int_cmp_6.io.enable <> exe_block_0.io.Out(17)

  int_lshr_7.io.enable <> exe_block_0.io.Out(18)

  // cast_8.io.enable <> exe_block_0.io.Out(19)

  address_9.io.enable <> exe_block_0.io.Out(20)

  load_10.io.enable <> exe_block_0.io.Out(21)

  select_11.io.enable <> exe_block_0.io.Out(22)

  int_andi_12.io.enable <> exe_block_0.io.Out(23)

  int_cmp_13.io.enable <> exe_block_0.io.Out(24)

  address_14.io.enable <> exe_block_0.io.Out(25)

  load_15.io.enable <> exe_block_0.io.Out(26)

  int_mul_16.io.enable <> exe_block_0.io.Out(27)

  select_17.io.enable <> exe_block_0.io.Out(28)

  int_add_18.io.enable <> exe_block_0.io.Out(29)

  int_shr_19.io.enable <> exe_block_0.io.Out(30)

  int_add_20.io.enable <> exe_block_0.io.Out(31)

  int_cmp_21.io.enable <> exe_block_0.io.Out(12)

  state_branch_22.io.enable <> exe_block_0.io.Out(19)

  io.out <> return_23.io.Out

}

