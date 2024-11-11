
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

abstract class histogramDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32, 32, 32))))
	  // val MemResp = Flipped(Valid(new MemResp))
	  // val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List()))
	})
}

class histogramDF(implicit p: Parameters) extends histogramDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1, 1 )))
  FineGrainedArgCall.io.In <> io.in

  // //Cache
  // val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 3, NumWrite = 1))

  // io.MemReq <> mem_ctrl_cache.io.cache.MemReq
  // mem_ctrl_cache.io.cache.MemResp <> io.MemResp


  val mem_ctrl_cache = Module(new MemoryEngine(Size=1000, ID = 0, NumRead = 2, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/histogram/in.txt")
  

  val mem_ctrl_cache1 = Module(new MemoryEngine(Size=500, ID = 0, NumRead = 1, NumWrite = 1))
  mem_ctrl_cache1.initMem("dataset/histogram/out.txt")
  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c0 = arith.constant 0 : index
  // val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c1 = arith.constant 1 : index
  val int_const_1 = Module(new ConstFastNode(value = 1, ID = 1))

  //%c100 = arith.constant 100 : index
  val int_const_2 = Module(new ConstFastNode(value = 100, ID = 2))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 15, NumPhi = 0, BID = 0))



  /* ================================================================== *
   *                   Printing Operation nodes. 16                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%0 = dataflow.merge %c0 or %arg3 {Select = "Loop_Signal"} : index
  // val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 3, ID = 1, Res = false))

  //%1 = dataflow.addr %arg1[%0] {memShape = [100]} : memref<100xi32>[index] -> i32
  val address_2 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 2)(ElementSize = 1, ArraySize = List()))

  //%2 = dataflow.load %1 : i32 -> i32
  val load_3 = Module(new Load(NumOuts = 1, ID = 3, RouteID = 0))

  //%3 = dataflow.addr %arg0[%0] {memShape = [100]} : memref<100xi32>[index] -> i32
  val address_4 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 4)(ElementSize = 1, ArraySize = List()))

  //%4 = dataflow.load %3 : i32 -> i32
  val load_5 = Module(new Load(NumOuts = 1, ID = 5, RouteID = 1))

  //%5 = arith.index_cast %4 : i32 to index
  val cast_6 = Module(new BitCastNodeWithoutState(NumOuts = 2, ID = 6))

  //%6 = dataflow.addr %arg2[%5] {memShape = [100]} : memref<100xi32>[index] -> i32
  val address_7 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 7)(ElementSize = 1, ArraySize = List()))

  //%7 = dataflow.load %6 : i32 -> i32
  val load_8 = Module(new Load(NumOuts = 1, ID = 8, RouteID = 2))

  //%8 = arith.addi %7, %2 : i32
  val int_add_9 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 9, opCode = "Add")(sign = false, Debug = false))

  //%9 = dataflow.addr %arg2[%5] {memShape = [100]} : memref<100xi32>[index] -> i32
  val address_10 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 10)(ElementSize = 1, ArraySize = List()))

  //dataflow.store %8 %9 : i32 i32
  val store_11 = Module(new Store(NumOuts = 1, ID = 11, RouteID = 3))

  //%10 = arith.addi %0, %c1 {Exe = "Loop"} : index
  val int_add_12 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 12, opCode = "Add")(sign = false, Debug = false))

  //%11 = arith.cmpi eq, %10, %c100 {Exe = "Loop"} : index
  val int_cmp_13 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 13, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %11, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_14 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 14))

  //func.return
  val return_15 = Module(new RetNode2(retTypes = List(), ID = 15))


  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode1(NumIns = List(1, 1, 2), NumOuts = List(), NumCarry = List(3), NumExits = 1, ID = 0))

  loop_0.io.loopExit(0) <>return_15.io.In.enable

  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back



  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> state_branch_0.io.Out(0)

  loop_0.io.loopBack(0) <> state_branch_14.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_14.io.TrueOutput(0)

  store_11.io.Out(0).ready := true.B

  // state_branch_14.io.PredOp(0) <> store_11.io.SuccOp(0)



  /* ================================================================== *
   *                   Loop dependencies.                               *
   * ================================================================== */



  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

 loop_0.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)

  loop_0.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

  loop_0.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)



  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  address_2.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

  address_4.io.baseAddress <> loop_0.io.OutLiveIn.elements("field1")(0)

  address_7.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(0)

  address_10.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(1)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */



  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_12.io.Out(1)

  // merge_1.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  // merge_1.io.Mask <> exe_block_0.io.MaskBB(0)

  // merge_1.io.InData(0) <> int_const_0.io.Out

  int_add_12.io.LeftIO <> int_const_1.io.Out

  int_cmp_13.io.LeftIO <> int_const_2.io.Out

  address_2.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(0)//merge_1.io.Out(0)

  address_4.io.idx(0) <>loop_0.io.CarryDepenOut.elements("field0")(1)// merge_1.io.Out(1)

  int_add_12.io.RightIO <> loop_0.io.CarryDepenOut.elements("field0")(2)//merge_1.io.Out(2)

  load_3.GepAddr <> address_2.io.Out(0)

  int_add_9.io.RightIO <> load_3.io.Out(0)

  load_5.GepAddr <> address_4.io.Out(0)

  cast_6.io.Input <> load_5.io.Out(0)

  address_7.io.idx(0) <> cast_6.io.Out(0)

  address_10.io.idx(0) <> cast_6.io.Out(1)

  load_8.GepAddr <> address_7.io.Out(0)

  int_add_9.io.LeftIO <> load_8.io.Out(0)

  store_11.inData <> int_add_9.io.Out(0)

  store_11.GepAddr <> address_10.io.Out(0)

  int_cmp_13.io.RightIO <> int_add_12.io.Out(0)

  state_branch_14.io.CmpIO <> int_cmp_13.io.Out(0)


  mem_ctrl_cache.io.load_address(0) <> load_3.address_out

  load_3.data_in <> mem_ctrl_cache.io.load_data(0)

  mem_ctrl_cache.io.load_address(1) <> load_5.address_out

  load_5.data_in <> mem_ctrl_cache.io.load_data(1)


  mem_ctrl_cache1.io.load_address(0) <> load_8.address_out

  load_8.data_in <> mem_ctrl_cache1.io.load_data(0)

  mem_ctrl_cache1.io.store_address(0) <> store_11.address_out

  store_11.io.Out(0) <> mem_ctrl_cache1.io.store_data(0)



  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  // int_const_0.io.enable <> exe_block_0.io.Out(0)

  int_const_1.io.enable <> exe_block_0.io.Out(1)

  int_const_2.io.enable <> exe_block_0.io.Out(2)

  // merge_1.io.enable <> exe_block_0.io.Out(3)

  address_2.io.enable <> exe_block_0.io.Out(4)

  load_3.io.enable <> exe_block_0.io.Out(5)

  address_4.io.enable <> exe_block_0.io.Out(6)

  load_5.io.enable <> exe_block_0.io.Out(7)

  cast_6.io.enable <> exe_block_0.io.Out(8)

  address_7.io.enable <> exe_block_0.io.Out(9)

  load_8.io.enable <> exe_block_0.io.Out(10)

  int_add_9.io.enable <> exe_block_0.io.Out(11)

  address_10.io.enable <> exe_block_0.io.Out(12)

  store_11.io.enable <> exe_block_0.io.Out(13)

  int_add_12.io.enable <> exe_block_0.io.Out(14)

  int_cmp_13.io.enable <> exe_block_0.io.Out(0)

  state_branch_14.io.enable <> exe_block_0.io.Out(3)

  io.out <> return_15.io.Out

}

