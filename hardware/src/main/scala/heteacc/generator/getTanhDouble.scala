
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

abstract class getTanhDoubleDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32))))
	  // val MemResp = Flipped(Valid(new MemResp))
	  // val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List(32)))
	})
}

class getTanhDoubleDF(implicit p: Parameters) extends getTanhDoubleDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1)))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  // val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 1, NumWrite = 0))

  // io.MemReq <> mem_ctrl_cache.io.cache.MemReq
  // mem_ctrl_cache.io.cache.MemResp <> io.MemResp
  
  val mem_ctrl_cache = Module(new MemoryEngine(Size=100, ID = 0, NumRead = 1, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/memory/gettanh.txt")

  // val mem_ctrl_cache = Module(new MemoryEngine(Size = 100, ID = 0, NumRead = 1, NumWrite = 0))
  // io.MemReq <> mem_ctrl_cache.io.cache.MemReq
  // mem_ctrl_cache.io.cache.MemResp <> io.MemResp
  // mem_ctrl_cache.initMem("dataset/memory/in_0.txt")

  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c0_i32 = arith.constant 0 : i32
  // val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c0 = arith.constant 0 : index
  // val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_2 = Module(new ConstFastNode(value = 1, ID = 2))

  //%c19_i32 = arith.constant 19 : i32
  val int_const_3 = Module(new ConstFastNode(value = 19, ID = 3))

  //%c3_i32 = arith.constant 3 : i32
  val int_const_4 = Module(new ConstFastNode(value = 3, ID = 4))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

  //%c1 = arith.constant 1 : index
  val int_const_6 = Module(new ConstFastNode(value = 1, ID = 6))

  //%c100 = arith.constant 100 : index
  val int_const_7 = Module(new ConstFastNode(value = 100, ID = 7))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 14, NumPhi = 0, BID = 0))



  /* ================================================================== *
   *                   Printing Operation nodes. 18                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%4 = dataflow.merge %c0_i32 or %arg3 {Select = "Loop_Signal"} : i32
  // val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 1, ID = 1, Res = false))

  // //%5 = dataflow.merge %c0 or %arg2 {Select = "Loop_Signal"} : index
  // val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 2, Res = false))

  //%6 = dataflow.addr %arg0[%5] {memShape = [100]} : memref<100xi32>[index] -> i32
  val address_3 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 3)(ElementSize = 1, ArraySize = List()))

  //%7 = dataflow.load %6 : i32 -> i32
  // val load_4 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 6, ID = 4, RouteID = 0))
  val load_4 = Module(new Load(NumOuts = 6, ID = 4, RouteID = 0))
  //%8 = arith.cmpi slt, %7, %c1_i32 : i32
  val int_cmp_5 = Module(new ComputeNode(NumOuts = 1, ID = 5, opCode = "slt")(sign = false, Debug = false))



val m0 = Module(new Chain(NumOps = 6, ID = 0, OpCodes = Array("Mul","Add","Mul","Mul","Add","Mul"))(sign = false)(p))


  //%9 = arith.muli %7, %7 : i32
  // val int_mul_6 = Module(new ComputeNode(NumOuts = 1, ID = 6, opCode = "Mul")(sign = false, Debug = false))

  // //%10 = arith.addi %9, %c19_i32 : i32
  // val int_add_7 = Module(new ComputeNode(NumOuts = 1, ID = 7, opCode = "Add")(sign = false, Debug = false))

  // //%11 = arith.muli %10, %7 : i32
  // val int_mul_8 = Module(new ComputeNode(NumOuts = 1, ID = 8, opCode = "Mul")(sign = false, Debug = false))

  // //%12 = arith.muli %11, %7 : i32
  // val int_mul_9 = Module(new ComputeNode(NumOuts = 1, ID = 9, opCode = "Mul")(sign = false, Debug = false))

  // //%13 = arith.addi %12, %c3_i32 : i32
  // val int_add_10 = Module(new ComputeNode(NumOuts = 1, ID = 10, opCode = "Add")(sign = false, Debug = false))

  // //%14 = arith.muli %13, %7 : i32
  // val int_mul_11 = Module(new ComputeNode(NumOuts = 1, ID = 11, opCode = "Mul")(sign = false, Debug = false))

  //%15 = dataflow.select %8, %14, %c1_i32 : i32
  val select_12 = Module(new SelectNodeWithoutState(NumOuts = 1, ID = 12))

  //%16 = arith.addi %4, %15 : i32
  val int_add_13 = Module(new ComputeNode(NumOuts = 2, ID = 13, opCode = "Add")(sign = false, Debug = false))

val m1 = Module(new Chain(NumOps = 2, ID = 0, OpCodes = Array("Add","eq"))(sign = false)(p))


  //%17 = arith.addi %5, %c1 {Exe = "Loop"} : index
  // val int_add_14 = Module(new ComputeNode(NumOuts = 2, ID = 14, opCode = "Add")(sign = false, Debug = false))

  //%18 = arith.cmpi eq, %17, %c100 {Exe = "Loop"} : index
  // val int_cmp_15 = Module(new ComputeNode(NumOuts = 1, ID = 15, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %18, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_16 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 16))

  //func.return %0 : i32
  val return_17 = Module(new RetNode2(retTypes = List(32), ID = 17))

 m1.io.Out(1).ready := true.B

  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1), NumOuts = List(1), NumCarry = List(2, 1), NumExits = 1, ID = 0))



  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_0.io.loopExit(0) <> return_17.io.In.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back



  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> state_branch_0.io.Out(0)

  loop_0.io.loopBack(0) <> state_branch_16.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_16.io.TrueOutput(0)



  /* ================================================================== *
   *                   Loop dependencies.                               *
   * ================================================================== */



  /* ================================================================== *
   *                   Input Data dependencies.                         *
   * ================================================================== */

  loop_0.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)



  /* ================================================================== *
   *                   Live-in dependencies.                            *
   * ================================================================== */

  address_3.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> int_add_13.io.Out(0)



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_17.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)


m1.io.In(0) <>int_const_6.io.Out
  m1.io.In(1) <>loop_0.io.CarryDepenOut.elements("field0")(1)
  m1.io.In(2) <>int_const_7.io.Out

  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> m1.io.Out(0)

  // merge_2.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  loop_0.io.CarryDepenIn(1) <> int_add_13.io.Out(1)

  // merge_1.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(0)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  // merge_1.io.Mask <> exe_block_0.io.MaskBB(0)

  // merge_2.io.Mask <> exe_block_0.io.MaskBB(1)

  // merge_1.io.InData(0) <> int_const_0.io.Out

  // merge_2.io.InData(0) <> int_const_1.io.Out
  m0.io.In(0) <>load_4.io.Out(1)
  m0.io.In(1) <>load_4.io.Out(2)
  m0.io.In(2) <>int_const_3.io.Out
  m0.io.In(3) <>load_4.io.Out(3)
  m0.io.In(4) <>load_4.io.Out(4)
  m0.io.In(5) <>int_const_4.io.Out
  m0.io.In(6) <>load_4.io.Out(5)
  select_12.io.InData2 <> m0.io.Out(6)

  for(i <- 0 until 6)
    m0.io.Out(i).ready := true.B

  int_cmp_5.io.LeftIO <> int_const_2.io.Out

  // int_add_7.io.LeftIO <> int_const_3.io.Out

  // int_add_10.io.LeftIO <> int_const_4.io.Out

  select_12.io.Select <> int_const_5.io.Out

  // int_add_14.io.LeftIO <> int_const_6.io.Out

  // int_cmp_15.io.LeftIO <> int_const_7.io.Out

  int_add_13.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field1")(0)

  address_3.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  // int_add_14.io.RightIO <> loop_0.io.CarryDepenOut.elements("field0")(1)

  load_4.GepAddr <> address_3.io.Out(0)

  int_cmp_5.io.RightIO <> load_4.io.Out(0)

  // int_mul_6.io.LeftIO <> load_4.io.Out(1)

  // int_mul_6.io.RightIO <> load_4.io.Out(2)

  // int_mul_8.io.RightIO <> load_4.io.Out(3)

  // int_mul_9.io.RightIO <> load_4.io.Out(4)

  // int_mul_11.io.RightIO <> load_4.io.Out(5)

  select_12.io.InData1 <> int_cmp_5.io.Out(0)

  // int_add_7.io.RightIO <> int_mul_6.io.Out(0)

  // int_mul_8.io.LeftIO <> int_add_7.io.Out(0)

  // int_mul_9.io.LeftIO <> int_mul_8.io.Out(0)

  // int_add_10.io.RightIO <> int_mul_9.io.Out(0)

  // int_mul_11.io.LeftIO <> int_add_10.io.Out(0)

  // select_12.io.InData2 <> int_mul_11.io.Out(0)

  int_add_13.io.RightIO <> select_12.io.Out(0)

  // int_cmp_15.io.RightIO <> int_add_14.io.Out(0)

  state_branch_16.io.CmpIO <> m1.io.Out(2)

  // mem_ctrl_cache.io.rd.mem(0).MemReq <> load_4.io.MemReq

  // load_4.io.MemResp <> mem_ctrl_cache.io.rd.mem(0).MemResp

  mem_ctrl_cache.io.load_address(0) <> load_4.address_out

  load_4.data_in <> mem_ctrl_cache.io.load_data(0)

  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  // int_const_0.io.enable <> exe_block_0.io.Out(0)

  // int_const_1.io.enable <> exe_block_0.io.Out(1)

  int_const_2.io.enable <> exe_block_0.io.Out(2)

  int_const_3.io.enable <> exe_block_0.io.Out(3)

  int_const_4.io.enable <> exe_block_0.io.Out(4)

  int_const_5.io.enable <> exe_block_0.io.Out(5)

  int_const_6.io.enable <> exe_block_0.io.Out(6)

  int_const_7.io.enable <> exe_block_0.io.Out(7)

  // merge_1.io.enable <> exe_block_0.io.Out(8)

  // merge_2.io.enable <> exe_block_0.io.Out(9)

  address_3.io.enable <> exe_block_0.io.Out(10)

  load_4.io.enable <> exe_block_0.io.Out(11)

  int_cmp_5.io.enable <> exe_block_0.io.Out(12)

  // int_mul_6.io.enable <> exe_block_0.io.Out(13)

  // int_add_7.io.enable <> exe_block_0.io.Out(14)

  // int_mul_8.io.enable <> exe_block_0.io.Out(15)

  // int_mul_9.io.enable <> exe_block_0.io.Out(16)

  // int_add_10.io.enable <> exe_block_0.io.Out(17)

  // int_mul_11.io.enable <> exe_block_0.io.Out(18)
 
  m0.io.enable <> exe_block_0.io.Out(8)

  select_12.io.enable <> exe_block_0.io.Out(13)

  int_add_13.io.enable <> exe_block_0.io.Out(1)

  // int_add_14.io.enable <> exe_block_0.io.Out(0)

  // int_cmp_15.io.enable <> exe_block_0.io.Out(8)
  m1.io.enable <> exe_block_0.io.Out(0)

  state_branch_16.io.enable <> exe_block_0.io.Out(9)

  io.out <> return_17.io.Out

}



// import java.io.{File, FileWriter}

// object getTanhDoubleTop extends App {
//   implicit val p = new WithAccelConfig ++ new WithTestConfig
//   val verilogString = getVerilogString(new getTanhDouble())
//   val filePath = "RTL/getTanhDouble.v"
//   val writer = new PrintWriter(filePath)
//   try { 
//       writer.write(verilogString)
//   } finally {
//     writer.close()
//   }
// }