
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

abstract class sumi3_memDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32))))
	  val out = Decoupled(new Call(List(32)))
	})
}

class sumi3_memDF(implicit p: Parameters) extends sumi3_memDFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1)))
  FineGrainedArgCall.io.In <> io.in

  //Cache
  val mem_ctrl_cache = Module(new MemoryEngine(Size=200, ID = 0, NumRead = 1, NumWrite = 0))
  mem_ctrl_cache.initMem("dataset/sumi3_mem/sumi3_mem.txt")

  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c1 = arith.constant 1 : index
  val int_const_0 = Module(new ConstFastNode(value = 1, ID = 0))

  //%c200 = arith.constant 200 : index
  val int_const_1 = Module(new ConstFastNode(value = 200, ID = 1))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 8, NumPhi = 0, BID = 0))



  /* ================================================================== *
   *                   Printing Operation nodes. 10                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode(ID = 0))

  //%4 = dataflow.addr %arg0[%arg1] {memShape = [200]} : memref<200xi32>[index] -> i32
  val address_1 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 1)(ElementSize = 1, ArraySize = List()))

  //%5 = dataflow.load %4 : i32 -> i32
  val load_2 = Module(new Load(NumOuts = 3, ID = 2, RouteID = 0))

  val m0 = Module(new Chain(NumOps = 3, ID = 0, OpCodes = Array("Mul","Mul","Add"))(sign = false)(p))

  //%9 = arith.addi %arg1, %c1 {Exe = "Loop"} : index
  val int_add_6 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 6, opCode = "Add")(sign = false, Debug = false))

  //%10 = arith.cmpi eq, %9, %c200 {Exe = "Loop"} : index
  val int_cmp_7 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 7, opCode = "eq")(sign = false, Debug = false))

  //dataflow.state %10, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_8 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 8))

  //func.return %0 : i32
  val return_9 = Module(new RetNode2(retTypes = List(32), ID = 9))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode(NumIns = List(1), NumOuts = List(1), NumCarry = List(2, 1), NumExits = 1, ID = 0))



  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  loop_0.io.loopExit(0) <> return_9.io.In.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back



  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> state_branch_0.io.Out(0)

  loop_0.io.loopBack(0) <> state_branch_8.io.FalseOutput(0)

  loop_0.io.loopFinish(0) <> state_branch_8.io.TrueOutput(0)



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

  address_1.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)



  /* ================================================================== *
   *                   Output Data dependencies.                        *
   * ================================================================== */

  loop_0.io.InLiveOut(0) <> m0.io.Out(3)



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */

  return_9.io.In.data("field0") <> loop_0.io.OutLiveOut.elements("field0")(0)



  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  loop_0.io.CarryDepenIn(0) <> int_add_6.io.Out(0)

  address_1.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  int_add_6.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(1)

  loop_0.io.CarryDepenIn(1) <> m0.io.Out(3)

  m0.io.In(3) <> loop_0.io.CarryDepenOut.elements("field1")(0)



  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  int_add_6.io.RightIO <> int_const_0.io.Out

  int_cmp_7.io.RightIO <> int_const_1.io.Out

  load_2.GepAddr <> address_1.io.Out(0)

  m0.io.In(0) <> load_2.io.Out(0)

  m0.io.In(1) <> load_2.io.Out(1)

  m0.io.In(2) <> load_2.io.Out(2)
  
  for(i <- 0 until 3)
    m0.io.Out(i).ready := true.B

  int_cmp_7.io.LeftIO <> int_add_6.io.Out(1)

  state_branch_8.io.CmpIO <> int_cmp_7.io.Out(0)

  mem_ctrl_cache.io.load_address(0) <> load_2.address_out

  load_2.data_in <> mem_ctrl_cache.io.load_data(0)



  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  int_const_0.io.enable <> exe_block_0.io.Out(0)

  int_const_1.io.enable <> exe_block_0.io.Out(1)

  address_1.io.enable <> exe_block_0.io.Out(2)

  load_2.io.enable <> exe_block_0.io.Out(3)

  int_add_6.io.enable <> exe_block_0.io.Out(4)

  int_cmp_7.io.enable <> exe_block_0.io.Out(5)

  state_branch_8.io.enable <> exe_block_0.io.Out(6)

  m0.io.enable <> exe_block_0.io.Out(7)

  io.out <> return_9.io.Out

}

