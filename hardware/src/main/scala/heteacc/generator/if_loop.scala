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


abstract class if_loop_1DFIO(implicit val p: Parameters) extends Module with HasAccelParams {
	val io = IO(new Bundle {
	  val in = Flipped(Decoupled(new Call(List( 32))))
	  // val MemResp = Flipped(Valid(new MemResp))
	  // val MemReq = Decoupled(new MemReq)
	  val out = Decoupled(new Call(List(32)))
	})
}

class if_loop_1DF(implicit p: Parameters) extends if_loop_1DFIO()(p){

  val FineGrainedArgCall = Module(new SplitCallDCR(List(1 )))
  FineGrainedArgCall.io.In <> io.in


  val mem_ctrl_cache = Module(new MemoryEngine(Size=100, ID = 0, NumRead = 1, NumWrite = 0))
  // val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 1, NumWrite = 0))
  // io.MemReq <> mem_ctrl_cache.io.cache.MemReq
  // mem_ctrl_cache.io.cache.MemResp <> io.MemResp
  mem_ctrl_cache.initMem("dataset/memory/if_loop.txt")
  // val mem_ctrl_cache = Module(new TypeStackFile(ID = 0, Size = 32, NReads = 1) 
  // (RControl = new ReadMemController(NumOps = 1, BaseSize = 2, NumEntries = 2)) )

//   io.MemReq <> mem_ctrl_cache.io.MemReq
//   mem_ctrl_cache.io.MemResp <> io.MemResp
// // io.MemRep <> DontCare
  /* ================================================================== *
   *                   Printing Const nodes.                            *
   * ================================================================== */

  //%c0_i32 = arith.constant 0 : i32
  // val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

  //%c0 = arith.constant 0 : index
  // val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

  //%c2_i32 = arith.constant 2 : i32
  val int_const_2 = Module(new ConstFastNode(value = 2, ID = 2))

  //%c1_i32 = arith.constant 1 : i32
  val int_const_3 = Module(new ConstFastNode(value = 10, ID = 3))

  //%c1 = arith.constant 1 : index
  val int_const_4 = Module(new ConstFastNode(value = 1, ID = 4))

  //%c100 = arith.constant 100 : index
  val int_const_5 = Module(new ConstFastNode(value = 100, ID = 5))



  /* ================================================================== *
   *                   Printing Execution Block nodes.                  *
   * ================================================================== */

  val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 13, NumPhi = 0, BID = 0))


  //%1 = arith.index_cast %0 : index to i32
  // val cast_2 = Module(new BitCastNode(NumOuts = 1, ID = 2))
  /* ================================================================== *
   *                   Printing Operation nodes. 13                     *
   * ================================================================== */

  //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
  val state_branch_0 = Module(new UBranchNode( ID = 0))

  //%4 = dataflow.merge %c0_i32 or %arg2 {Select = "Loop_Signal"} : i32
  // val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 1, Res = false))

  //%5 = dataflow.merge %c0 or %arg1 {Select = "Loop_Signal"} : index
  // val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 2, Res = false))

  //%6 = dataflow.addr %arg0[%5] {memShape = [100]} : memref<100xi32>[index] -> i32
  val address_3 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 3)(ElementSize = 1, ArraySize = List()))

  //%7 = dataflow.load %6 : i32 -> i32
  // val load_4 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 4, RouteID = 0))
  val load_4 = Module(new Load(NumOuts = 1, ID = 4, RouteID = 0))

  // val m0 = Module(new Chain(NumOps = 2, ID = 0, OpCodes = Array("Mul","ugt"))(sign = false)(p))
  // m0.io.Out(1).ready := true.B
  //%8 = arith.muli %7, %c2_i32 : i32
  val int_mul_5 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 5, opCode = "Mul")(sign = false, Debug = false))

  //%9 = arith.cmpi ugt, %8, %c1_i32 : i32
  val int_cmp_6 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 6, opCode = "ult")(sign = false, Debug = false))
  
  // val state_branch_0123 = Module(new CBranchNode(ID = 123))

  //%10 = arith.addi %8, %4 : i32
  val int_add_7 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 7, opCode = "Add")(sign = false, Debug = false))

  //%11 = dataflow.select %9, %10, %4 : i32
  val select_8 = Module(new SelectNodeWithoutState(NumOuts = 2, ID = 8))

  //%12 = arith.addi %5, %c1 {Exe = "Loop"} : index
  val int_add_9 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 9, opCode = "Add")(sign = false, Debug = false))

  //%13 = arith.cmpi eq, %12, %c100 {Exe = "Loop"} : index
  val int_cmp_10 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 10, opCode = "eq")(sign = false, Debug = false))
  
  
  //dataflow.state %13, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
  val state_branch_11 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 11))

  //func.return %0 : i32
  val return_12 = Module(new RetNode2(retTypes = List(32), ID = 12))



  /* ================================================================== *
   *                   Printing Loop nodes.                             *
   * ================================================================== */

  val loop_0 = Module(new LoopBlockNode (NumIns = List(1), NumOuts = List(1), NumCarry = List(2,2), NumExits = 1, ID = 0))


 
  /* ================================================================== *
   *                   Control Signal.                                  *
   * ================================================================== */

  FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

  exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

  exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back


  // m0.io.In(0) <>int_const_2.io.Out
  // m0.io.In(1) <>load_4.io.Out(0)
  // m0.io.In(2) <>int_const_3.io.Out

  /* ================================================================== *
   *                   Loop Control Signal.                             *
   * ================================================================== */

  loop_0.io.enable <> state_branch_0.io.Out(0)

  loop_0.io.loopFinish(0)  <> state_branch_11.io.TrueOutput(0)

  loop_0.io.loopBack(0)  <> state_branch_11.io.FalseOutput(0)


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

  loop_0.io.InLiveOut(0) <> select_8.io.Out(0)



  /* ================================================================== *
   *                   Live-out dependencies.                           *
   * ================================================================== */


  loop_0.io.OutLiveOut.elements("field0")(0) <> return_12.io.In.data("field0")


  /* ================================================================== *
   *                   Carry dependencies                               *
   * ================================================================== */

  // loop_0.io.CarryDepenIn(0) <>  select_8.io.Out(1)

  // merge_1.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

  int_add_7.io.RightIO <> loop_0.io.CarryDepenOut.elements("field1")(1)//merge_2.io.Out(1)

  // select_8.io.InData2 <> loop_0.io.CarryDepenOut.elements("field0")(1)

  loop_0.io.CarryDepenIn(0) <>  int_add_9.io.Out(1)

  // merge_2.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(0)

  // merge_2.io.InData(1) <> cast_2.io.Out(0) 
  // merge_2.io.InData(0) <> int_const_0.io.Out
  select_8.io.InData2 <> loop_0.io.CarryDepenOut.elements("field1")(0)//merge_2.io.Out(0)
  select_8.io.Out(1) <> loop_0.io.CarryDepenIn(1)//cast_2.io.Input
  /* ================================================================== *
   *                   Printing Connection.                             *
   * ================================================================== */

  // merge_1.io.Mask <> exe_block_0.io.MaskBB(0)

  // merge_2.io.Mask <> exe_block_0.io.MaskBB(0)

  // merge_1.io.InData(0) <> int_const_0.io.Out

  // merge_2.io.InData(0) <> int_const_1.io.Out

  int_mul_5.io.LeftIO <> int_const_2.io.Out

  int_cmp_6.io.LeftIO <> int_const_3.io.Out

  int_add_9.io.LeftIO <> int_const_4.io.Out

  int_cmp_10.io.LeftIO <> int_const_5.io.Out

  // int_add_7.io.RightIO <> merge_1.io.Out(0)

  // select_8.io.InData2 <> merge_1.io.Out(1)

  address_3.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(1)

  int_add_9.io.RightIO <> loop_0.io.CarryDepenOut.elements("field0")(0)
  
  load_4.GepAddr <> address_3.io.Out(0)
  // load_4.io.GepAddr <> address_3.io.Out(0)
  int_mul_5.io.RightIO <> load_4.io.Out(0)

  int_cmp_6.io.RightIO <> int_mul_5.io.Out(1)

  int_add_7.io.LeftIO <> int_mul_5.io.Out(0)//m0.io.Out(0)

  select_8.io.Select <> int_cmp_6.io.Out(0)//m0.io.Out(2)

  // state_branch_0123.io.CmpIO <> m0.io.Out(2)

  select_8.io.InData1 <> int_add_7.io.Out(0)

  int_cmp_10.io.RightIO <> int_add_9.io.Out(0)

  state_branch_11.io.CmpIO <> int_cmp_10.io.Out(0)

  // mem_ctrl_cache.io.ReadIn(0) <> load_4.io.MemReq

  // load_4.io.MemResp <> mem_ctrl_cache.io.ReadOut(0)


  mem_ctrl_cache.io.load_address(0) <> load_4.address_out

  load_4.data_in <> mem_ctrl_cache.io.load_data(0)
  

  // mem_ctrl_cache.io.rd.mem(0).MemReq <> load_4.io.MemReq

  // load_4.io.MemResp <> mem_ctrl_cache.io.rd.mem(0).MemResp
  /* ================================================================== *
   *                   Printing Execution Block Enable.                 *
   * ================================================================== */

  // int_const_0.io.enable <> exe_block_0.io.Out(12)

  // int_const_1.io.enable <> exe_block_0.io.Out(1)

  int_const_2.io.enable <> exe_block_0.io.Out(2)

  int_const_3.io.enable <> exe_block_0.io.Out(3)

  int_const_4.io.enable <> exe_block_0.io.Out(4)

  int_const_5.io.enable <> exe_block_0.io.Out(5)

  // merge_1.io.enable <> exe_block_0.io.Out(6)

  // merge_2.io.enable <> exe_block_0.io.Out(13)

  address_3.io.enable <> exe_block_0.io.Out(8)

  load_4.io.enable <> exe_block_0.io.Out(9)

  int_mul_5.io.enable <> exe_block_0.io.Out(10)
  // m0.io.enable  <> exe_block_0.io.Out(10)
  int_cmp_6.io.enable <> exe_block_0.io.Out(12)

  int_add_7.io.enable <> exe_block_0.io.Out(11)

  // state_branch_0123.io.Out(0) <> int_add_7.io.enable

  select_8.io.enable <> exe_block_0.io.Out(0)

  int_add_9.io.enable <> exe_block_0.io.Out(6)

  int_cmp_10.io.enable <> exe_block_0.io.Out(1)

  state_branch_11.io.enable <> exe_block_0.io.Out(7)

  // cast_2.io.enable <> exe_block_0.io.Out(14)
  loop_0.io.loopExit(0) <>return_12.io.In.enable

  io.out <> return_12.io.Out

}

