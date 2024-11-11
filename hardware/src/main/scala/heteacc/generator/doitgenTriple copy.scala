
// //===------------------------------------------------------------*- Scala -*-===//
// //
// // Automatically generated file for High-level Synthesis (HLS).
// //
// //===----------------------------------------------------------------------===//

// package heteacc.generator

// import chipsalliance.rocketchip.config._
// import chisel3._
// import chisel3.util._
// import chisel3.Module._
// import chisel3.testers._
// import chisel3.iotesters._


// import heteacc.config._
// import heteacc.fpu._
// import heteacc.interfaces._
// import heteacc.junctions._
// import heteacc.memory._
// import heteacc.node._
// import heteacc.loop._
// import heteacc.execution._
// import utility._

// abstract class doitgenTripleDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
// 	val io = IO(new Bundle {
// 	  val in = Flipped(Decoupled(new Call(List( 32, 32, 32))))
// 	  val MemResp = Flipped(Valid(new MemResp))
// 	  val MemReq = Decoupled(new MemReq)
// 	  val out = Decoupled(new Call(List()))
// 	})
// }

// class doitgenTripleDF(implicit p: Parameters) extends doitgenTripleDFIO()(p){

//   val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1, 1 )))
//   FineGrainedArgCall.io.In <> io.in

//   //Cache
//   val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 2, NumWrite = 1))

//   io.MemReq <> mem_ctrl_cache.io.cache.MemReq
//   mem_ctrl_cache.io.cache.MemResp <> io.MemResp



//   /* ================================================================== *
//    *                   Printing Const nodes.                            *
//    * ================================================================== */

//   //%c0 = arith.constant 0 : index
//   // val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

//   //%c0_i32 = arith.constant 0 : i32
//   val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

//   //%c0 = arith.constant 0 : index
//   val int_const_2 = Module(new ConstFastNode(value = 0, ID = 2))

//   //%c16 = arith.constant 16 : index
//   val int_const_3 = Module(new ConstFastNode(value = 16, ID = 3))

//   //%c0_i32 = arith.constant 0 : i32
//   val int_const_4 = Module(new ConstFastNode(value = 0, ID = 4))

//   //%c1 = arith.constant 1 : index
//   val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

//   //%c16 = arith.constant 16 : index
//   val int_const_6 = Module(new ConstFastNode(value = 16, ID = 6))

//   //%c1 = arith.constant 1 : index
//   val int_const_7 = Module(new ConstFastNode(value = 1, ID = 7))

//   //%c16 = arith.constant 16 : index
//   val int_const_8 = Module(new ConstFastNode(value = 16, ID = 8))



//   /* ================================================================== *
//    *                   Printing Execution Block nodes.                  *
//    * ================================================================== */

//   val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 23, NumPhi = 2, BID = 0))

//   val exe_block_1 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 11, NumPhi = 0, BID = 1))



//   /* ================================================================== *
//    *                   Printing Operation nodes. 25                     *
//    * ================================================================== */

//   //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
//   val state_branch_0 = Module(new UBranchNode(ID = 0))

//   //%0 = dataflow.merge %c0 or %arg3 {Select = "Loop_Signal"} : index
//   // val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 3, ID = 1, Res = false))

//   //%6 = dataflow.merge %c0_i32 or %arg5 {Select = "Loop_Signal"} : i32
//   val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 2, Res = false))

//   //%7 = dataflow.merge %c0 or %arg4 {Select = "Loop_Signal"} : index
//   val merge_3 = Module(new MergeNode(NumInputs = 2, NumOutputs = 3, ID = 3, Res = false))

//   //%8 = dataflow.addr %arg0[%7] {memShape = [16]} : memref<16xi32>[index] -> i32
//   val address_4 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 4)(ElementSize = 32, ArraySize = List()))

//   //%9 = dataflow.load %8 : i32 -> i32
//   val load_5 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 3, ID = 5, RouteID = 0))

//   //%10 = arith.muli %0, %c16 : index
//   val int_mul_6 = Module(new ComputeNode(NumOuts = 1, ID = 6, opCode = "Mul")(sign = false, Debug = false))

//   //%11 = arith.addi %7, %10 : index
//   val int_add_7 = Module(new ComputeNode(NumOuts = 1, ID = 7, opCode = "Add")(sign = false, Debug = false))

//   //%12 = dataflow.addr %arg2[%11] {memShape = [256]} : memref<256xi32>[index] -> i32
//   val address_8 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 8)(ElementSize = 32, ArraySize = List()))

//   //%13 = dataflow.load %12 : i32 -> i32
//   val load_9 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 2, ID = 9, RouteID = 1))

//   //%14 = arith.cmpi sgt, %9, %c0_i32 : i32
//   val int_cmp_10 = Module(new ComputeNode(NumOuts = 1, ID = 10, opCode = "sgt")(sign = false, Debug = false))

//   //%15 = arith.muli %9, %13 : i32
//   val int_mul_11 = Module(new ComputeNode(NumOuts = 1, ID = 11, opCode = "Mul")(sign = false, Debug = false))

//   //%16 = arith.addi %15, %13 : i32
//   val int_add_12 = Module(new ComputeNode(NumOuts = 1, ID = 12, opCode = "Add")(sign = false, Debug = false))

//   //%17 = arith.muli %16, %9 : i32
//   val int_mul_13 = Module(new ComputeNode(NumOuts = 1, ID = 13, opCode = "Mul")(sign = false, Debug = false))

//   //%18 = arith.addi %6, %17 : i32
//   val int_add_14 = Module(new ComputeNode(NumOuts = 1, ID = 14, opCode = "Add")(sign = false, Debug = false))

//   //%19 = dataflow.select %14, %18, %6 : i32
//   val select_15 = Module(new SelectNode(NumOuts = 2, ID = 15))

//   //%20 = arith.addi %7, %c1 {Exe = "Loop"} : index
//   val int_add_16 = Module(new ComputeNode(NumOuts = 2, ID = 16, opCode = "Add")(sign = false, Debug = false))

//   //%21 = arith.cmpi eq, %20, %c16 {Exe = "Loop"} : index
//   val int_cmp_17 = Module(new ComputeNode(NumOuts = 1, ID = 17, opCode = "eq")(sign = false, Debug = false))

//   //dataflow.state %21, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//   val state_branch_18 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 18))

//   //%2 = dataflow.addr %arg1[%0] {memShape = [16]} : memref<16xi32>[index] -> i32
//   val address_19 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 19)(ElementSize = 32, ArraySize = List()))

//   //dataflow.store %1 %2 : i32 i32
//   val store_20 = Module(new UnTypStoreCache(NumPredOps = 0, NumSuccOps = 1, ID = 20, RouteID = 2))

//   //%3 = arith.addi %0, %c1 {Exe = "Loop"} : index
//   val int_add_21 = Module(new ComputeNode(NumOuts = 2, ID = 21, opCode = "Add")(sign = false, Debug = false))

//   //%4 = arith.cmpi eq, %3, %c16 {Exe = "Loop"} : index
//   val int_cmp_22 = Module(new ComputeNode(NumOuts = 1, ID = 22, opCode = "eq")(sign = false, Debug = false))

//   //dataflow.state %4, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//   val state_branch_23 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 1, ID = 23))

//   //func.return
//   val return_24 = Module(new RetNode2(retTypes = List(), ID = 24))



//   /* ================================================================== *
//    *                   Printing Loop nodes.                             *
//    * ================================================================== */

//   val loop_0 = Module(new LoopBlockNode(NumIns = List(1, 1, 1), NumOuts = List(1), NumCarry = List(1, 1), NumExits = 1, ID = 0))

//   val loop_1 = Module(new LoopBlockNode(NumIns = List(1, 1, 1), NumOuts = List(), NumCarry = List(3), NumExits = 1, ID = 1))

// //   loop_0.io.loopExit(0) <>  DontCare
//   loop_1.io.loopExit(0) <>  return_24.io.In.enable

//   /* ================================================================== *
//    *                   Control Signal.                                  *
//    * ================================================================== */

//   FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

//   exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

//   exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back

//   exe_block_1.io.predicateIn(0) <> loop_1.io.activate_loop_start

//   exe_block_1.io.predicateIn(1) <> loop_1.io.activate_loop_back



//   /* ================================================================== *
//    *                   Loop Control Signal.                             *
//    * ================================================================== */
//   loop_0.io.enable <> exe_block_1.io.Out(0)

//   loop_0.io.loopBack(0) <> state_branch_18.io.FalseOutput(0)

//   loop_0.io.loopFinish(0) <> state_branch_18.io.TrueOutput(0)

//   loop_1.io.enable <> state_branch_0.io.Out(0)

//   loop_1.io.loopBack(0) <> state_branch_23.io.FalseOutput(0)

//   loop_1.io.loopFinish(0) <> state_branch_23.io.TrueOutput(0)

//   store_20.io.Out(0).ready := true.B

//   state_branch_23.io.PredOp(0) <> store_20.io.SuccOp(0)



//   /* ================================================================== *
//    *                   Loop dependencies.                               *
//    * ================================================================== */



//   /* ================================================================== *
//    *                   Input Data dependencies.                         *
//    * ================================================================== */

// //   loop_0.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

// //   loop_0.io.InLiveIn(1) <> merge_1.io.Out(0)

// //   loop_0.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)

// //   loop_1.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)

// //   loop_1.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)



//   loop_0.io.InLiveIn(0) <> loop_1.io.OutLiveIn.elements("field1")(0)//FineGrainedArgCall.io.Out.data.elements("field0")(0)

//   // loop_0.io.InLiveIn(1) <> merge_1.io.Out(0)
//   loop_0.io.InLiveIn(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)

//   loop_0.io.InLiveIn(2) <> loop_1.io.OutLiveIn.elements("field2")(0)//FineGrainedArgCall.io.Out.data.elements("field2")(0)



  
//   loop_1.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)

//   loop_1.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

//   loop_1.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)


//   /* ================================================================== *
//    *                   Live-in dependencies.                            *
//    * ================================================================== */

//   address_4.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

//   int_mul_6.io.RightIO <> loop_0.io.OutLiveIn.elements("field1")(0)

//   address_8.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(0)
    
//   address_19.io.baseAddress <> loop_1.io.OutLiveIn.elements("field0")(0)

  

//   /* ================================================================== *
//    *                   Output Data dependencies.                        *
//    * ================================================================== */

//   loop_0.io.InLiveOut(0) <> select_15.io.Out(0)



//   /* ================================================================== *
//    *                   Live-out dependencies.                           *
//    * ================================================================== */


//  loop_0.io.OutLiveOut.elements("field0")(0) <> store_20.io.inData

//  store_20.io.GepAddr <> address_19.io.Out(0)
//   /* ================================================================== *
//    *                   Carry dependencies                               *
//    * ================================================================== */

//   loop_0.io.CarryDepenIn(0) <> int_add_16.io.Out(1) //loop_0.io.CarryDepenOut.elements("field0")(0)

//   merge_3.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

//   loop_0.io.CarryDepenIn(1) <> select_15.io.Out(1)//loop_0.io.CarryDepenOut.elements("field1")(0)

//   merge_2.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(0)

//   loop_1.io.CarryDepenIn(0) <> int_add_21.io.Out(1)//loop_1.io.CarryDepenOut.elements("field0")(0)

//   // merge_1.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)



//   /* ================================================================== *
//    *                   Printing Connection.                             *
//    * ================================================================== */

//   merge_2.io.Mask <> exe_block_0.io.MaskBB(0)

//   merge_3.io.Mask <> exe_block_0.io.MaskBB(1)

//   // merge_1.io.Mask <> exe_block_1.io.MaskBB(0)

//   // merge_1.io.InData(0) <> int_const_0.io.Out

//   merge_2.io.InData(0) <> int_const_1.io.Out

//   merge_3.io.InData(0) <> int_const_2.io.Out

//   int_mul_6.io.LeftIO <> int_const_3.io.Out

//   int_cmp_10.io.LeftIO <> load_5.io.Out(0)

//   int_add_16.io.LeftIO <> int_const_5.io.Out

//   int_cmp_17.io.LeftIO <> int_const_6.io.Out

//   int_add_21.io.LeftIO <> int_const_7.io.Out

//   int_cmp_22.io.LeftIO <> int_const_8.io.Out

//   address_19.io.idx(0) <> loop_1.io.CarryDepenOut.elements("field0")(1)//merge_1.io.Out(1)

//   int_add_21.io.RightIO <> loop_1.io.CarryDepenOut.elements("field0")(2)//merge_1.io.Out(2)

//   int_add_14.io.LeftIO <> merge_2.io.Out(0)

//   select_15.io.InData2 <> merge_2.io.Out(1)

//   address_4.io.idx(0) <> merge_3.io.Out(0)

//   int_add_7.io.LeftIO <> merge_3.io.Out(1)

//   int_add_16.io.RightIO <> merge_3.io.Out(2)

//   load_5.io.GepAddr <> address_4.io.Out(0)

//   int_cmp_10.io.RightIO <> int_const_4.io.Out

//   int_mul_11.io.LeftIO <> load_5.io.Out(1)

//   int_mul_13.io.RightIO <> load_5.io.Out(2)

//   int_add_7.io.RightIO <> int_mul_6.io.Out(0)

//   address_8.io.idx(0) <> int_add_7.io.Out(0)

//   load_9.io.GepAddr <> address_8.io.Out(0)

//   int_mul_11.io.RightIO <> load_9.io.Out(0)

//   int_add_12.io.RightIO <> load_9.io.Out(1)

//   select_15.io.Select <> int_cmp_10.io.Out(0)

//   int_add_12.io.LeftIO <> int_mul_11.io.Out(0)

//   int_mul_13.io.LeftIO <> int_add_12.io.Out(0)

//   int_add_14.io.RightIO <> int_mul_13.io.Out(0)

//   select_15.io.InData1 <> int_add_14.io.Out(0)

//   int_cmp_17.io.RightIO <> int_add_16.io.Out(0)

//   state_branch_18.io.CmpIO <> int_cmp_17.io.Out(0)

//   int_cmp_22.io.RightIO <> int_add_21.io.Out(0)

//   state_branch_23.io.CmpIO <> int_cmp_22.io.Out(0)

//   mem_ctrl_cache.io.rd.mem(0).MemReq <> load_5.io.MemReq

//   load_5.io.MemResp <> mem_ctrl_cache.io.rd.mem(0).MemResp

//   mem_ctrl_cache.io.rd.mem(1).MemReq <> load_9.io.MemReq

//   load_9.io.MemResp <> mem_ctrl_cache.io.rd.mem(1).MemResp

//   mem_ctrl_cache.io.wr.mem(0).MemReq <> store_20.io.MemReq

//   store_20.io.MemResp <> mem_ctrl_cache.io.wr.mem(0).MemResp



//   /* ================================================================== *
//    *                   Printing Execution Block Enable.                 *
//    * ================================================================== */

//   int_const_1.io.enable <> exe_block_0.io.Out(0)

//   int_const_2.io.enable <> exe_block_0.io.Out(1)

//   int_const_3.io.enable <> exe_block_0.io.Out(2)

//   int_const_4.io.enable <> exe_block_0.io.Out(3)

//   int_const_5.io.enable <> exe_block_0.io.Out(4)

//   int_const_6.io.enable <> exe_block_0.io.Out(5)

//   merge_2.io.enable <> exe_block_0.io.Out(6)

//   merge_3.io.enable <> exe_block_0.io.Out(7)

//   address_4.io.enable <> exe_block_0.io.Out(8)

//   load_5.io.enable <> exe_block_0.io.Out(9)

//   int_mul_6.io.enable <> exe_block_0.io.Out(10)

//   int_add_7.io.enable <> exe_block_0.io.Out(11)

//   address_8.io.enable <> exe_block_0.io.Out(12)

//   load_9.io.enable <> exe_block_0.io.Out(13)

//   int_cmp_10.io.enable <> exe_block_0.io.Out(14)

//   int_mul_11.io.enable <> exe_block_0.io.Out(15)

//   int_add_12.io.enable <> exe_block_0.io.Out(16)

//   int_mul_13.io.enable <> exe_block_0.io.Out(17)

//   int_add_14.io.enable <> exe_block_0.io.Out(18)

//   select_15.io.enable <> exe_block_0.io.Out(19)

//   int_add_16.io.enable <> exe_block_0.io.Out(20)

//   int_cmp_17.io.enable <> exe_block_0.io.Out(21)

//   state_branch_18.io.enable <> exe_block_0.io.Out(22)

//   // int_const_0.io.enable <> exe_block_1.io.Out(1)

//   int_const_7.io.enable <> exe_block_1.io.Out(2)

//   int_const_8.io.enable <> exe_block_1.io.Out(3)

//   // merge_1.io.enable <> exe_block_1.io.Out(4)

//   address_19.io.enable <> exe_block_1.io.Out(5)

//   store_20.io.enable <> exe_block_1.io.Out(6)

//   int_add_21.io.enable <> exe_block_1.io.Out(7)

//   int_cmp_22.io.enable <> exe_block_1.io.Out(1)



//   address_19.io.enable <> exe_block_1.io.Out(8)

//   store_20.io.enable <> exe_block_1.io.Out(9)

//   int_add_21.io.enable <> exe_block_1.io.Out(10)

//   int_cmp_22.io.enable <> exe_block_1.io.Out(4)


//   state_branch_23.io.enable <> loop_0.io.loopExit(0)

//   io.out <> return_24.io.Out

// }

