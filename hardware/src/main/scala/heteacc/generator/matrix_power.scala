
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

// abstract class matrix_powerDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
// 	val io = IO(new Bundle {
// 	  val in = Flipped(Decoupled(new Call(List( 32, 32, 32, 32))))
// 	  val MemResp = Flipped(Valid(new MemResp))
// 	  val MemReq = Decoupled(new MemReq)
// 	  val out = Decoupled(new Call(List()))
// 	})
// }

// class matrix_powerDF(implicit p: Parameters) extends matrix_powerDFIO()(p){

//   val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1, 1, 1 )))
//   FineGrainedArgCall.io.In <> io.in

//   //Cache
//   // val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 5, NumWrite = 1))
//   val mem_ctrl_cache = Module(new TypeStackFile(ID = 0, Size = 32, NReads = 5, NWrites = 1)
//   (WControl = new WriteMemController(NumOps = 1, BaseSize = 2, NumEntries = 2))
//   (RControl = new ReadMemController(NumOps = 5, BaseSize = 2, NumEntries = 2)) )

//   // io.MemReq <> mem_ctrl_cache.io.cache.MemReq
//   // mem_ctrl_cache.io.cache.MemResp <> io.MemResp



//   /* ================================================================== *
//    *                   Printing Const nodes.                            *
//    * ================================================================== */

//   //%c1 = arith.constant 1 : index
//   val int_const_0 = Module(new ConstFastNode(value = 1, ID = 0))

//   //%c-1_i32 = arith.constant -1 : i32
//   val int_const_1 = Module(new ConstFastNode(value = -1, ID = 1))

//   //%c0 = arith.constant 0 : index
//   val int_const_2 = Module(new ConstFastNode(value = 0, ID = 2))

//   //%c1 = arith.constant 1 : index
//   val int_const_3 = Module(new ConstFastNode(value = 1, ID = 3))

//   //%c20 = arith.constant 20 : index
//   val int_const_4 = Module(new ConstFastNode(value = 20, ID = 4))

//   //%c1 = arith.constant 1 : index
//   val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

//   //%c20 = arith.constant 20 : index
//   val int_const_6 = Module(new ConstFastNode(value = 20, ID = 6))



//   /* ================================================================== *
//    *                   Printing Execution Block nodes.                  *
//    * ================================================================== */

//   val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 23, NumPhi = 1, BID = 0))

//   val exe_block_1 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 11, NumPhi = 1, BID = 1))



//   /* ================================================================== *
//    *                   Printing Operation nodes. 29                     *
//    * ================================================================== */

//   //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
//   val state_branch_0 = Module(new UBranchNode(ID = 0))

//   //%0 = dataflow.merge %c1 or %arg4 {Select = "Loop_Signal"} : index
//   val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 3, ID = 1, Res = false))

//   //%1 = arith.index_cast %0 : index to i32
//   val cast_2 = Module(new BitCastNode(NumOuts = 1, ID = 2))

//   //%2 = arith.addi %1, %c-1_i32 : i32
//   val int_add_3 = Module(new ComputeNode(NumOuts = 1, ID = 3, opCode = "Add")(sign = false, Debug = false))

//   //%3 = arith.index_cast %2 : i32 to index
//   val cast_4 = Module(new BitCastNode(NumOuts = 1, ID = 4))

//   //%6 = dataflow.merge %c0 or %arg5 {Select = "Loop_Signal"} : index
//   val merge_5 = Module(new MergeNode(NumInputs = 2, NumOutputs = 4, ID = 5, Res = false))

//   //%7 = dataflow.addr %arg0[%6] {memShape = [20]} : memref<20xi32>[index] -> i32
//   val address_6 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 6)(ElementSize = 32, ArraySize = List()))

//   //%8 = dataflow.load %7 : i32 -> i32
//   val load_7 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 7, RouteID = 0))

//   //%9 = arith.index_cast %8 : i32 to index
//   val cast_8 = Module(new BitCastNode(NumOuts = 2, ID = 8))

//   //%10 = dataflow.addr %arg2[%6] {memShape = [20]} : memref<20xi32>[index] -> i32
//   val address_9 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 9)(ElementSize = 32, ArraySize = List()))

//   //%11 = dataflow.load %10 : i32 -> i32
//   val load_10 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 10, RouteID = 1))

//   //%12 = dataflow.addr %arg1[%6] {memShape = [20]} : memref<20xi32>[index] -> i32
//   val address_11 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 11)(ElementSize = 32, ArraySize = List()))

//   //%13 = dataflow.load %12 : i32 -> i32
//   val load_12 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 12, RouteID = 2))

//   //%14 = arith.index_cast %13 : i32 to index
//   val cast_13 = Module(new BitCastNode(NumOuts = 1, ID = 13))

//   //%15 = dataflow.addr %arg3[%3, %14] {memShape = [20, 20]} : memref<20x20xi32>[index, index] -> i32
//   val address_14 = Module(new GepNode(NumIns = 2, NumOuts = 1, ID = 14)(ElementSize = 32, ArraySize = List()))

//   //%16 = dataflow.load %15 : i32 -> i32
//   val load_15 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 15, RouteID = 3))

//   //%17 = arith.muli %11, %16 : i32
//   val int_mul_16 = Module(new ComputeNode(NumOuts = 1, ID = 16, opCode = "Mul")(sign = false, Debug = false))

//   //%18 = dataflow.addr %arg3[%0, %9] {memShape = [20, 20]} : memref<20x20xi32>[index, index] -> i32
//   val address_17 = Module(new GepNode(NumIns = 2, NumOuts = 1, ID = 17)(ElementSize = 32, ArraySize = List()))

//   //%19 = dataflow.load %18 : i32 -> i32
//   val load_18 = Module(new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 18, RouteID = 4))

//   //%20 = arith.addi %19, %17 : i32
//   val int_add_19 = Module(new ComputeNode(NumOuts = 1, ID = 19, opCode = "Add")(sign = false, Debug = false))

//   //%21 = dataflow.addr %arg3[%0, %9] {memShape = [20, 20]} : memref<20x20xi32>[index, index] -> i32
//   val address_20 = Module(new GepNode(NumIns = 2, NumOuts = 1, ID = 20)(ElementSize = 32, ArraySize = List()))

//   //dataflow.store %20 %21 : i32 i32
//   val store_21 = Module(new UnTypStoreCache(NumPredOps = 0, NumSuccOps = 0, ID = 21, RouteID = 5))

//   //%22 = arith.addi %6, %c1 {Exe = "Loop"} : index
//   val int_add_22 = Module(new ComputeNode(NumOuts = 2, ID = 22, opCode = "Add")(sign = false, Debug = false))

//   //%23 = arith.cmpi eq, %22, %c20 {Exe = "Loop"} : index
//   val int_cmp_23 = Module(new ComputeNode(NumOuts = 1, ID = 23, opCode = "eq")(sign = false, Debug = false))

//   //dataflow.state %23, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//   val state_branch_24 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 24))

//   //%4 = arith.addi %0, %c1 {Exe = "Loop"} : index
//   val int_add_25 = Module(new ComputeNode(NumOuts = 2, ID = 25, opCode = "Add")(sign = false, Debug = false))

//   //%5 = arith.cmpi eq, %4, %c20 {Exe = "Loop"} : index
//   val int_cmp_26 = Module(new ComputeNode(NumOuts = 1, ID = 26, opCode = "eq")(sign = false, Debug = false))

//   //dataflow.state %5, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//   val state_branch_27 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 27))

//   //func.return
//   val return_28 = Module(new RetNode2(retTypes = List(), ID = 28))



//   /* ================================================================== *
//    *                   Printing Loop nodes.                             *
//    * ================================================================== */

//   val loop_0 = Module(new LoopBlockNode(NumIns = List(1, 1, 1, 3, 1, 2), NumOuts = List(), NumCarry = List(1), NumExits = 1, ID = 0))

//   val loop_1 = Module(new LoopBlockNode(NumIns = List(), NumOuts = List(), NumCarry = List(1), NumExits = 1, ID = 1))

// loop_1.io.loopExit(0) <> return_28.io.In.enable
// loop_0.io.loopExit(0) <> state_branch_27.io.enable//DontCare
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

//   loop_0.io.loopBack(0) <> state_branch_24.io.FalseOutput(0)

//   loop_0.io.loopFinish(0) <> state_branch_24.io.TrueOutput(0)

//   loop_1.io.enable <> state_branch_0.io.Out(0)

//   loop_1.io.loopBack(0) <> state_branch_27.io.FalseOutput(0)

//   loop_1.io.loopFinish(0) <> state_branch_27.io.TrueOutput(0)

//   store_21.io.Out(0).ready := true.B

//   // state_branch_24.io.PredOp(0) <> store_21.io.SuccOp(0)



//   /* ================================================================== *
//    *                   Loop dependencies.                               *
//    * ================================================================== */



//   /* ================================================================== *
//    *                   Input Data dependencies.                         *
//    * ================================================================== */

//   loop_0.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

//   loop_0.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)

//   loop_0.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)

//   loop_0.io.InLiveIn(3) <> FineGrainedArgCall.io.Out.data.elements("field3")(0)

//   loop_0.io.InLiveIn(4) <> cast_4.io.Out(0)

//   loop_0.io.InLiveIn(5) <> merge_1.io.Out(0)



//   /* ================================================================== *
//    *                   Live-in dependencies.                            *
//    * ================================================================== */

//   address_6.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

//   address_9.io.baseAddress <> loop_0.io.OutLiveIn.elements("field1")(0)

//   address_11.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(0)

//   address_14.io.baseAddress <> loop_0.io.OutLiveIn.elements("field3")(0)

//   address_17.io.baseAddress <> loop_0.io.OutLiveIn.elements("field3")(1)

//   address_20.io.baseAddress <> loop_0.io.OutLiveIn.elements("field3")(2)

//   address_14.io.idx(0) <> loop_0.io.OutLiveIn.elements("field4")(0)

//   address_17.io.idx(0) <> loop_0.io.OutLiveIn.elements("field5")(0)

//   address_20.io.idx(0) <> loop_0.io.OutLiveIn.elements("field5")(1)



//   /* ================================================================== *
//    *                   Output Data dependencies.                        *
//    * ================================================================== */



//   /* ================================================================== *
//    *                   Live-out dependencies.                           *
//    * ================================================================== */



//   /* ================================================================== *
//    *                   Carry dependencies                               *
//    * ================================================================== */

//   loop_0.io.CarryDepenIn(0) <> int_add_22.io.Out(1)

//   merge_5.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

//   loop_1.io.CarryDepenIn(0) <> int_add_25.io.Out(1)

//   merge_1.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)



//   /* ================================================================== *
//    *                   Printing Connection.                             *
//    * ================================================================== */

//   merge_5.io.Mask <> exe_block_0.io.MaskBB(0)

//   merge_1.io.Mask <> exe_block_1.io.MaskBB(0)

//   merge_1.io.InData(0) <> int_const_0.io.Out

//   int_add_3.io.LeftIO <> int_const_1.io.Out

//   merge_5.io.InData(0) <> int_const_2.io.Out

//   int_add_22.io.LeftIO <> int_const_3.io.Out

//   int_cmp_23.io.LeftIO <> int_const_4.io.Out

//   int_add_25.io.LeftIO <> int_const_5.io.Out

//   int_cmp_26.io.LeftIO <> int_const_6.io.Out

//   cast_2.io.Input <> merge_1.io.Out(1)

//   int_add_25.io.RightIO <> merge_1.io.Out(2)

//   int_add_3.io.RightIO <> cast_2.io.Out(0)

//   cast_4.io.Input <> int_add_3.io.Out(0)

//   address_6.io.idx(0) <> merge_5.io.Out(0)

//   address_9.io.idx(0) <> merge_5.io.Out(1)

//   address_11.io.idx(0) <> merge_5.io.Out(2)

//   int_add_22.io.RightIO <> merge_5.io.Out(3)

//   load_7.io.GepAddr <> address_6.io.Out(0)

//   cast_8.io.Input <> load_7.io.Out(0)

//   address_17.io.idx(1) <> cast_8.io.Out(0)

//   address_20.io.idx(1) <> cast_8.io.Out(1)

//   load_10.io.GepAddr <> address_9.io.Out(0)

//   int_mul_16.io.LeftIO <> load_10.io.Out(0)

//   load_12.io.GepAddr <> address_11.io.Out(0)

//   cast_13.io.Input <> load_12.io.Out(0)

//   address_14.io.idx(1) <> cast_13.io.Out(0)

//   load_15.io.GepAddr <> address_14.io.Out(0)

//   int_mul_16.io.RightIO <> load_15.io.Out(0)

//   int_add_19.io.RightIO <> int_mul_16.io.Out(0)

//   load_18.io.GepAddr <> address_17.io.Out(0)

//   int_add_19.io.LeftIO <> load_18.io.Out(0)

//   store_21.io.inData <> int_add_19.io.Out(0)

//   store_21.io.GepAddr <> address_20.io.Out(0)

//   int_cmp_23.io.RightIO <> int_add_22.io.Out(0)

//   state_branch_24.io.CmpIO <> int_cmp_23.io.Out(0)

//   int_cmp_26.io.RightIO <> int_add_25.io.Out(0)

//   state_branch_27.io.CmpIO <> int_cmp_26.io.Out(0)

//   mem_ctrl_cache.io.ReadIn(0) <> load_7.io.MemReq

//   load_7.io.MemResp <> mem_ctrl_cache.io.ReadOut(0)

//   mem_ctrl_cache.io.ReadIn(1) <> load_10.io.MemReq

//   load_10.io.MemResp <> mem_ctrl_cache.io.ReadOut(1)

//   mem_ctrl_cache.io.ReadIn(2) <> load_12.io.MemReq

//   load_12.io.MemResp <> mem_ctrl_cache.io.ReadOut(2)

//   mem_ctrl_cache.io.ReadIn(3) <> load_15.io.MemReq

//   load_15.io.MemResp <> mem_ctrl_cache.io.ReadOut(3)

//   mem_ctrl_cache.io.ReadIn(4) <> load_18.io.MemReq

//   load_18.io.MemResp <> mem_ctrl_cache.io.ReadOut(4)

//   mem_ctrl_cache.io.WriteIn(0) <> store_21.io.MemReq

//   store_21.io.MemResp <> mem_ctrl_cache.io.WriteOut(0)



//   /* ================================================================== *
//    *                   Printing Execution Block Enable.                 *
//    * ================================================================== */

//   int_const_2.io.enable <> exe_block_0.io.Out(0)

//   int_const_3.io.enable <> exe_block_0.io.Out(1)

//   int_const_4.io.enable <> exe_block_0.io.Out(2)

//   merge_5.io.enable <> exe_block_0.io.Out(3)

//   address_6.io.enable <> exe_block_0.io.Out(4)

//   load_7.io.enable <> exe_block_0.io.Out(5)

//   cast_8.io.enable <> exe_block_0.io.Out(6)

//   address_9.io.enable <> exe_block_0.io.Out(7)

//   load_10.io.enable <> exe_block_0.io.Out(8)

//   address_11.io.enable <> exe_block_0.io.Out(9)

//   load_12.io.enable <> exe_block_0.io.Out(10)

//   cast_13.io.enable <> exe_block_0.io.Out(11)

//   address_14.io.enable <> exe_block_0.io.Out(12)

//   load_15.io.enable <> exe_block_0.io.Out(13)

//   int_mul_16.io.enable <> exe_block_0.io.Out(14)

//   address_17.io.enable <> exe_block_0.io.Out(15)

//   load_18.io.enable <> exe_block_0.io.Out(16)

//   int_add_19.io.enable <> exe_block_0.io.Out(17)

//   address_20.io.enable <> exe_block_0.io.Out(18)

//   store_21.io.enable <> exe_block_0.io.Out(19)

//   int_add_22.io.enable <> exe_block_0.io.Out(20)

//   int_cmp_23.io.enable <> exe_block_0.io.Out(21)

//   state_branch_24.io.enable <> exe_block_0.io.Out(22)

//   int_const_0.io.enable <> exe_block_1.io.Out(1)

//   int_const_1.io.enable <> exe_block_1.io.Out(2)

//   int_const_5.io.enable <> exe_block_1.io.Out(3)

//   int_const_6.io.enable <> exe_block_1.io.Out(4)

//   merge_1.io.enable <> exe_block_1.io.Out(5)

//   cast_2.io.enable <> exe_block_1.io.Out(6)

//   int_add_3.io.enable <> exe_block_1.io.Out(7)

//   cast_4.io.enable <> exe_block_1.io.Out(8)

//   int_add_25.io.enable <> exe_block_1.io.Out(9)

//   int_cmp_26.io.enable <> exe_block_1.io.Out(10)

// //   state_branch_27.io.enable <> exe_block_1.io.Out(11)

//   io.out <> return_28.io.Out

// }



// // import java.io.{File, FileWriter}

// // object matrix_powerTop extends App {
// //   implicit val p = new WithAccelConfig ++ new WithTestConfig
// //   val verilogString = getVerilogString(new matrix_power())
// //   val filePath = "RTL/matrix_power.v"
// //   val writer = new PrintWriter(filePath)
// //   try { 
// //       writer.write(verilogString)
// //   } finally {
// //     writer.close()
// //   }
// // }