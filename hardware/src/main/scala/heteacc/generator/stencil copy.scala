
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

// abstract class stencilDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
// 	val io = IO(new Bundle {
// 	  val in = Flipped(Decoupled(new Call(List( 32, 32, 32))))
// 	  // val MemResp = Flipped(Valid(new MemResp))
// 	  // val MemReq = Decoupled(new MemReq)
// 	  val out = Decoupled(new Call(List()))
// 	})
// }

// class stencilDF(implicit p: Parameters) extends stencilDFIO()(p){

//   val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1, 1 )))
//   FineGrainedArgCall.io.In <> io.in

//   //Cache
//   // val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 2, NumWrite = 1))

//   // io.MemReq <> mem_ctrl_cache.io.cache.MemReq
//   // mem_ctrl_cache.io.cache.MemResp <> io.MemResp

//   val mem_ctrl_cache = Module(new MemoryEngine(Size=8200, ID = 0, NumRead = 2, NumWrite = 0))
//   mem_ctrl_cache.initMem("dataset/stencil2d/in.txt")

//   val mem_ctrl_cache_store = Module(new MemoryEngine(Size=8192, ID = 0, NumRead = 0, NumWrite = 1))
//   mem_ctrl_cache_store.initMem("dataset/stencid2d/out.txt")

//   /* ================================================================== *
//    *                   Printing Const nodes.                            *
//    * ================================================================== */

//   //%c0 = arith.constant 0 : index
//   // val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

//   //%c0 = arith.constant 0 : index
//   val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

//   //%c0_i32 = arith.constant 0 : i32
//   // val int_const_2 = Module(new ConstFastNode(value = 0, ID = 2))

//   //%c0 = arith.constant 0 : index
//   // val int_const_3 = Module(new ConstFastNode(value = 0, ID = 3))

//   //%c0 = arith.constant 0 : index
//   // val int_const_4 = Module(new ConstFastNode(value = 0, ID = 4))

//   //%c1 = arith.constant 1 : index
//   val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

//   //%c6 = arith.constant 6 : index
//   val int_const_6 = Module(new ConstFastNode(value = 6, ID = 6))

//   //%c1 = arith.constant 1 : index
//   val int_const_7 = Module(new ConstFastNode(value = 1, ID = 7))

//   //%c2 = arith.constant 2 : index
//   val int_const_8 = Module(new ConstFastNode(value = 2, ID = 8))

//   //%c1 = arith.constant 1 : index
//   val int_const_9 = Module(new ConstFastNode(value = 1, ID = 9))

//   //%c2 = arith.constant 2 : index
//   val int_const_10 = Module(new ConstFastNode(value = 2, ID = 10))

//   //%c6 = arith.constant 6 : index
//   val int_const_11 = Module(new ConstFastNode(value = 6, ID = 11))

//   //%c1 = arith.constant 1 : index
//   val int_const_12 = Module(new ConstFastNode(value = 1, ID = 12))

//   //%c61 = arith.constant 61 : index
//   val int_const_13 = Module(new ConstFastNode(value = 61, ID = 13))

//   //%c1 = arith.constant 1 : index
//   val int_const_14 = Module(new ConstFastNode(value = 1, ID = 14))

//   //%c125 = arith.constant 125 : index
//   val int_const_15 = Module(new ConstFastNode(value = 125, ID = 15))



//   /* ================================================================== *
//    *                   Printing Execution Block nodes.                  *
//    * ================================================================== */

//   val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 20, NumPhi = 0, BID = 0))

//   val exe_block_1 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 5, NumPhi = 0, BID = 1))

//   val exe_block_2 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 12, NumPhi = 1, BID = 2))

//   val exe_block_3 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 5, NumPhi = 0, BID = 3))



//   /* ================================================================== *
//    *                   Printing Operation nodes. 37                     *
//    * ================================================================== */

//   //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
//   val state_branch_0 = Module(new UBranchNode(ID = 0))

//   //%0 = dataflow.merge %c0 or %arg3 {Select = "Loop_Signal"} : index
//   // val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 1, Res = false))

//   //%3 = dataflow.merge %c0 or %arg4 {Select = "Loop_Signal"} : index
//   val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 3, ID = 2, Res = false))

//   //%11 = dataflow.merge %c0_i32 or %arg6 {Select = "Loop_Signal"} : i32
//   // val merge_3 = Module(new MergeNode(NumInputs = 2, NumOutputs = 1, ID = 3, Res = false))

//   //%12 = dataflow.merge %c0 or %arg5 {Select = "Loop_Signal"} : index
//   // val merge_4 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 4, Res = false))

//   //%17 = dataflow.merge %11 or %arg8 {Select = "Loop_Signal"} : i32
//   // val merge_5 = Module(new MergeNode(NumInputs = 2, NumOutputs = 1, ID = 5, Res = false))

//   //%18 = dataflow.merge %c0 or %arg7 {Select = "Loop_Signal"} : index
//   // val merge_6 = Module(new MergeNode(NumInputs = 2, NumOutputs = 3, ID = 6, Res = false))

//   //%19 = arith.shli %12, %c1 : index
//   val int_shl_7 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 7, opCode = "shl")(sign = false, Debug = false))

//   //%20 = arith.addi %12, %19 : index
//   val int_add_8 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 8, opCode = "Add")(sign = false, Debug = false))

//   //%21 = arith.addi %20, %18 : index
//   val int_add_9 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 9, opCode = "Add")(sign = false, Debug = false))

//   //%22 = dataflow.addr %arg2[%21] {memShape = [9]} : memref<9xi32>[index] -> i32
//   val address_10 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 10)(ElementSize = 1, ArraySize = List()))

//   //%23 = dataflow.load %22 : i32 -> i32
//   val load_11 = Module(new Load(NumOuts = 1, ID = 11, RouteID = 0))

//   //%24 = arith.addi %0, %12 : index
//   val int_add_12 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 12, opCode = "Add")(sign = false, Debug = false))

//   //%25 = arith.shli %24, %c6 : index
//   val int_shl_13 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 13, opCode = "shl")(sign = false, Debug = false))

//   //%26 = arith.addi %25, %3 : index
//   val int_add_14 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 14, opCode = "Add")(sign = false, Debug = false))

//   //%27 = arith.addi %26, %18 : index
//   val int_add_15 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 15, opCode = "Add")(sign = false, Debug = false))

//   //%28 = dataflow.addr %arg0[%27] {memShape = [8192]} : memref<8192xi32>[index] -> i32
//   val address_16 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 16)(ElementSize = 1, ArraySize = List()))

//   //%29 = dataflow.load %28 : i32 -> i32
//   val load_17 = Module(new Load(NumOuts = 1, ID = 17, RouteID = 1))

//   //%30 = arith.muli %23, %29 : i32
//   val int_mul_18 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 18, opCode = "Mul")(sign = false, Debug = false))

//   //%31 = arith.addi %17, %30 : i32
//   val int_add_19 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 19, opCode = "Add")(sign = false, Debug = false))

//   //%32 = arith.addi %18, %c1 {Exe = "Loop"} : index
//   val int_add_20 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 20, opCode = "Add")(sign = false, Debug = false))

//   //%33 = arith.cmpi eq, %32, %c2 {Exe = "Loop"} : index
//   val int_cmp_21 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 21, opCode = "eq")(sign = false, Debug = false))

//   //dataflow.state %33, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//   val state_branch_22 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 22))

//   //%14 = arith.addi %12, %c1 {Exe = "Loop"} : index
//   val int_add_23 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 23, opCode = "Add")(sign = false, Debug = false))

//   //%15 = arith.cmpi eq, %14, %c2 {Exe = "Loop"} : index
//   val int_cmp_24 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 24, opCode = "eq")(sign = false, Debug = false))

//   //dataflow.state %15, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//   val state_branch_25 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 25))

//   //%5 = arith.shli %0, %c6 : index
//   val int_shl_26 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 26, opCode = "shl")(sign = false, Debug = false))

//   //%6 = arith.addi %5, %3 : index
//   val int_add_27 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 27, opCode = "Add")(sign = false, Debug = false))

//   //%7 = dataflow.addr %arg1[%6] {memShape = [8192]} : memref<8192xi32>[index] -> i32
//   val address_28 = Module(new GepNodeWithoutState(NumIns = 1, NumOuts = 1, ID = 28)(ElementSize = 1, ArraySize = List()))

//   //dataflow.store %4 %7 : i32 i32
//   val store_29 = Module(new Store(NumOuts = 1, ID = 29, RouteID = 2))

//   //%8 = arith.addi %3, %c1 {Exe = "Loop"} : index
//   val int_add_30 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 30, opCode = "Add")(sign = false, Debug = false))

//   //%9 = arith.cmpi eq, %8, %c61 {Exe = "Loop"} : index
//   val int_cmp_31 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 31, opCode = "eq")(sign = false, Debug = false))

//   //dataflow.state %9, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//   val state_branch_32 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 32))

//   //%1 = arith.addi %0, %c1 {Exe = "Loop"} : index
//   val int_add_33 = Module(new ComputeNodeWithoutState(NumOuts = 2, ID = 33, opCode = "Add")(sign = false, Debug = false))

//   //%2 = arith.cmpi eq, %1, %c125 {Exe = "Loop"} : index
//   val int_cmp_34 = Module(new ComputeNodeWithoutState(NumOuts = 1, ID = 34, opCode = "eq")(sign = false, Debug = false))

//   //dataflow.state %2, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//   val state_branch_35 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 35))

//   //func.return
//   val return_36 = Module(new RetNode2(retTypes = List(), ID = 36))



//   /* ================================================================== *
//    *                   Printing Loop nodes.                             *
//    * ================================================================== */

//   val loop_0 = Module(new LoopBlockNode(NumIns = List(3, 1, 1, 1, 1), NumOuts = List(1), NumCarry = List(1, 3), NumExits = 1, ID = 0))

//   val loop_1 = Module(new LoopBlockNode(NumIns = List(1, 1,1,1), NumOuts = List(1), NumCarry = List(2), NumExits = 1, ID = 1))

//   val loop_2 = Module(new LoopBlockNode(NumIns = List(1, 1,1,2), NumOuts = List(), NumCarry = List(1), NumExits = 1, ID = 2))

//   val loop_3 = Module(new LoopBlockNode(NumIns = List(1,1,1), NumOuts = List(), NumCarry = List(2), NumExits = 1, ID = 3))



//   /* ================================================================== *
//    *                   Control Signal.                                  *
//    * ================================================================== */

//   FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable

//   exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

//   exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back

//   exe_block_1.io.predicateIn(0) <> loop_1.io.activate_loop_start

//   exe_block_1.io.predicateIn(1) <> loop_1.io.activate_loop_back

//   exe_block_2.io.predicateIn(0) <> loop_2.io.activate_loop_start

//   exe_block_2.io.predicateIn(1) <> loop_2.io.activate_loop_back

//   exe_block_3.io.predicateIn(0) <> loop_3.io.activate_loop_start

//   exe_block_3.io.predicateIn(1) <> loop_3.io.activate_loop_back



//   /* ================================================================== *
//    *                   Loop Control Signal.                             *
//    * ================================================================== */

//   loop_0.io.enable <> exe_block_1.io.Out(0)

//   loop_0.io.loopBack(0) <> state_branch_22.io.FalseOutput(0)

//   loop_0.io.loopFinish(0) <> state_branch_22.io.TrueOutput(0)

//   loop_1.io.enable <> exe_block_2.io.Out(0)

//   loop_1.io.loopBack(0) <> state_branch_25.io.FalseOutput(0)

//   loop_1.io.loopFinish(0) <> state_branch_25.io.TrueOutput(0)

//   loop_2.io.enable <> exe_block_3.io.Out(0)

//   loop_2.io.loopBack(0) <> state_branch_32.io.FalseOutput(0)

//   loop_2.io.loopFinish(0) <> state_branch_32.io.TrueOutput(0)

//   loop_3.io.enable <> state_branch_0.io.Out(0)

//   loop_3.io.loopBack(0) <> state_branch_35.io.FalseOutput(0)

//   loop_3.io.loopFinish(0) <> state_branch_35.io.TrueOutput(0)

//   store_29.io.Out(0).ready := true.B





//   /* ================================================================== *
//    *                   Loop dependencies.                               *
//    * ================================================================== */

//   loop_0.io.loopExit(0) <> state_branch_25.io.enable
//   loop_1.io.loopExit(0) <> state_branch_32.io.enable
//   loop_2.io.loopExit(0) <> state_branch_35.io.enable
//   loop_3.io.loopExit(0) <> return_36.io.In.enable
//   /* ================================================================== *
//    *                   Input Data dependencies.                         *
//    * ================================================================== */

//   loop_0.io.InLiveIn(0) <> loop_1.io.CarryDepenOut.elements("field0")(0)//merge_4.io.Out(0)

//   loop_0.io.InLiveIn(1) <> loop_1.io.OutLiveIn.elements("field1")(0)

//   loop_0.io.InLiveIn(2) <> loop_1.io.OutLiveIn.elements("field2")(0)//merge_1.io.Out(0)

//   loop_0.io.InLiveIn(3) <>  loop_1.io.OutLiveIn.elements("field3")(0)//merge_2.io.Out(0)

//   loop_0.io.InLiveIn(4) <> loop_1.io.OutLiveIn.elements("field0")(0)


//   loop_1.io.InLiveIn(3) <> merge_2.io.Out(0)
  
//   loop_1.io.InLiveIn(2) <> loop_2.io.OutLiveIn.elements("field3")(1)


//   loop_1.io.InLiveIn(0) <> loop_2.io.OutLiveIn.elements("field0")(0)

//   // loop_1.io.InLiveIn(1) <> loop_2.io.OutLiveIn.elements("field1")(0)

//   loop_1.io.InLiveIn(1) <> loop_2.io.OutLiveIn.elements("field2")(0)


//   // loop_3.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

//   // loop_3.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)

//   // loop_3.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)


//   // loop_2.io.InLiveIn(0) <> merge_1.io.Out(1)

//   // loop_2.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)

//   loop_2.io.InLiveIn(3) <> loop_3.io.CarryDepenOut.elements("field0")(0)//merge_1.io.Out(0)

//   loop_3.io.OutLiveIn.elements("field0")(0) <> loop_2.io.InLiveIn(0)

//   loop_3.io.OutLiveIn.elements("field1")(0) <> loop_2.io.InLiveIn(1)

//   loop_3.io.OutLiveIn.elements("field2")(0) <> loop_2.io.InLiveIn(2)
  

//   loop_3.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)

//   loop_3.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)

//   loop_3.io.InLiveIn(2) <> FineGrainedArgCall.io.Out.data.elements("field2")(0)
//   /* ================================================================== *
//    *                   Live-in dependencies.                            *
//    * ================================================================== */

//   int_shl_7.io.LeftIO <> loop_0.io.OutLiveIn.elements("field0")(0)

//   int_add_8.io.LeftIO <> loop_0.io.OutLiveIn.elements("field0")(1)

//   int_add_12.io.RightIO <> loop_0.io.OutLiveIn.elements("field0")(2)

//   address_10.io.baseAddress <> loop_0.io.OutLiveIn.elements("field1")(0)

//   int_add_12.io.LeftIO <> loop_0.io.OutLiveIn.elements("field2")(0)

//   int_add_14.io.RightIO <> loop_0.io.OutLiveIn.elements("field3")(0)

//   address_16.io.baseAddress <> loop_0.io.OutLiveIn.elements("field4")(0)


//   int_shl_26.io.LeftIO <> loop_2.io.OutLiveIn.elements("field3")(0)//loop_2.io.OutLiveIn.elements("field0")(0)

//   address_28.io.baseAddress <> loop_2.io.OutLiveIn.elements("field1")(0)



//   /* ================================================================== *
//    *                   Output Data dependencies.                        *
//    * ================================================================== */

//   loop_0.io.InLiveOut(0) <> int_add_19.io.Out(0)

//   loop_1.io.InLiveOut(0) <> loop_0.io.OutLiveOut.elements("field0")(0)//loop_1.io.OutLiveOut.elements("field0")(0)

//   loop_1.io.OutLiveOut.elements("field0")(0) <> store_29.inData
  
//   store_29.GepAddr <> address_28.io.Out(0)

//   /* ================================================================== *
//    *                   Live-out dependencies.                           *
//    * ================================================================== */

  

//   /* ================================================================== *
//    *                   Carry dependencies                               *
//    * ================================================================== */

//   loop_0.io.CarryDepenIn(1) <> int_add_20.io.Out(1)// loop_0.io.CarryDepenOut.elements("field0")(0)

//   // merge_5.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

//   loop_0.io.CarryDepenIn(0) <> int_add_19.io.Out(1)//loop_0.io.CarryDepenOut.elements("field1")(0)

//   // merge_6.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(0)//loop_0.io.CarryDepenOut.elements("field1")(1)

//   // loop_1.io.CarryDepenIn(0) <> loop_1.io.CarryDepenOut.elements("field1")(0)//loop_1.io.CarryDepenOut.elements("field0")(0)

//   // merge_3.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)

//   loop_1.io.CarryDepenIn(0) <> int_add_23.io.Out(1)//loop_1.io.CarryDepenOut.elements("field1")(0)

//   // merge_4.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field1")(0)

//   loop_2.io.CarryDepenIn(0) <> int_add_30.io.Out(1)//loop_2.io.CarryDepenOut.elements("field0")(0)

//   merge_2.io.InData(1) <> loop_2.io.CarryDepenOut.elements("field0")(0)

//   loop_3.io.CarryDepenIn(0) <> int_add_33.io.Out(1)//loop_3.io.CarryDepenOut.elements("field0")(0)

//   // merge_1.io.InData(1) <> loop_3.io.CarryDepenOut.elements("field0")(0)



//   /* ================================================================== *
//    *                   Printing Connection.                             *
//    * ================================================================== */

//   // merge_5.io.Mask <> exe_block_0.io.MaskBB(0)

//   // merge_6.io.Mask <> exe_block_0.io.MaskBB(1)

//   // merge_3.io.Mask <> exe_block_1.io.MaskBB(0)

//   // merge_4.io.Mask <> exe_block_1.io.MaskBB(1)

//   merge_2.io.Mask <> exe_block_2.io.MaskBB(0)

//   // merge_1.io.Mask <> exe_block_3.io.MaskBB(0)

//   // merge_1.io.InData(0) <> int_const_0.io.Out

//   merge_2.io.InData(0) <> int_const_1.io.Out

//   // merge_3.io.InData(0) <> int_const_2.io.Out

//   // merge_4.io.InData(0) <> int_const_3.io.Out

//   // merge_6.io.InData(0) <> int_const_4.io.Out

//   int_shl_7.io.RightIO <> int_const_5.io.Out

//   int_shl_13.io.RightIO <> int_const_6.io.Out

//   int_add_20.io.LeftIO <> int_const_7.io.Out

//   int_cmp_21.io.LeftIO <> int_const_8.io.Out

//   int_add_23.io.LeftIO <> int_const_9.io.Out

//   int_cmp_24.io.LeftIO <> int_const_10.io.Out

//   int_shl_26.io.RightIO <> int_const_11.io.Out

//   int_add_30.io.LeftIO <> int_const_12.io.Out

//   int_cmp_31.io.LeftIO <> int_const_13.io.Out

//   int_add_33.io.LeftIO <> int_const_14.io.Out

//   int_cmp_34.io.LeftIO <> int_const_15.io.Out

//   int_add_33.io.RightIO <> loop_3.io.CarryDepenOut.elements("field0")(1)//merge_1.io.Out(1)

//   int_add_27.io.RightIO <> merge_2.io.Out(1)

//   int_add_30.io.RightIO <> merge_2.io.Out(2)

//   // merge_5.io.InData(0) <> merge_3.io.Out(0)

  

//   int_add_23.io.RightIO <> loop_1.io.CarryDepenOut.elements("field0")(1)//merge_4.io.Out(1)

//   int_add_19.io.LeftIO <> loop_0.io.CarryDepenOut.elements("field0")(0)//merge_5.io.Out(0)

//   int_add_9.io.RightIO <> loop_0.io.CarryDepenOut.elements("field1")(0)//merge_6.io.Out(0)

//   int_add_15.io.RightIO <> loop_0.io.CarryDepenOut.elements("field1")(1)//merge_6.io.Out(1)

//   int_add_20.io.RightIO <> loop_0.io.CarryDepenOut.elements("field1")(2)//merge_6.io.Out(2)

//   int_add_8.io.RightIO <> int_shl_7.io.Out(0)

//   int_add_9.io.LeftIO <> int_add_8.io.Out(0)

//   address_10.io.idx(0) <> int_add_9.io.Out(0)

//   load_11.GepAddr <> address_10.io.Out(0)

//   int_mul_18.io.LeftIO <> load_11.io.Out(0)

//   int_shl_13.io.LeftIO <> int_add_12.io.Out(0)

//   int_add_14.io.LeftIO <> int_shl_13.io.Out(0)

//   int_add_15.io.LeftIO <> int_add_14.io.Out(0)

//   address_16.io.idx(0) <> int_add_15.io.Out(0)

//   load_17.GepAddr <> address_16.io.Out(0)

//   int_mul_18.io.RightIO <> load_17.io.Out(0)

//   int_add_19.io.RightIO <> int_mul_18.io.Out(0)

//   int_cmp_21.io.RightIO <> int_add_20.io.Out(0)

//   state_branch_22.io.CmpIO <> int_cmp_21.io.Out(0)

//   int_cmp_24.io.RightIO <> int_add_23.io.Out(0)

//   state_branch_25.io.CmpIO <> int_cmp_24.io.Out(0)

//   int_add_27.io.LeftIO <> int_shl_26.io.Out(0)

//   address_28.io.idx(0) <> int_add_27.io.Out(0)

//   int_cmp_31.io.RightIO <> int_add_30.io.Out(0)

//   state_branch_32.io.CmpIO <> int_cmp_31.io.Out(0)

//   int_cmp_34.io.RightIO <> int_add_33.io.Out(0)

//   state_branch_35.io.CmpIO <> int_cmp_34.io.Out(0)

//   mem_ctrl_cache.io.load_address(0) <> load_11.address_out

//   load_11.data_in <> mem_ctrl_cache.io.load_data(0)


//   mem_ctrl_cache.io.load_address(1) <> load_17.address_out

//   load_17.data_in <> mem_ctrl_cache.io.load_data(1)

//   mem_ctrl_cache_store.io.store_address(0)<> store_29.address_out

//   store_29.io.Out(0) <> mem_ctrl_cache_store.io.store_data(0)



//   /* ================================================================== *
//    *                   Printing Execution Block Enable.                 *
//    * ================================================================== */

//   // int_const_4.io.enable <> exe_block_0.io.Out(0)

//   int_const_5.io.enable <> exe_block_0.io.Out(1)

//   int_const_6.io.enable <> exe_block_0.io.Out(2)

//   int_const_7.io.enable <> exe_block_0.io.Out(3)

//   int_const_8.io.enable <> exe_block_0.io.Out(4)

//   // merge_5.io.enable <> exe_block_0.io.Out(5)

//   // merge_6.io.enable <> exe_block_0.io.Out(6)

//   int_shl_7.io.enable <> exe_block_0.io.Out(7)

//   int_add_8.io.enable <> exe_block_0.io.Out(8)

//   int_add_9.io.enable <> exe_block_0.io.Out(9)

//   address_10.io.enable <> exe_block_0.io.Out(10)

//   load_11.io.enable <> exe_block_0.io.Out(11)

//   int_add_12.io.enable <> exe_block_0.io.Out(12)

//   int_shl_13.io.enable <> exe_block_0.io.Out(13)

//   int_add_14.io.enable <> exe_block_0.io.Out(14)

//   int_add_15.io.enable <> exe_block_0.io.Out(15)

//   address_16.io.enable <> exe_block_0.io.Out(16)

//   load_17.io.enable <> exe_block_0.io.Out(17)

//   int_mul_18.io.enable <> exe_block_0.io.Out(18)

//   int_add_19.io.enable <> exe_block_0.io.Out(19)

//   int_add_20.io.enable <> exe_block_0.io.Out(5)

//   int_cmp_21.io.enable <> exe_block_0.io.Out(0)

//   state_branch_22.io.enable <> exe_block_0.io.Out(6)

//   // int_const_2.io.enable <> exe_block_1.io.Out(1)

//   // int_const_3.io.enable <> exe_block_1.io.Out(2)

//   int_const_9.io.enable <> exe_block_1.io.Out(3)

//   int_const_10.io.enable <> exe_block_1.io.Out(4)

//   // merge_3.io.enable <> exe_block_1.io.Out(5)

//   // merge_4.io.enable <> exe_block_1.io.Out(6)

//   int_add_23.io.enable <> exe_block_1.io.Out(2)

//   int_cmp_24.io.enable <> exe_block_1.io.Out(1)

//   // state_branch_25.io.enable <> exe_block_1.io.Out(9)

//   int_const_1.io.enable <> exe_block_2.io.Out(1)

//   int_const_11.io.enable <> exe_block_2.io.Out(2)

//   int_const_12.io.enable <> exe_block_2.io.Out(3)

//   int_const_13.io.enable <> exe_block_2.io.Out(4)

//   merge_2.io.enable <> exe_block_2.io.Out(5)

//   int_shl_26.io.enable <> exe_block_2.io.Out(6)

//   int_add_27.io.enable <> exe_block_2.io.Out(7)

//   address_28.io.enable <> exe_block_2.io.Out(8)

//   store_29.io.enable <> exe_block_2.io.Out(9)

//   int_add_30.io.enable <> exe_block_2.io.Out(10)

//   int_cmp_31.io.enable <> exe_block_2.io.Out(11)

//   // state_branch_32.io.enable <> exe_block_2.io.Out(12)

//   // int_const_0.io.enable <> exe_block_3.io.Out(1)

//   int_const_14.io.enable <> exe_block_3.io.Out(2)

//   int_const_15.io.enable <> exe_block_3.io.Out(3)

//   // merge_1.io.enable <> exe_block_3.io.Out(4)

//   int_add_33.io.enable <> exe_block_3.io.Out(4)

//   int_cmp_34.io.enable <> exe_block_3.io.Out(1)

//   // state_branch_35.io.enable <> exe_block_3.io.Out(7)

//   io.out <> return_36.io.Out

// }

