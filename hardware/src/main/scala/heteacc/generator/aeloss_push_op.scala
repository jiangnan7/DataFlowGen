
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

// abstract class aeloss_pushDFIO(implicit val p: Parameters) extends Module with HasAccelParams {
// 	val io = IO(new Bundle {
// 	  val in = Flipped(Decoupled(new Call(List( 32, 32))))
// 	  // val MemResp = Flipped(Valid(new MemResp))
// 	  // val MemReq = Decoupled(new MemReq)
// 	  val out = Decoupled(new Call(List(32)))
// 	})
// }

// class aeloss_pushDF(implicit p: Parameters) extends aeloss_pushDFIO()(p){

//   val FineGrainedArgCall = Module(new SplitCallDCR(argTypes = List(1, 1 )))
//   FineGrainedArgCall.io.In <> io.in

//   //Cache
//   val mem_ctrl_cache = Module(new MemoryEngine(Size=2048, ID = 0, NumRead = 4, NumWrite = 0))
//   // val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 1, NumWrite = 0))
//   // io.MemReq <> mem_ctrl_cache.io.cache.MemReq
//   // mem_ctrl_cache.io.cache.MemResp <> io.MemResp
//   mem_ctrl_cache.initMem("dataset/aeloss_push/data.txt")
  
//   // val mem_ctrl_cache = Module(new CacheMemoryEngine(ID = 0, NumRead = 4, NumWrite = 0))

//   // io.MemReq <> mem_ctrl_cache.io.cache.MemReq
//   // mem_ctrl_cache.io.cache.MemResp <> io.MemResp



//   /* ================================================================== *
//    *                   Printing Const nodes.                            *
//    * ================================================================== */

//   //%c0_i32 = arith.constant 0 : i32
//   val int_const_0 = Module(new ConstFastNode(value = 0, ID = 0))

//   //%c0 = arith.constant 0 : index
//   val int_const_1 = Module(new ConstFastNode(value = 0, ID = 1))

//   //%c0 = arith.constant 0 : index
// //   val int_const_2 = Module(new ConstFastNode(value = 0, ID = 2))

//   //%c0_i32 = arith.constant 0 : i32
//   val int_const_3 = Module(new ConstFastNode(value = 0, ID = 3))

//   //%c1_i32 = arith.constant 1 : i32
//   val int_const_4 = Module(new ConstFastNode(value = 1, ID = 4))

//   //%c1_i32 = arith.constant 1 : i32
//   val int_const_5 = Module(new ConstFastNode(value = 1, ID = 5))

//   //%c0_i32 = arith.constant 0 : i32
//   val int_const_6 = Module(new ConstFastNode(value = 0, ID = 6))

//   //%c0_i32 = arith.constant 0 : i32
//   val int_const_7 = Module(new ConstFastNode(value = 0, ID = 7))

//   //%c94_i32 = arith.constant 94 : i32
//   val int_const_8 = Module(new ConstFastNode(value = 94, ID = 8))

//   //%c1 = arith.constant 1 : index
//   val int_const_9 = Module(new ConstFastNode(value = 1, ID = 9))

//   //%c1023 = arith.constant 1023 : index
//   val int_const_10 = Module(new ConstFastNode(value = 1023, ID = 10))

//   //%c1 = arith.constant 1 : index
//   val int_const_11 = Module(new ConstFastNode(value = 1, ID = 11))

//   //%c1023 = arith.constant 1023 : index
//   val int_const_12 = Module(new ConstFastNode(value = 1023, ID = 12))



//   /* ================================================================== *
//    *                   Printing Execution Block nodes.                  *
//    * ================================================================== */

//   // val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 34, NumPhi = 2, BID = 0))

//   // val exe_block_1 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 14, NumPhi = 2, BID = 1))

//   val exe_block_0 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 32, NumPhi = 1, BID = 0))

//   val exe_block_1 = Module(new ExecutionBlockNode(NumInputs = 1, NumOuts = 1, BID = 1))

//   val exe_block_2 = Module(new ExecutionBlockNode(NumInputs = 1, NumOuts = 1, BID = 2))

//   val exe_block_3 = Module(new BasicBlockNode(NumInputs = 2, NumOuts = 15, NumPhi = 2, BID = 3))

//   /* ================================================================== *
//    *                   Printing Operation nodes. 37                     *
//    * ================================================================== */

//   //dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
//   val state_branch_0 = Module(new UBranchNode(ID = 0))

//   //%4 = dataflow.merge %c0_i32 or %arg3 {Select = "Loop_Signal"} : i32
//   val merge_1 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 1, Res = false))

//   //%5 = dataflow.merge %c0 or %arg2 {Select = "Loop_Signal"} : index
//   val merge_2 = Module(new MergeNode(NumInputs = 2, NumOutputs = 5, ID = 2, Res = false))

//   //%6 = arith.index_cast %5 : index to i32
//   val cast_3 = Module(new BitCastNode(NumOuts = 1, ID = 3))

//   //%7 = dataflow.addr %arg1[%5] {memShape = [1024]} : memref<1024xi32>[index] -> i32
//   val address_4 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 4)(ElementSize = 1, ArraySize = List()))

//   //%8 = dataflow.load %7 : i32 -> i32
//   val load_5 = Module(new Load(NumOuts = 1, ID = 5, RouteID = 0))

//   //%9 = arith.trunci %8 : i32 to i1
//   val trunc_6 = Module(new BitCastNode(NumOuts = 2, ID = 6))

//   //dataflow.state %9, "if_then" or "if_else" : i1
//   val state_branch_7 = Module(new CBranchNodeIET(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 7))

//   //%17 = dataflow.merge %4 or %arg5 {Select = "Loop_Signal"} : i32
//   val merge_8 = Module(new MergeNode(NumInputs = 2, NumOutputs = 2, ID = 8, Res = false))

//   //%18 = dataflow.merge %c0 or %arg4 {Select = "Loop_Signal"} : index
// //   val merge_9 = Module(new MergeNode(NumInputs = 2, NumOutputs = 4, ID = 9, Res = false))

//   //%19 = arith.index_cast %18 : index to i32
//   val cast_10 = Module(new BitCastNode(NumOuts = 1, ID = 10))

//   //%20 = dataflow.addr %arg0[%5] {memShape = [1024]} : memref<1024xi32>[index] -> i32
//   val address_11 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 11)(ElementSize = 1, ArraySize = List()))

//   //%21 = dataflow.load %20 : i32 -> i32
//   val load_12 = Module(new Load(NumOuts = 1, ID = 12, RouteID = 1))

//   //%22 = dataflow.addr %arg0[%18] {memShape = [1024]} : memref<1024xi32>[index] -> i32
//   val address_13 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 13)(ElementSize = 1, ArraySize = List()))

//   //%23 = dataflow.load %22 : i32 -> i32
//   val load_14 = Module(new Load(NumOuts = 1, ID = 14, RouteID = 2))

//   //%24 = arith.subi %21, %23 : i32
//   val int_sub_15 = Module(new ComputeNode(NumOuts = 3, ID = 15, opCode = "Sub")(sign = false, Debug = false))

//   //%25 = arith.cmpi ugt, %24, %c0_i32 : i32
//   val int_cmp_16 = Module(new ComputeNode(NumOuts = 1, ID = 16, opCode = "ugt")(sign = false, Debug = false))

//   //%26 = arith.subi %c1_i32, %24 : i32
//   val int_sub_17 = Module(new ComputeNode(NumOuts = 1, ID = 17, opCode = "Sub")(sign = false, Debug = false))

//   //%27 = arith.addi %24, %c1_i32 : i32
//   val int_add_18 = Module(new ComputeNode(NumOuts = 1, ID = 18, opCode = "Add")(sign = false, Debug = false))

//   //%28 = arith.select %25, %26, %27 : i32
//   val select_19 = Module(new SelectNode(NumOuts = 2, ID = 19))

//   //%29 = arith.cmpi ugt, %28, %c0_i32 : i32
//   val int_cmp_20 = Module(new ComputeNode(NumOuts = 1, ID = 20, opCode = "ugt")(sign = false, Debug = false))

//   //%30 = arith.select %29, %28, %c0_i32 : i32
//   val select_21 = Module(new SelectNode(NumOuts = 1, ID = 21))

//   //%31 = dataflow.addr %arg1[%18] {memShape = [1024]} : memref<1024xi32>[index] -> i32
//   val address_22 = Module(new GepNode(NumIns = 1, NumOuts = 1, ID = 22)(ElementSize = 1, ArraySize = List()))

//   //%32 = dataflow.load %31 : i32 -> i32
//   val load_23 = Module(new Load(NumOuts = 1, ID = 23, RouteID = 3))

//   //%33 = arith.cmpi ne, %6, %19 : i32
//   val int_cmp_24 = Module(new ComputeNode(NumOuts = 1, ID = 24, opCode = "ne")(sign = false, Debug = false))

//   //%34 = arith.trunci %32 : i32 to i1
//   val trunc_25 = Module(new BitCastNode(NumOuts = 1, ID = 25))

//   //%35 = arith.andi %34, %33 : i1
//   val int_andi_26 = Module(new ComputeNode(NumOuts = 1, ID = 26, opCode = "and")(sign = false, Debug = false))

//   //%36 = arith.divsi %30, %c94_i32 : i32
//   val int_divsi27 = Module(new ComputeNode(NumOuts = 1, ID = 27, opCode = "udiv")(sign = false, Debug = false))

//   //%37 = arith.addi %17, %36 : i32
//   val int_add_28 = Module(new ComputeNode(NumOuts = 1, ID = 28, opCode = "Add")(sign = false, Debug = false))

//   //%38 = dataflow.select %35, %37, %17 : i32
//   val select_29 = Module(new SelectNode(NumOuts = 2, ID = 29))

//   //%39 = arith.addi %18, %c1 {Exe = "Loop"} : index
//   val int_add_30 = Module(new ComputeNode(NumOuts = 2, ID = 30, opCode = "Add")(sign = false, Debug = false))

//   //%40 = arith.cmpi eq, %39, %c1023 {Exe = "Loop"} : index
//   val int_cmp_31 = Module(new ComputeNode(NumOuts = 1, ID = 31, opCode = "eq")(sign = false, Debug = false))

//   //dataflow.state %40, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//   val state_branch_32 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 32))

//   //%15 = arith.index_cast %5 : index to i32
//   val cast_33 = Module(new BitCastNode(NumOuts = 1, ID = 33))

//   //%11 = dataflow.select %9, %10, %4 {Data = "IF-THEN-ELSE", Select = "Data"} : i32
//   val select_34 = Module(new SelectNode(NumOuts = 2, ID = 34))

//   //%12 = arith.addi %5, %c1 {Exe = "Loop"} : index
//   val int_add_35 = Module(new ComputeNode(NumOuts = 2, ID = 35, opCode = "Add")(sign = false, Debug = false))

//   //%13 = arith.cmpi eq, %12, %c1023 {Exe = "Loop"} : index
//   val int_cmp_36 = Module(new ComputeNode(NumOuts = 1, ID = 36, opCode = "eq")(sign = false, Debug = false))

//   //dataflow.state %13, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//   val state_branch_37 = Module(new CBranchNodeVariable(NumTrue = 1, NumFalse = 1, NumPredecessor = 0, ID = 37))

//   //func.return %0 : i32
//   val return_38 = Module(new RetNode2(retTypes = List(32), ID = 38))


//   /* ================================================================== *
//    *                   Printing Loop nodes.                             *
//    * ================================================================== */

//   val loop_0 = Module(new LoopBlockNode(NumIns = List(2, 1, 1, 1), NumOuts = List(1), NumCarry = List(4, 1), NumExits = 1, ID = 0))

//   val loop_1 = Module(new LoopBlockNode(NumIns = List(2, 1), NumOuts = List(1), NumCarry = List(1, 1), NumExits = 1, ID = 1))



//   /* ================================================================== *
//    *                   Control Signal.                                  *
//    * ================================================================== */

//   loop_1.io.loopExit(0) <> return_38.io.In.enable
//   loop_0.io.loopExit(0) <> DontCare//state_branch_37.io.enable

//   FineGrainedArgCall.io.Out.enable <> state_branch_0.io.enable


//   exe_block_0.io.predicateIn(0) <> loop_0.io.activate_loop_start

//   exe_block_0.io.predicateIn(1) <> loop_0.io.activate_loop_back

//   exe_block_1.io.predicateIn(0) <> state_branch_7.io.TrueOutput(0)

//   exe_block_2.io.predicateIn(0) <> state_branch_7.io.FalseOutput(0)

//   exe_block_3.io.predicateIn(0) <> loop_1.io.activate_loop_start

//   exe_block_3.io.predicateIn(1) <> loop_1.io.activate_loop_back



//   /* ================================================================== *
//    *                   Loop Control Signal.                             *
//    * ================================================================== */

//   loop_0.io.enable <> exe_block_1.io.Out(0)

//   loop_0.io.loopBack(0) <> state_branch_32.io.FalseOutput(0)

//   loop_0.io.loopFinish(0) <> state_branch_32.io.TrueOutput(0)

//   loop_1.io.enable <> state_branch_0.io.Out(0)

//   loop_1.io.loopBack(0) <> state_branch_37.io.FalseOutput(0)

//   loop_1.io.loopFinish(0) <> state_branch_37.io.TrueOutput(0)


//   /* ================================================================== *
//    *                   Loop dependencies.                               *
//    * ================================================================== */



//   /* ================================================================== *
//    *                   Input Data dependencies.                         *
//    * ================================================================== */

//   loop_0.io.InLiveIn(0) <> loop_1.io.OutLiveIn.elements("field1")(0)//FineGrainedArgCall.io.Out.data.elements("field0")(0)

//   loop_0.io.InLiveIn(1) <> merge_2.io.Out(0)

//   loop_0.io.InLiveIn(2) <> loop_1.io.OutLiveIn.elements("field0")(1)//FineGrainedArgCall.io.Out.data.elements("field1")(0)

//   loop_0.io.InLiveIn(3) <> cast_3.io.Out(0)


//   loop_1.io.InLiveIn(0) <> FineGrainedArgCall.io.Out.data.elements("field1")(0)
  
//   loop_1.io.InLiveIn(1) <> FineGrainedArgCall.io.Out.data.elements("field0")(0)


//   /* ================================================================== *
//    *                   Live-in dependencies.                            *
//    * ================================================================== */
 
//   address_11.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(0)

//   address_13.io.baseAddress <> loop_0.io.OutLiveIn.elements("field0")(1)

//   address_11.io.idx(0) <> loop_0.io.OutLiveIn.elements("field1")(0)

//   address_22.io.baseAddress <> loop_0.io.OutLiveIn.elements("field2")(0)

//   int_cmp_24.io.LeftIO <> loop_0.io.OutLiveIn.elements("field3")(0)

//   address_4.io.baseAddress <> loop_1.io.OutLiveIn.elements("field0")(0)



//   /* ================================================================== *
//    *                   Output Data dependencies.                        *
//    * ================================================================== */

//   loop_0.io.InLiveOut(0) <> select_29.io.Out(0)

//   loop_1.io.InLiveOut(0) <> select_34.io.Out(0)

//   loop_0.io.OutLiveOut.elements("field0")(0) <>   select_34.io.InData1

//   loop_1.io.OutLiveOut.elements("field0")(0) <>   return_38.io.In.data("field0")



//   /* ================================================================== *
//    *                   Live-out dependencies.                           *
//    * ================================================================== */



//   /* ================================================================== *
//    *                   Carry dependencies                               *
//    * ================================================================== */

//   loop_0.io.CarryDepenIn(0) <> int_add_30.io.Out(1)//loop_0.io.CarryDepenOut.elements("field0")(0)

// //   merge_9.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field0")(0)

//   loop_0.io.CarryDepenIn(1) <> select_29.io.Out(1)//loop_0.io.CarryDepenOut.elements("field1")(0)

//   merge_8.io.InData(1) <> loop_0.io.CarryDepenOut.elements("field1")(0)

//   loop_1.io.CarryDepenIn(0) <> int_add_35.io.Out(1)//loop_1.io.CarryDepenOut.elements("field0")(0)

//   merge_2.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field0")(0)

//   loop_1.io.CarryDepenIn(1) <> select_34.io.Out(1)//loop_1.io.CarryDepenOut.elements("field1")(0)

//   merge_1.io.InData(1) <> loop_1.io.CarryDepenOut.elements("field1")(0)



//   /* ================================================================== *
//    *                   Printing Connection.                             *
//    * ================================================================== */

//   merge_8.io.Mask <> exe_block_0.io.MaskBB(0)

// //   merge_9.io.Mask <> exe_block_0.io.MaskBB(1)

//   merge_1.io.Mask <> exe_block_3.io.MaskBB(0)

//   merge_2.io.Mask <> exe_block_3.io.MaskBB(1)

//   merge_1.io.InData(0) <> int_const_0.io.Out

//   merge_2.io.InData(0) <> int_const_1.io.Out

// //   merge_9.io.InData(0) <> int_const_2.io.Out

//   int_cmp_16.io.LeftIO <> int_const_3.io.Out

//   int_sub_17.io.LeftIO <> int_const_4.io.Out

//   int_add_18.io.LeftIO <> int_const_5.io.Out

//   int_cmp_20.io.LeftIO <> int_const_6.io.Out

//   select_21.io.Select <> int_const_7.io.Out

//   int_divsi27.io.LeftIO <> int_const_8.io.Out

//   int_add_30.io.LeftIO <> int_const_9.io.Out

//   int_cmp_31.io.LeftIO <> int_const_10.io.Out

//   int_add_35.io.LeftIO <> int_const_11.io.Out

//   int_cmp_36.io.LeftIO <> int_const_12.io.Out

//   merge_8.io.InData(0) <> merge_1.io.Out(0)

//   cast_3.io.Input <> merge_2.io.Out(1)

//   address_4.io.idx(0) <> merge_2.io.Out(2)

//   cast_33.io.Input <> merge_2.io.Out(3)

//   int_add_35.io.RightIO <> merge_2.io.Out(4)

//   load_5.GepAddr <> address_4.io.Out(0)

//   trunc_6.io.Input <> load_5.io.Out(0)

//   select_34.io.Select <> trunc_6.io.Out(0)



//   select_34.io.InData2 <> merge_1.io.Out(1)
// //////////////////////////////////////

//   state_branch_7.io.CmpIO <> trunc_6.io.Out(1)

//   int_add_28.io.LeftIO <> merge_8.io.Out(0)

//   select_29.io.InData2 <> merge_8.io.Out(1)

//   cast_10.io.Input <> loop_0.io.CarryDepenOut.elements("field0")(0)//merge_9.io.Out(0)

//   address_13.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(1)//merge_9.io.Out(1)

//   address_22.io.idx(0) <> loop_0.io.CarryDepenOut.elements("field0")(2)//merge_9.io.Out(2)

//   int_add_30.io.RightIO <> loop_0.io.CarryDepenOut.elements("field0")(3)//merge_9.io.Out(3)

//   int_cmp_24.io.RightIO <> cast_10.io.Out(0)

//   load_12.GepAddr <> address_11.io.Out(0)

//   int_sub_15.io.LeftIO <> load_12.io.Out(0)

//   load_14.GepAddr <> address_13.io.Out(0)

//   int_sub_15.io.RightIO <> load_14.io.Out(0)

//   int_cmp_16.io.RightIO <> int_sub_15.io.Out(0)

//   int_sub_17.io.RightIO <> int_sub_15.io.Out(1)

//   int_add_18.io.RightIO <> int_sub_15.io.Out(2)

//   select_19.io.Select <> int_cmp_16.io.Out(0)

//   select_19.io.InData1 <> int_sub_17.io.Out(0)

//   select_19.io.InData2 <> int_add_18.io.Out(0)

//   int_cmp_20.io.RightIO <> select_19.io.Out(0)

//   select_21.io.InData2 <> select_19.io.Out(1)

//   select_21.io.InData1 <> int_cmp_20.io.Out(0)

//   int_divsi27.io.RightIO <> select_21.io.Out(0)

//   load_23.GepAddr <> address_22.io.Out(0)

//   trunc_25.io.Input <> load_23.io.Out(0)

//   int_andi_26.io.RightIO <> int_cmp_24.io.Out(0)

//   int_andi_26.io.LeftIO <> trunc_25.io.Out(0)

//   select_29.io.Select <> int_andi_26.io.Out(0)

//   int_add_28.io.RightIO <> int_divsi27.io.Out(0)

//   select_29.io.InData1 <> int_add_28.io.Out(0)

//   int_cmp_31.io.RightIO <> int_add_30.io.Out(0)

//   state_branch_32.io.CmpIO <> int_cmp_31.io.Out(0)

//   select_34.io.InData1 <> cast_33.io.Out(0)

//   int_cmp_36.io.RightIO <> int_add_35.io.Out(0)

//   state_branch_37.io.CmpIO <> int_cmp_36.io.Out(0)



//   mem_ctrl_cache.io.load_address(0) <> load_5.address_out

//   load_5.data_in <> mem_ctrl_cache.io.load_data(0)

//   mem_ctrl_cache.io.load_address(1) <> load_12.address_out

//   load_12.data_in <> mem_ctrl_cache.io.load_data(1)

//   mem_ctrl_cache.io.load_address(2) <> load_14.address_out

//   load_14.data_in <> mem_ctrl_cache.io.load_data(2)

//   mem_ctrl_cache.io.load_address(3) <> load_23.address_out

//   load_23.data_in <> mem_ctrl_cache.io.load_data(3)



//   /* ================================================================== *
//    *                   Printing Execution Block Enable.                 *
//    * ================================================================== */

// //   int_const_2.io.enable <> exe_block_0.io.Out(0)

//   int_const_3.io.enable <> exe_block_0.io.Out(1)

//   int_const_4.io.enable <> exe_block_0.io.Out(2)

//   int_const_5.io.enable <> exe_block_0.io.Out(3)

//   int_const_6.io.enable <> exe_block_0.io.Out(4)

//   int_const_7.io.enable <> exe_block_0.io.Out(5)

//   int_const_8.io.enable <> exe_block_0.io.Out(6)

//   int_const_9.io.enable <> exe_block_0.io.Out(7)

//   int_const_10.io.enable <> exe_block_0.io.Out(8)

//   merge_8.io.enable <> exe_block_0.io.Out(9)

// //   merge_9.io.enable <> exe_block_0.io.Out(10)

//   cast_10.io.enable <> exe_block_0.io.Out(11)

//   address_11.io.enable <> exe_block_0.io.Out(12)

//   load_12.io.enable <> exe_block_0.io.Out(13)

//   address_13.io.enable <> exe_block_0.io.Out(14)

//   load_14.io.enable <> exe_block_0.io.Out(15)

//   int_sub_15.io.enable <> exe_block_0.io.Out(16)

//   int_cmp_16.io.enable <> exe_block_0.io.Out(17)

//   int_sub_17.io.enable <> exe_block_0.io.Out(18)

//   int_add_18.io.enable <> exe_block_0.io.Out(19)

//   select_19.io.enable <> exe_block_0.io.Out(20)

//   int_cmp_20.io.enable <> exe_block_0.io.Out(21)

//   select_21.io.enable <> exe_block_0.io.Out(22)

//   address_22.io.enable <> exe_block_0.io.Out(23)

//   load_23.io.enable <> exe_block_0.io.Out(24)

//   int_cmp_24.io.enable <> exe_block_0.io.Out(25)

//   trunc_25.io.enable <> exe_block_0.io.Out(26)

//   int_andi_26.io.enable <> exe_block_0.io.Out(27)

//   int_divsi27.io.enable <> exe_block_0.io.Out(28)

//   int_add_28.io.enable <> exe_block_0.io.Out(29)

//   select_29.io.enable <> exe_block_0.io.Out(30)

//   int_add_30.io.enable <> exe_block_0.io.Out(31)

//   int_cmp_31.io.enable <> exe_block_0.io.Out(10)

//   state_branch_32.io.enable <> exe_block_0.io.Out(0)

//   cast_33.io.enable <> exe_block_2.io.Out(0)

//   int_const_0.io.enable <> exe_block_3.io.Out(0)

//   int_const_1.io.enable <> exe_block_3.io.Out(1)

//   int_const_11.io.enable <> exe_block_3.io.Out(2)

//   int_const_12.io.enable <> exe_block_3.io.Out(3)

//   merge_1.io.enable <> exe_block_3.io.Out(4)

//   merge_2.io.enable <> exe_block_3.io.Out(5)

//   cast_3.io.enable <> exe_block_3.io.Out(6)

//   address_4.io.enable <> exe_block_3.io.Out(7)

//   load_5.io.enable <> exe_block_3.io.Out(8)

//   trunc_6.io.enable <> exe_block_3.io.Out(9)

//   state_branch_7.io.enable <> exe_block_3.io.Out(10)

//   select_34.io.enable <> exe_block_3.io.Out(11)

//   int_add_35.io.enable <> exe_block_3.io.Out(12)

//   int_cmp_36.io.enable <> exe_block_3.io.Out(13)

//   state_branch_37.io.enable <> exe_block_3.io.Out(14)

//   io.out <> return_38.io.Out

// }

