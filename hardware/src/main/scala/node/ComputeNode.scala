package heteacc.node

import chisel3._
import chisel3.util._
import chisel3.util.experimental.BoringUtils
import org.scalatest.{FlatSpec, Matchers}
import heteacc.config._
import chisel3.Module
import heteacc.interfaces._
import util._
import chipsalliance.rocketchip.config._
import heteacc.mul._
import utility.UniformPrintfs

class ComputeNodeIO(NumOuts: Int, Debug: Boolean)
                   (implicit p: Parameters)
  extends HandShakingIONPS(NumOuts, Debug)(new DataBundle) {
  val LeftIO = Flipped(Decoupled(new DataBundle()))
  val RightIO = Flipped(Decoupled(new DataBundle()))
}

class ComputeNode(NumOuts: Int, ID: Int, opCode: String)
                 (sign: Boolean, Debug: Boolean = false)
                 (implicit p: Parameters,
                  name: sourcecode.Name,
                  file: sourcecode.File)
  extends HandShakingNPS(NumOuts, ID, Debug)(new DataBundle())(p)
    with HasAccelShellParams{
  override lazy val io = IO(new ComputeNodeIO(NumOuts, Debug))
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)
  /*===========================================*
   *            Registers                      *
   *===========================================*/
  // Left Input
  val left_R = RegInit(DataBundle.default)
  val left_valid_R = RegInit(false.B)

  // Right Input
  val right_R = RegInit(DataBundle.default)
  val right_valid_R = RegInit(false.B)

  //Instantiate ALU with selected code
  val FU = Module(new UALU(xlen, opCode, issign = sign))

  // val FU = Module(new DSPALU(FixedPoint(32.W, 4.BP), opCode))
  val s_IDLE :: s_COMPUTE :: Nil = Enum(2)
  val state = RegInit(s_IDLE)

    //Output register
  val out_data_R = RegNext(Mux(enable_R.control, FU.io.out, 0.U), init = 0.U)
  val predicate = Mux(enable_valid_R, enable_R.control ,io.enable.bits.control)
  val taskID = Mux(enable_valid_R, enable_R.taskID ,io.enable.bits.taskID)

  /**
    * Debug variables
    */

  def IsInputValid(): Bool = {
    right_valid_R && left_valid_R
  }

  def isInFire(): Bool = {
    enable_valid_R && IsInputValid() && enable_R.control && state === s_IDLE
  }



  //Output register
  // val out_data_R = RegNext(FU.io.out)
  // val predicate = Mux(enable_valid_R, enable_R.control, io.enable.bits.control)
  // val taskID = Mux(enable_valid_R, enable_R.taskID, io.enable.bits.taskID)

 
  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/

  FU.io.in1 := left_R.data
  FU.io.in2 := right_R.data

  io.LeftIO.ready := ~left_valid_R
  when(io.LeftIO.fire) {
    // FU.io.in1 := io.LeftIO.bits.data
    left_R <> io.LeftIO.bits
    left_valid_R := true.B
  }

  io.RightIO.ready := ~right_valid_R
  when(io.RightIO.fire) {
    // FU.io.in2 := io.RightIO.bits.data
    right_R <> io.RightIO.bits
    right_valid_R := true.B
  }




  // val (guard_index, _) = Counter(isInFire(), GuardVals.length)


  // io.Out.foreach(_.bits := DataBundle(out_data_R ))
  io.Out.foreach(_.bits := DataBundle(out_data_R, taskID, predicate))

  /*============================================*
   *            State Machine                   *
   *============================================*/

  switch(state) {
    is(s_IDLE) {
      when(enable_valid_R && left_valid_R && right_valid_R) {

        io.Out.foreach(_.bits := DataBundle(out_data_R, taskID, predicate))
        io.Out.foreach(_.valid := true.B)
        ValidOut()
        left_valid_R := false.B
        right_valid_R := false.B
        state := s_COMPUTE
        if (log) {
          printf(p"[LOG] [${module_name}] [TID: ${taskID}] [COMPUTE] [Name: ${node_name}] " +
            p"[ID: ${ID}] " +
            p"[Pred: ${enable_R.control}] " +
            p"[In(0): 0x${Hexadecimal(left_R.data)}] " +
            p"[In(1) 0x${Hexadecimal(right_R.data)}] " +
            p"[Out: 0x${Hexadecimal(FU.io.out)}] " +
            p"[OpCode: ${opCode}] " +
            p"[Cycle: ${cycleCount}]\n")
        }
      }
    }
    is(s_COMPUTE) {
      when(IsOutReady()) {
        // Reset data
        out_data_R := 0.U
        //Reset state
        state := s_IDLE
        Reset()
      }
    }
  }

}


class DelayBuffer(latency: Int, size: Int = 32) extends MultiIOModule {
  val valid_in = IO(Input(UInt(size.W)))
  val ready_in = IO(Input(Bool()))
  val valid_out = IO(Output(UInt(size.W)))

  val shift_register = RegInit(VecInit(Seq.fill(latency)(0.U(size.W))))

  when(ready_in) {
    shift_register(0) := valid_in
    for (i <- 1 until latency) {
      shift_register(i) := shift_register(i - 1)
    }
  }
  valid_out := shift_register(latency - 1)
}
class Join(size: Int = 2) extends MultiIOModule {
  val pValid = IO(Vec(size, Input(Bool())))
  val ready = IO(Vec(size, Output(Bool())))

  val valid = IO(Output(Bool()))
  val nReady = IO(Input(Bool()))

  valid := pValid.foldLeft(true.B)(_ && _)

  for (i <- 0 until size) {
    ready(i) := pValid.zipWithIndex.filter(_._2 != i).foldLeft(nReady)(_ && _._1)
  }
}
class OEHB(size: Int = 32) extends MultiIOModule {
  val dataIn = IO(Flipped(DecoupledIO(UInt(size.W))))
  val dataOut = IO(DecoupledIO(UInt(size.W)))

  private val full_reg = RegInit(Bool(), false.B)
  private val reg_en = Wire(Bool())
  private val data_reg = RegInit(UInt(size.W), 0.U)

  dataOut.valid := full_reg
  dataIn.ready := (!full_reg) | dataOut.ready
  reg_en := dataIn.ready & dataIn.valid

  full_reg := dataIn.valid | (!dataIn.ready)

  when(reg_en) {
    data_reg := dataIn.bits
  }

  dataOut.bits := data_reg
}

class ComputeNodeWithoutStateIO(NumOuts: Int, Debug: Boolean)
                   (implicit p: Parameters)
  extends HandShakingIONPS(NumOuts, Debug)(new DataBundle) {
  val LeftIO = Flipped(Decoupled(new DataBundle()))
  val RightIO = Flipped(Decoupled(new DataBundle()))
}

class ComputeNodeWithoutState(NumOuts: Int, ID: Int, opCode: String)
                 (sign: Boolean, Debug: Boolean = false)
                 (implicit p: Parameters,
                  name: sourcecode.Name,
                  file: sourcecode.File)
  extends HandShakingNPS(NumOuts, ID, Debug)(new DataBundle())(p)
    with HasAccelShellParams{
  override lazy val io = IO(new ComputeNodeWithoutStateIO(NumOuts, Debug))
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)
  /*===========================================*
   *            Registers                      *
   *===========================================*/
  
  // val left_R = RegInit(DataBundle.default)
  // val left_valid_R = RegInit(false.B)

  // // Right Input
  // val right_R = RegInit(DataBundle.default)
  // val right_valid_R = RegInit(false.B)

  private val join = Module(new Join())
  private val oehb = Module(new OEHB(0))

  join.pValid(0) := io.LeftIO.valid
  join.pValid(1) := io.RightIO.valid
  io.LeftIO.ready := join.ready(0)
  io.RightIO.ready := join.ready(1)
  join.nReady := oehb.dataIn.ready

  oehb.dataIn.bits := DontCare
  // oehb.dataOut.ready := io.Out(0).ready
  oehb.dataIn.valid := join.valid
  // io.Out(0).valid := oehb.dataOut.valid

  oehb.dataOut.ready := io.Out.map(_.ready).reduce(_ && _)
  for (i <- 0 until NumOuts) {
    io.Out(i).valid := oehb.dataOut.valid
  }

  
  //Instantiate ALU with selected code
  val FU = Module(new UALU(xlen, opCode, issign = sign))

  val out_data_R = RegNext(Mux(enable_R.control, FU.io.out, 0.U), init = 0.U)
  val predicate = io.enable.bits.control
  val taskID = io.enable.bits.taskID

  FU.io.in1 := io.LeftIO.bits.data
  FU.io.in2 := io.RightIO.bits.data

  private val buffer = ShiftRegister(FU.io.out, 1)
  // io.RightIO.ready := true.B  
  // io.LeftIO.ready := true.B  
  // io.Out.foreach(_.bits := DataBundle(out_data_R ))
  io.Out.foreach(_.bits := DataBundle(buffer, taskID, predicate))
  when(IsOutReady()) {
    if (log) {
          printf(p"[LOG] [${module_name}] [TID: ${taskID}] [COMPUTE] [Name: ${node_name}] " +
            p"[ID: ${ID}] " +
            p"[Pred: ${enable_R.control}] " +
            p"[In(0): 0x${Hexadecimal(io.LeftIO.bits.data)}] " +
            p"[In(1) 0x${Hexadecimal(io.RightIO.bits.data)}] " +
            p"[Out: 0x${Hexadecimal(FU.io.out)}] " +
            p"[OpCode: ${opCode}] " +
            p"[Cycle: ${cycleCount}]\n")
        } 
    Reset()
  }
}

class ComputeNodeWithVectorizationIO(NumOuts: Seq[Int], NumLanes: Int, Debug: Boolean)
    (implicit p: Parameters) extends AccelBundle {

  val LeftIO = Vec(NumLanes, Flipped(Decoupled(new DataBundle())))
  val RightIO = Vec(NumLanes, Flipped(Decoupled(new DataBundle())))

  val Out = new VariableDecoupledVec(NumOuts)
}

class ComputeNodeWithVectorization(NumOuts: Seq[Int], NumLanes: Int, ID: Int, opCode: String)
                              (sign: Boolean, Debug: Boolean = false)
                              (implicit val p: Parameters,
                    name: sourcecode.Name,
                    file: sourcecode.File) extends Module with HasAccelParams with UniformPrintfs {


  val io = IO(new ComputeNodeWithVectorizationIO(NumOuts, NumLanes, Debug))

  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  /*===========================================*
   *            Registers                      *
   *===========================================*/

  private val join = Module(new Join(NumLanes*2))
  private val oehb = Module(new OEHB())


  for (lane <- 0 until NumLanes) {
    join.pValid(lane * 2) := io.LeftIO(lane).valid
    join.pValid(lane * 2 + 1) := io.RightIO(lane).valid
    io.LeftIO(lane).ready := join.ready(lane * 2)
    io.RightIO(lane).ready := join.ready(lane * 2 + 1)
  }

  join.nReady := oehb.dataIn.ready

  oehb.dataIn.bits := DontCare
  oehb.dataIn.valid := join.valid

  val in_live_in_valid_R = Seq.fill(NumOuts.length)(RegInit(false.B))

  def ValidOut(): Bool = {
    io.Out.elements.map { case (_, vec) =>
      vec.map(_.ready).reduce(_ && _)
    }.reduce(_ && _)
  }

  // for (i <- 0 until NumOuts.length) {
  //   in_live_in_valid_R(i) := io.Out.elements.map(_.ready).reduce(_ && _)
  // }


  oehb.dataOut.ready := ValidOut()

  for (i <- NumOuts.indices) {
    io.Out.elements(s"field$i").foreach(_.valid := oehb.dataOut.valid)

  }

  for (i <- NumOuts.indices) {
    for (j <- 0 until NumOuts(i)) {
        io.Out.elements(s"field$i")(j).valid := oehb.dataOut.valid
    }
  }



  val FUs = Seq.fill(NumLanes)(Module(new UALU(xlen, opCode, issign = sign)))

  for (lane <- 0 until NumLanes) {
      FUs(lane).io.in1 := io.LeftIO(lane).bits.data
      FUs(lane).io.in2 := io.RightIO(lane).bits.data
  }

  private val buffer = FUs.map(FU => ShiftRegister(FU.io.out, 1))



  for (i <- NumOuts.indices) {
    for (j <- 0 until NumOuts(i)) {
      io.Out.elements(s"field$i")(j).bits := DataBundle(buffer(i), 0.U, 1.U)
      io.Out.elements(s"field$i")(j).valid := oehb.dataOut.valid
    }
  }

  def IsOutReady(): Bool = {
    if (NumOuts.isEmpty) {
      true.B
    } else {
      val out_ready_R = RegInit(VecInit(Seq.fill(NumOuts.sum)(false.B)))
      val fire_mask = (out_ready_R zip io.Out.elements.flatMap { case (_, vec) =>
        vec.map(out => out.valid && out.ready)
      }).map { case (a, b) => a | b }
      fire_mask.reduce(_ && _)
    }
  }
  
  when(IsOutReady()) {
    if (Debug) {
      for (lane <- 0 until NumLanes) {
        printf(p"[LOG] [${module_name}] [TID: ${0}] [COMPUTE] [Name: ${node_name}] " +
          p"[ID: ${ID}] [Lane: ${lane}] " +
          p"[Pred: ${1}] " +
          p"[In(0): 0x${Hexadecimal(io.LeftIO(lane).bits.data)}] " +
          p"[In(1): 0x${Hexadecimal(io.RightIO(lane).bits.data)}] " +
          p"[Out: 0x${Hexadecimal(FUs(lane).io.out)}] " +
          p"[OpCode: ${opCode}] [Cycle: ${cycleCount}]\n")
      }
    }
    Reset()
  }

}
import java.io.PrintWriter
object ComputeNodeGen extends App {
  implicit val p = new WithAccelConfig ++ new WithTestConfig
  // sbt "test:runMain heteacc.node.ComputeNodeGen  -td ./test_run_dir"
  // println(getVerilogString(new ComputeNode(NumOuts = 1, ID = 0, opCode = "Add")(sign = false, Debug = false)))
  val opCodes = Seq(
    ("Add", 1), ("Sub", 2), ("And", 3), ("Or", 4), ("Xor", 5), 
    ("Xnor", 6), ("ShiftLeft", 7), ("ShiftRight", 8), ("ShiftRightLogical", 9), 
    ("ShiftRightArithmetic", 10), ("EQ", 11), ("NE", 12), ("LT", 13), 
    ("GT", 14), ("LTE", 15), ("GTE", 16), ("PassA", 17), ("PassB", 18), 
    ("Mul", 19), ("Udiv", 20), ("Max", 21), ("Min", 22)
  )

  opCodes.foreach { case (opName, opCode) =>
    val verilogString = getVerilogString(new ComputeNodeWithoutState(NumOuts = 1, ID = 0, opCode = opName)(sign = false, Debug = false))
    val filePath = s"RTL/ComputeNode/ComputeNode_$opName.v"
    val writer = new PrintWriter(filePath)
    try {
      writer.write(verilogString)
      println(s"Generated Verilog for $opName and saved to $filePath")
    } finally {
      writer.close()
    }
  }
  // (new chisel3.stage.ChiselStage).emitVerilog(new ComputeNode(NumOuts = 1, ID = 0, opCode = "Add")(sign = false, Debug = false))

} 
