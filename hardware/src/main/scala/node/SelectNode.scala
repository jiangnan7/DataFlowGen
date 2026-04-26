package heteacc.node

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, Driver, OrderedDecoupledHWIOTester, PeekPokeTester}
import chisel3.Module
import chisel3.testers._
import chisel3.util._
import org.scalatest.{FlatSpec, Matchers}
import chipsalliance.rocketchip.config._
import heteacc.config._
import heteacc.interfaces._
import muxes._
import util._
import utility.UniformPrintfs


class SelectNodeIO(NumOuts: Int, Debug:Boolean=false)
                  (implicit p: Parameters)
  extends HandShakingIONPS(NumOuts, Debug )(new DataBundle) {

  // Input data 1
  val InData1 = Flipped(Decoupled(new DataBundle()))

  // Input data 2
  val InData2 = Flipped(Decoupled(new DataBundle()))

  // Select input data
  val Select = Flipped(Decoupled(new DataBundle()))

}

class SelectNode(NumOuts: Int, ID: Int, Debug : Boolean = false)
                (implicit p: Parameters,
                 name: sourcecode.Name,
                 file: sourcecode.File)
  extends HandShakingNPS(NumOuts, ID, Debug)(new DataBundle())(p) {
  override lazy val io = IO(new SelectNodeIO(NumOuts, Debug))

  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize

  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  /*===========================================*
   *            Registers                      *
   *===========================================*/
  // Indata1 Input
  val indata1_R = RegInit(DataBundle.default)
  val indata1_valid_R = RegInit(false.B)

  // Indata2 Input
  val indata2_R = RegInit(DataBundle.default)
  val indata2_valid_R = RegInit(false.B)

  // Select Input
  val select_R = RegInit(DataBundle.default)
  val select_valid_R = RegInit(false.B)

  val s_IDLE :: s_COMPUTE :: Nil = Enum(2)
  val state = RegInit(s_IDLE)


  val predicate = enable_R.control | io.enable.bits.control
  val taskID = Mux(enable_valid_R, enable_R.taskID ,io.enable.bits.taskID)

  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/

  io.InData1.ready := ~indata1_valid_R
  when(io.InData1.fire) {
    indata1_R <> io.InData1.bits
    indata1_valid_R := true.B
  }

  io.InData2.ready := ~indata2_valid_R
  when(io.InData2.fire) {
    indata2_R <> io.InData2.bits
    indata2_valid_R := true.B
  }

  io.Select.ready := ~select_valid_R
  when(io.Select.fire) {
    select_R <> io.Select.bits
    select_valid_R := true.B
  }

  // Wire up Outputs
  val output_data = Mux(select_R.data.orR, indata1_R.data, indata2_R.data)

    // The taskID's should be identical except in the case
    // when one input is tied to a constant.  In that case
    // the taskID will be zero.  Logical OR'ing the IDs
    // Should produce a valid ID in either case regardless of
    // which input is constant.
    io.Out.foreach(_.bits := DataBundle(output_data, taskID, predicate))

  /*============================================*
   *            State Machine                   *
   *============================================*/
  switch(state) {
    is(s_IDLE) {
        when(enable_valid_R && indata1_valid_R && indata2_valid_R && select_valid_R) {
          io.Out.foreach( _.valid := true.B)
          ValidOut()
          state := s_COMPUTE
          if(log){
            printf(p"[LOG] [${module_name}] [TID: %d] [SELECT] " +
              p"[${node_name}] [Task: ${taskID}] [Out: ${output_data}] [Cycle: ${cycleCount}]\n")
          }
        }
    }
    is(s_COMPUTE) {
      when(IsOutReady()) {
        // Reset data
        indata1_valid_R := false.B
        indata2_valid_R := false.B
        select_valid_R := false.B

        indata1_R := DataBundle.default
        indata2_R := DataBundle.default
        //Reset state
        state := s_IDLE
        //Reset output
        Reset()
      }
    }
  }
}

class SelectNodeWithoutStateIO(NumOuts: Int, Debug:Boolean=false)
                  (implicit p: Parameters)
  extends HandShakingDynIO(NumOuts, Debug )(new DataBundle) {

  // Input data 1
  val InData1 = Flipped(Decoupled(new DataBundle()))

  // Input data 2
  val InData2 = Flipped(Decoupled(new DataBundle()))

  // Select input data
  val Select = Flipped(Decoupled(new DataBundle()))

}

class SelectNodeWithoutState(NumOuts: Int, ID: Int, Debug : Boolean = false)
                (implicit p: Parameters,
                 name: sourcecode.Name,
                 file: sourcecode.File)
  extends HandShakingDyn(NumOuts, ID, Debug)(new DataBundle())(p) {
  override lazy val io = IO(new SelectNodeWithoutStateIO(NumOuts, Debug))

  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize

  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "
  val (cycleCount, _) = Counter(true.B, 32 * 1024)


  // val predicate = enable_R.control | io.enable.bits.control
  // val taskID = Mux(enable_valid_R, enable_R.taskID ,io.enable.bits.taskID)


  private val join = Module(new Join(3))
  join.pValid(0) := io.InData1.valid
  join.pValid(1) := io.InData2.valid
  join.pValid(2) := io.Select.valid

  join.nReady := io.Out.map(_.ready).reduce(_ && _)//io.Out(0).ready

  io.InData1.ready := join.ready(0)
  io.InData2.ready := join.ready(1)
  io.Select.ready  := join.ready(2)

  // dataOut.valid := join.valid
  io.Out.foreach(_.valid := join.valid)


  // Wire up Outputs
  val output_data = Mux(io.Select.bits.data.orR, io.InData1.bits.data, io.InData2.bits.data)
  // private val buffer = ShiftRegister(output_data, 1)
  io.Out.foreach(_.bits := DataBundle(output_data, true.B, true.B))
  when(IsOutReady()){
    Reset()
    printf(p"[LOG] [${module_name}] [TID: %d] [SELECT] " +
      p"[${node_name}] [Task: ${true.B}] [Out: ${output_data}] [Cycle: ${cycleCount}]\n")
  }

  // private val join = Module(new Join(3))
  // private val oehb = Module(new OEHB(32))

  // join.pValid(0) := io.InData1.valid
  // join.pValid(1) := io.InData2.valid
  // join.pValid(2) := io.Select.valid

  // join.nReady := oehb.dataIn.ready

  // oehb.dataIn.bits := DontCare
  // oehb.dataIn.valid := join.valid

  // io.InData1.ready := join.ready(0)
  // io.InData2.ready := join.ready(1)
  // io.Select.ready  := join.ready(2)

  // // dataOut.valid := join.valid
  // // io.Out.foreach(_.valid := join.valid)
  // oehb.dataOut.ready := io.Out.map(_.ready).reduce(_ && _)

  // for (i <- 0 until NumOuts) {
  //   io.Out(i).valid := oehb.dataOut.valid
  // }

}


class SelectNodeWithoutStateIOSupportCarry(NumOuts: Int, Debug:Boolean=false)
                  (implicit p: Parameters)
  extends HandShakingIONPS(NumOuts, Debug )(new DataBundle) {

  // Input data 1
  val InData1 = Flipped(Decoupled(new DataBundle()))


  // Select input data
  val Select = Flipped(Decoupled(new DataBundle()))

}

class SelectNodeWithoutStateSupportCarry(NumOuts: Int, ID: Int, Debug : Boolean = false)
                (implicit p: Parameters,
                 name: sourcecode.Name,
                 file: sourcecode.File)
  extends HandShakingNPS(NumOuts, ID, Debug)(new DataBundle())(p) {
  override lazy val io = IO(new SelectNodeWithoutStateIOSupportCarry(NumOuts, Debug))

  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize

  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  val valueReg = RegInit(0.U(32.W))

  // val predicate = enable_R.control | io.enable.bits.control
  // val taskID = Mux(enable_valid_R, enable_R.taskID ,io.enable.bits.taskID)


  private val join = Module(new Join(2))
  join.pValid(0) := io.InData1.valid
  join.pValid(1) := io.Select.valid

  val nReadyReg = RegNext(io.Out.map(_.ready).reduce(_ && _), false.B)


  join.nReady := nReadyReg//io.Out(0).ready

  io.InData1.ready := join.ready(0)
  io.Select.ready  := join.ready(1)

  // dataOut.valid := join.valid
  io.Out.foreach(_.valid := join.valid)


  // Wire up Outputs
  val output_data = Mux(io.Select.bits.data.orR, io.InData1.bits.data, valueReg)
  io.Out.foreach(_.bits := DataBundle(output_data, true.B, true.B))
  when(IsOutReady()){
    Reset()
    valueReg := valueReg + output_data
    printf(p"[LOG] [${module_name}] [TID: %d] [SELECT] " +
      p"[${node_name}] [Task: ${true.B}] [Out: ${output_data}] [Cycle: ${cycleCount}]\n")
  }


}


class SelectNodeWithVectorizationIO(NumOuts: Seq[Int], NumLanes: Int)
                                   (implicit p: Parameters) extends AccelBundle {
  val InData1 = Vec(NumLanes, Flipped(Decoupled(new DataBundle())))
  val InData2 = Vec(NumLanes, Flipped(Decoupled(new DataBundle())))
  val Select = Vec(NumLanes, Flipped(Decoupled(new DataBundle())))

  val Out = new VariableDecoupledVec(NumOuts)
}

class SelectNodeWithVectorization(NumOuts: Seq[Int], NumLanes: Int, ID: Int)
                                 (implicit val p: Parameters,
                                  name: sourcecode.Name,
                                  file: sourcecode.File) extends Module with HasAccelParams with UniformPrintfs {

  val io = IO(new SelectNodeWithVectorizationIO(NumOuts, NumLanes))

  val node_name = name.value
  val module_name = file.value.split("/").last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  /*===========================================*
   *            Registers                      *
   *===========================================*/

  private val join = Module(new Join(NumLanes * 3))
  private val oehb = Module(new OEHB())

  for (lane <- 0 until NumLanes) {
    join.pValid(lane * 3) := io.InData1(lane).valid
    join.pValid(lane * 3 + 1) := io.InData2(lane).valid
    join.pValid(lane * 3 + 2) := io.Select(lane).valid
    io.InData1(lane).ready := join.ready(lane * 3)
    io.InData2(lane).ready := join.ready(lane * 3 + 1)
    io.Select(lane).ready := join.ready(lane * 3 + 2)
  }

  join.nReady := oehb.dataIn.ready
  oehb.dataIn.bits := DontCare
  oehb.dataIn.valid := join.valid

  /*===========================================*
   *            Validity and Output Logic      *
   *===========================================*/

  // def ValidOut(): Bool = {
  //   io.Out.elements.map { case (_, vec) =>
  //     vec.map(_.ready).reduce(_ && _)
  //   }.reduce(_ && _)
  // }

  // oehb.dataOut.ready := ValidOut()
  val allOutputsReady = io.Out.elements.values.flatMap(_.map(_.ready)).reduce(_ && _)
  oehb.dataOut.ready := allOutputsReady


  // for (i <- NumOuts.indices) {
  //   io.Out.elements(s"field$i").foreach(_.valid := oehb.dataOut.valid)
  // }
  val outputValid = oehb.dataOut.valid
  io.Out.elements.values.foreach(_.foreach(_.valid := outputValid))
  // for (i <- NumOuts.indices) {
  //   for (j <- 0 until NumOuts(i)) {
  //     io.Out.elements(s"field$i")(j).valid := oehb.dataOut.valid
  //   }
  // }



  val output_data = Wire(Vec(NumLanes, UInt(xlen.W)))

  for (lane <- 0 until NumLanes) {
    output_data(lane) := Mux(io.Select(lane).bits.data.orR,
                            io.InData1(lane).bits.data,
                            io.InData2(lane).bits.data)
  }
  /*===========================================*
   *            Output Mapping and Logic       *
   *===========================================*/

  for (i <- 0 until NumOuts.length) {
    for (j <- 0 until NumOuts(i)) {
      io.Out.elements(s"field$i")(j).bits := DataBundle(output_data(i), 0.U, 1.U)
      io.Out.elements(s"field$i")(j).valid := oehb.dataOut.valid
    }
  }

  // def IsOutReady(): Bool = {
  //   if (NumOuts.isEmpty) {
  //     true.B
  //   } else {
  //     val out_ready_R = RegInit(VecInit(Seq.fill(NumOuts.sum)(false.B)))
  //     val fire_mask = (out_ready_R zip io.Out.elements.flatMap { case (_, vec) =>
  //       vec.map(out => out.valid && out.ready)
  //     }).map { case (a, b) => a | b }
  //     fire_mask.reduce(_ && _)
  //   }
  // }
  def IsOutReady(): Bool = {
    if (NumOuts.isEmpty) {
      true.B
    } else {
      val fire_mask = io.Out.elements.values.flatMap(_.map(out => out.valid && out.ready))
      if (fire_mask.isEmpty) {
        false.B
      } else {
        RegNext(fire_mask.reduce(_ && _), false.B)
      }
    }
  }

  when(IsOutReady()) {
    Reset()
    if (log) {
      // 输出调试信息
      for (lane <- 0 until NumLanes) {
        printf(p"[LOG] [${module_name}] [TID: 0] [SELECT] [${node_name}] [Lane: ${lane}] " +
          p"[InData1: 0x${Hexadecimal(io.InData1(lane).bits.data)}] " +
          p"[InData2: 0x${Hexadecimal(io.InData2(lane).bits.data)}] " +
          p"[Select: 0x${Hexadecimal(io.Select(lane).bits.data)}] " +
          p"[Out: 0x${Hexadecimal(output_data(lane))}] [Cycle: ${cycleCount}]\n")
      }
    }
  }
}



//sbt "test:runMain heteacc.node.SelectNodeGen"
import java.io.PrintWriter
object SelectNodeGen extends App {
    implicit val p = new WithAccelConfig

    val verilogString = getVerilogString(new SelectNodeWithoutState(NumOuts = 1, ID = 1))
    val filePath = "RTL/Select/SelectNode.v"
    val writer = new PrintWriter(filePath)
    try {
        writer.write(verilogString)
    } finally {
        writer.close()
    }
}
