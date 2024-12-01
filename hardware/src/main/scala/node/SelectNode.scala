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
  extends HandShakingIONPS(NumOuts, Debug )(new DataBundle) {

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
  extends HandShakingNPS(NumOuts, ID, Debug)(new DataBundle())(p) {
  override lazy val io = IO(new SelectNodeWithoutStateIO(NumOuts, Debug))

  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize

  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "
  val (cycleCount, _) = Counter(true.B, 32 * 1024)


  val predicate = enable_R.control | io.enable.bits.control
  val taskID = Mux(enable_valid_R, enable_R.taskID ,io.enable.bits.taskID)


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
  io.Out.foreach(_.bits := DataBundle(output_data, taskID, predicate))
  when(IsOutReady()){
    Reset()
    printf(p"[LOG] [${module_name}] [TID: %d] [SELECT] " +
      p"[${node_name}] [Task: ${taskID}] [Out: ${output_data}] [Cycle: ${cycleCount}]\n")
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