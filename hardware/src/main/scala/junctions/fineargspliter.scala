package heteacc.junctions

import chisel3._
import chisel3.util._
import heteacc.interfaces._
import chipsalliance.rocketchip.config._
import heteacc.config.HasAccelParams

class SplitCallDCRIO(val argTypes: Seq[Int])(implicit p: Parameters) extends Bundle {
  val In = Flipped(Decoupled(new Call(Seq.fill(argTypes.length)(32))))
  val Out = new CallDecoupledVec(argTypes)

}

class SplitCallDCR(val argTypes: Seq[Int])(implicit p: Parameters) extends Module {
  val io = IO(new SplitCallDCRIO(argTypes))
  val inputReg  = RegInit(0.U.asTypeOf(io.In.bits))
 
  val enableValidReg = RegInit(false.B)

  val outputValidReg = RegInit(VecInit(Seq.fill(argTypes.length)(VecInit(Seq.fill(32)(false.B)))))
  val allValid = for(i <- argTypes.indices) yield {
    val allValid = outputValidReg(i).reduceLeft(_ || _)
    allValid
  }

  val s_idle :: s_latched :: Nil = Enum(2)
  val state = RegInit(s_idle)

  io.In.ready := state === s_idle

  switch(state) {
    is(s_idle) {
      when (io.In.fire) {
        state := s_latched
        inputReg <> io.In.bits
      }
    }
    is(s_latched) {
      when (!allValid.reduceLeft(_ || _) && !enableValidReg) {
        state := s_idle
      }
    }
  }

  for (i <- argTypes.indices) {
    for (j <- 0 until argTypes(i)) {
      when(io.In.valid && state === s_idle) {
        outputValidReg(i)(j) := true.B
      }
      when(state === s_latched && io.Out.data(s"field$i")(j).ready) {
        outputValidReg(i)(j) := false.B
      }
      io.Out.data(s"field$i")(j).valid := outputValidReg(i)(j)
      io.Out.data(s"field$i")(j).bits := inputReg.data(s"field$i")
    }
  }

  when(io.In.valid && state === s_idle) {
    enableValidReg := true.B
  }
  when(state === s_latched && io.Out.enable.ready){
    enableValidReg := false.B
  }
  io.Out.enable.valid := enableValidReg
  io.Out.enable.bits := inputReg.enable

}