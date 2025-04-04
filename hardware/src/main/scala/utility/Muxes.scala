package muxes

import scala.math._
import heteacc.interfaces._
import chisel3._
import chisel3.util._
import chisel3.Module
import chipsalliance.rocketchip.config._
import heteacc.config._
import heteacc.interfaces._
import heteacc.config._

class Demux[T <: Data](gen: T, Nops: Int) extends Module {
  val io = IO(new Bundle {
    val en = Input(Bool())
    val input = Input(gen)
    val sel = Input(UInt(max(1, log2Ceil(Nops)).W))
    val outputs = Vec(Nops, Valid(gen))
  })

  io.outputs.foreach(_.bits := io.input)
  io.outputs.foreach(_.valid := false.B)

  when(io.en) {
    io.outputs(io.sel).valid := io.en
  }

}
// class Demux[T <: ValidT](gen: T, Nops: Int) extends Module {
//   val io = IO(new Bundle {
//     val en      = Input(Bool( ))
//     val input   = Input(gen)
//     val sel     = Input(UInt(max(1, log2Ceil(Nops)).W))
//     val outputs = Output(Vec(Nops, gen))
//     // val outputvalid = Output(Bool())
//   })

//   val x = io.sel

//   for (i <- 0 until Nops) {
//     io.outputs(i) := io.input
//   }
//   when(io.en) {
//     for (i <- 0 until Nops) {
//       io.outputs(i).valid := false.B
//     }
//     io.outputs(x).valid := true.B
//   }.otherwise {
//     for (i <- 0 until Nops) {
//       io.outputs(i).valid := false.B
//     }
//   }
//   // io.outputvalid := io.valids.asUInt.andR
// }

class DemuxGen[T <: Data](gen: T, Nops: Int) extends Module {
  val io = IO(new Bundle {
    val en = Input(Bool())
    val input = Input(gen)
    val sel = Input(UInt(max(1, log2Ceil(Nops)).W))
    val outputs = Output(Vec(Nops, gen))
    val valids = Output(Vec(Nops, Bool()))
    // val outputvalid = Output(Bool())
  })

  val x = io.sel

  for (i <- 0 until Nops) {
    io.outputs(i) := io.input
  }
  when(io.en) {
    for (i <- 0 until Nops) {
      io.valids(i) := false.B
    }
    io.valids(x) := true.B
  }.otherwise {
    for (i <- 0 until Nops) {
      io.valids(i) := false.B
    }
  }
}

abstract class AbstractDeMuxTree[T <: RouteID](Nops: Int, gen: T)(implicit p: Parameters)
  extends Module with HasAccelParams {
  val io = IO(new Bundle {
    val outputs = Vec(Nops, Output(Valid(gen)))//val outputs = Vec(Nops, Output(Valid(gen)))
    val input = Flipped(Valid(gen))
    val enable = Input(Bool())
  })
}

class DeMuxTree[T <: RouteID ](BaseSize: Int, NumOps: Int, gen: T)
                             (implicit val p: Parameters)
  extends AbstractDeMuxTree(NumOps, gen)(p) {
  require(NumOps > 0)
  require(isPow2(BaseSize))

  io.outputs.foreach(_ := DontCare)

  var prev = Seq.fill(0) {
    Module(new Demux(gen, 4)).io
  }
  var toplevel = Seq.fill(0) {
    Module(new Demux(gen, 4)).io
  }
  val SelBits = max(1, log2Ceil(BaseSize))
  var Level = (max(1, log2Ceil(NumOps)) + SelBits - 1) / SelBits
  var TopBits = Level * SelBits - 1
  var SelHIndex = 0
  var Muxes_Per_Level = (NumOps + BaseSize - 1) / BaseSize

  while (Muxes_Per_Level > 0) {
    val Demuxes = Seq.fill(Muxes_Per_Level) {
      val mux = Module(new Demux(gen, BaseSize))
      mux.io.input := DontCare
      mux
    }

    // io.outputs.foreach(_.RouteID := 0.U)
    Demuxes.foreach(_.io.input.RouteID := 0.U)

    if (prev.length != 0) {
      for (i <- 0 until prev.length) {
        val demuxInputReg = RegNext(Demuxes(i / BaseSize).io.outputs(indexcalc(i, BaseSize)))
        val demuxvalidreg = RegNext(init = false.B, next = Demuxes(i / BaseSize).io.outputs(indexcalc(i, BaseSize)).valid)
        prev(i).input := demuxInputReg
        prev(i).en := demuxvalidreg
        prev(i).sel := demuxInputReg.bits.RouteID(SelHIndex + log2Ceil(BaseSize) - 1, SelHIndex)
      }
      SelHIndex = SelHIndex + log2Ceil(BaseSize)
    }

    if (prev.length == 0) {
      toplevel = Demuxes.map(_.io)
      for (i <- 0 until Demuxes.length * BaseSize) {
        if (i < NumOps) {
          //println("Output["+i+"]"+"Source Demux["+i/BaseSize+","+indexcalc(i,BaseSize)+"]")
          io.outputs(i) <> Demuxes(i / BaseSize).io.outputs(indexcalc(i, BaseSize))
        }
      }
    }
    prev = Demuxes.map(_.io)
    if (Muxes_Per_Level == 1) {
      Muxes_Per_Level = 0
    } else {
      Muxes_Per_Level = (Muxes_Per_Level + BaseSize - 1) / BaseSize
    }
  }
  prev(0).input <> io.input
  prev(0).en := io.enable
  prev(0).sel := io.input.bits.RouteID(SelHIndex + log2Ceil(BaseSize) - 1, SelHIndex)

  object indexcalc {
    def apply(i: Int, BaseSize: Int): Int = {
      i - ((i / BaseSize) * BaseSize)
    }
  }

}


class Mux[T <: Data](gen: T, Nops: Int) extends Module {
  val io = IO(new Bundle {
    val en = Input(Bool())
    val output = Valid(gen)
    val sel = Input(UInt(max(1, log2Ceil(Nops)).W))
    val inputs = Input(Vec(Nops, gen))
  })

  val x = io.sel

  io.output.bits := io.inputs(x)
  io.output := DontCare
  when(!io.en) {
    io.output.valid := false.B
  }
}
