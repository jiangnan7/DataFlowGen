package matrix

import chisel3._
import chisel3.Module
import chisel3.util._
import org.scalatest.{FlatSpec, Matchers}
import heteacc.config._
import heteacc.node._
import heteacc.interfaces._
import muxes._
import utility._
import heteacc.junctions._
import chipsalliance.rocketchip.config._
import heteacc.config._
import heteacc.interfaces._
import NastiConstants._
import chipsalliance.rocketchip.config._
import heteacc.config._
import utility._
import heteacc.memory.cache.HasCacheAccelParams
import heteacc.interfaces.axi._

class PEIO(implicit p: Parameters) extends AccelBundle()(p) {
  // LeftIO: Left input data for computation
  val Left = Input(Valid(UInt(xlen.W)))

  // RightIO: Right input data for computation
  val Top = Input(Valid(UInt(xlen.W)))

  val Right = Output(Valid(UInt(xlen.W)))

  val Bottom = Output(Valid(UInt(xlen.W)))

  val Out = Output(Valid(UInt(xlen.W)))

}


class PE[T <: Data : MAC.OperatorMAC](gen: T, left_delay: Int, top_delay: Int, val row: Int, val col: Int)(implicit val p: Parameters)
  extends Module with HasAccelParams with UniformPrintfs {
  val io = IO(new PEIO)

  val top_reg  = Pipe(io.Top.valid, io.Top.bits, latency = top_delay)
  val left_reg = Pipe(io.Left.valid, io.Left.bits, latency = left_delay)

  val accumalator       = RegInit(init = 0.U(xlen.W))
  val accumalator_valid = RegInit(init = false.B)
  when(top_reg.valid & left_reg.valid) {
    accumalator := MAC.mac(left_reg.bits.asTypeOf(gen), top_reg.bits.asTypeOf(gen), accumalator.asTypeOf(gen)).asUInt
    accumalator_valid := top_reg.valid & left_reg.valid
  }

  io.Right := left_reg

  io.Bottom := top_reg

  io.Out.bits := accumalator

  io.Out.valid := accumalator_valid

}



class SystolicSquareBuffered[T <: Data : MAC.OperatorMAC](gen: T, val N: Int)(implicit val p: Parameters)
  extends Module with HasAccelParams with UniformPrintfs {
  private val enableDebugPrint = false

  val io = IO(new Bundle {
    val left        = Input(Vec(N * N, UInt(xlen.W)))
    val right       = Input(Vec(N * N, UInt(xlen.W)))
    val activate    = Input(Bool( ))
    val async_reset = Input(Bool( ))
    val output      = Valid(Vec(N * N, UInt(xlen.W)))
  })

  def latency(): Int = {
    val latency = 3 * N
    latency
  }

  val PEs =
    for (i <- 0 until N) yield
      for (j <- 0 until N) yield {
        if (i == 0 & j == 0)
          Module(new PE(gen, left_delay = 0, top_delay = 0, row = 0, col = 0))
        else if (j == 0)
          Module(new PE(gen, left_delay = i, top_delay = 1, row = i, col = j))
        else if (i == 0)
          Module(new PE(gen, left_delay = 1, top_delay = j, row = i, col = j))
        else
          Module(new PE(gen, left_delay = 1, top_delay = 1, row = i, col = j))
      }

  /* PE Control */
  val s_idle :: s_ACTIVE :: s_COMPUTE :: Nil = Enum(3)
  val state                                  = RegInit(s_idle)

  val input_steps = new Counter(3 * N - 1)
  io.output.valid := Mux((input_steps.value === ((3 * N) - 2).U), true.B, false.B)
  when(state === s_idle) {
    when(io.activate) {
      state := s_ACTIVE
    }
  }.elsewhen(state === s_ACTIVE) {
    input_steps.inc( )
    when(input_steps.value === (N - 1).U) {
      state := s_COMPUTE
    }
  }.elsewhen(state === s_COMPUTE) {
    input_steps.inc( )
    when(input_steps.value === ((3 * N) - 2).U) {
      state := s_idle
    }
  }.otherwise {
    io.output.valid := false.B
  }

  val io_lefts = for (i <- 0 until N) yield
    for (j <- 0 until N) yield {
      j.U -> io.left(i * N + j)
    }

  val io_rights = for (i <- 0 until N) yield
    for (j <- 0 until N) yield {
      j.U -> io.right(i + j * N)
    }


  val left_muxes = for (i <- 0 until N) yield {
    val mx = MuxLookup(input_steps.value, 0.U, io_lefts(i))
    mx
  }

  val top_muxes = for (i <- 0 until N) yield {
    val mx = MuxLookup(input_steps.value, 0.U, io_rights(i))
    mx
  }


  for (i <- 0 until N) {
    for (j <- 0 until N) {
      if (j != N - 1) {
        PEs(i)(j + 1).io.Left <> PEs(i)(j).io.Right
      }
      if (i != N - 1) {
        PEs(i + 1)(j).io.Top <> PEs(i)(j).io.Bottom
      }
      if (i == 0) {
        PEs(0)(j).io.Top.bits := top_muxes(j)
      }
      if (j == 0) {
        PEs(i)(0).io.Left.bits := left_muxes(i)
      }
    }
  }

  for (i <- 0 until N) {
    PEs(0)(i).io.Top.valid := false.B
    PEs(i)(0).io.Left.valid := false.B
  }

  when(state === s_ACTIVE) {
    for (i <- 0 until N) {
      PEs(0)(i).io.Top.valid := true.B
      PEs(i)(0).io.Left.valid := true.B
    }
  }

  if (enableDebugPrint) {
    printf("\nGrid  %d \n", input_steps.value)
    for (i <- 0 until N) {
      for (j <- 0 until N) {
        printf(p"  0x${Hexadecimal(PEs(i)(j).io.Out.bits)}")
      }
      printf(p"\n")
    }
  }

  for (i <- 0 until N) {
    for (j <- 0 until N) {
      io.output.bits(i * (N) + j) <> PEs(i)(j).io.Out.bits

      when(state === s_idle) {
        PEs(i)(j).reset := true.B
      }
    }
  }
}


class SystolicSquareWrapper[T <: Data : MAC.OperatorMAC](gen: T, val N: Int)(implicit val p: Parameters)
  extends Module with HasAccelParams with UniformPrintfs {
  private val enableDebugPrint = false

  val io = IO(new Bundle {
    val input_data = Flipped(Decoupled(UInt(xlen.W)))
    val input_sop  = Input(Bool( ))
    val input_eop  = Input(Bool( ))

    val output     = Decoupled(UInt(xlen.W))
    val output_sop = Output(Bool( ))
    val output_eop = Output(Bool( ))
  })

  val s_idle :: s_read :: s_execute :: s_write :: Nil = Enum(4)
  val state                                           = RegInit(s_idle)

  val ScratchPad_input = RegInit(VecInit(Seq.fill(2 * N * N)(0.U(xlen.W))))

  val input_counter  = RegInit(0.U(log2Ceil(2 * N * N).W))
  val output_counter = RegInit(0.U(log2Ceil(N * N).W))
  val last_input_idx = (2 * N * N - 1).U(input_counter.getWidth.W)
  val last_output_idx = (N * N - 1).U(output_counter.getWidth.W)

  val PE = Module(new SystolicSquareBuffered(UInt(xlen.W), N))


  for (i <- 0 until 2 * N * N) {
    if (i < N * N) {
      PE.io.left(i) := ScratchPad_input(i)
    } else {
      PE.io.right(i - (N * N)) := ScratchPad_input(i)
    }
  }

  io.input_data.ready := ((state === s_idle) || (state === s_read))
  PE.io.activate := ((state === s_idle) || (state === s_read)) && io.input_data.fire && (input_counter === last_input_idx)
  PE.io.async_reset := false.B
  io.output.bits := 0.U
  io.output.valid := false.B

  io.output_sop := false.B
  io.output_eop := false.B

  switch(state) {
    is(s_idle) {
      when(io.input_data.fire) {
        ScratchPad_input(0) := io.input_data.bits
        output_counter := 0.U
        when(last_input_idx === 0.U) {
          input_counter := 0.U
          state := s_execute
        }.otherwise {
          input_counter := 1.U
          state := s_read
        }
      }
    }
    is(s_read) {
      when(io.input_data.fire) {
        ScratchPad_input(input_counter) := io.input_data.bits
        when(input_counter === last_input_idx) {
          input_counter := 0.U
          output_counter := 0.U
          state := s_execute
        }.otherwise {
          input_counter := input_counter + 1.U
        }
      }
    }
    is(s_execute) {
      when(PE.io.output.valid) {
        output_counter := 0.U
        state := s_write
      }
    }
    is(s_write) {
      io.output.valid := true.B
      io.output.bits := PE.io.output.bits(output_counter)
      //io.output.bits := output_counter.value
      //end-of-packet signal
      when(output_counter === last_output_idx) {
        io.output_eop := true.B
      }.otherwise {
        io.output_eop := false.B
      }

      //start-of-packet signal
      when(output_counter === 0.U) {
        io.output_sop := true.B
      }.otherwise {
        io.output_sop := false.B
      }
      when(io.output.fire) {
        when(output_counter === last_output_idx) {
          input_counter := 0.U
          output_counter := 0.U
          state := s_idle
        }.otherwise {
          output_counter := output_counter + 1.U
        }
      }
    }
  }

  if (enableDebugPrint) {
    printf(p"[DEBUG] State: ${state}\n")
  }
}


import java.io.PrintWriter
object SystolicSquareWrapperMain extends App {

  implicit val p = new WithAccelConfig(HeteaccAccelParams())
  val verilogString = getVerilogString(new SystolicSquareWrapper(UInt(32.W), 8))
  val filePath = "RTL/SystolicSquareWrapperMain.v"
  val writer = new PrintWriter(filePath)
    try {
      writer.write(verilogString)
    } finally {
      writer.close()
    }
}
