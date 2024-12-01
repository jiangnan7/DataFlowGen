package matrix

import chisel3._
import chisel3.util._
import chisel3.iotesters.{ChiselFlatSpec, Driver, OrderedDecoupledHWIOTester, PeekPokeTester}
import org.scalatest.{FlatSpec, Matchers}
import muxes._
import heteacc.config._
import utility._
import chipsalliance.rocketchip.config._

class SystolicBaseTests(df: SystolicSquareBuffered[UInt])(implicit p: Parameters) extends PeekPokeTester(df) {
  poke(df.io.activate, false.B)
  
  df.io.left.zipWithIndex.foreach { case (io, i) => poke(io, (i + 1).U) }
  df.io.right.zipWithIndex.foreach { case (io, i) => poke(io, (i + 1).U) }
  
  poke(df.io.activate, true.B)
  step(1)                      
  poke(df.io.activate, false.B)
  
  var totalCycles = 0
  var outputReady = false


  while (!outputReady && totalCycles < 1000) {
    if (peek(df.io.output.valid) == 1) {

      for (i <- 0 until df.N * df.N) {
        print(peek(df.io.output.bits(i)) + ",")
      }
      print("\n")
      outputReady = true
    } else {
      step(1) 
      totalCycles += 1
    }
  }

  print(s"Total cycles taken: $totalCycles\n")
}


class Systolic_Tester extends FlatSpec with Matchers {

    implicit val p = new WithAccelConfig(HeteaccAccelParams())

  it should "Typ Compute Tester" in {
    chisel3.iotesters.Driver.execute(Array(
      
      // "-ll", "Info",
        "-tn", "matrix",
        "-tbn", "verilator",
        "-td", s"test_run_dir/matrix",
        "-tts", "0001",
        "--generate-vcd-output", "on"),

      () => new SystolicSquareBuffered(UInt(32.W), 16)) {
      c => new SystolicBaseTests(c)
    } should be(true)
  }
}
