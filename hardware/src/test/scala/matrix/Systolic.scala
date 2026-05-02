package matrix

import chisel3._
import chisel3.util._
import chisel3.iotesters.PeekPokeTester
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
