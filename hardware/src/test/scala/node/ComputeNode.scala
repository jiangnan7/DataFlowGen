package heteacc.node

import chisel3._
import chisel3.util._

import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester}
import org.scalatest.{Matchers, FlatSpec}

import chipsalliance.rocketchip.config._
import heteacc.config._

// Tester for vectorized ComputeNodeWithVectorization.
class VectorComputeTester(df: ComputeNodeWithVectorization)
                         (implicit p: Parameters) extends PeekPokeTester(df) {

  val NumLanes = df.io.LeftIO.length // Number of vector lanes

  def printInputs(): Unit = {
    for (lane <- 0 until NumLanes) {
      val leftData = peek(df.io.LeftIO(lane).bits.data)
      val rightData = peek(df.io.RightIO(lane).bits.data)
      val leftPred = peek(df.io.LeftIO(lane).bits.predicate)
      val rightPred = peek(df.io.RightIO(lane).bits.predicate)
      println(s"Lane $lane - Left: Data=$leftData, Pred=$leftPred | Right: Data=$rightData, Pred=$rightPred")
    }
  }

  def printOutputs(): Unit = {
    for ((name, vec) <- df.io.Out.elements) {
      for (idx <- vec.indices) {
        val outData = peek(vec(idx).bits.data)
        val outPred = peek(vec(idx).bits.predicate)
        val outValid = peek(vec(idx).valid)
        println(s"Output[$name][$idx] - Data=$outData, Pred=$outPred, Valid=$outValid")
      }
    }
  }

  for (lane <- 0 until NumLanes) {
    poke(df.io.LeftIO(lane).bits.data, lane + 1) // Example: lane 0 -> 1, lane 1 -> 2, ...
    poke(df.io.LeftIO(lane).bits.predicate, false.B)

    poke(df.io.RightIO(lane).bits.data, (lane + 1) * 2) // Example: lane 0 -> 2, lane 1 -> 4, ...
    poke(df.io.RightIO(lane).bits.predicate, false.B)

    poke(df.io.LeftIO(lane).valid, false.B)
    poke(df.io.RightIO(lane).valid, false.B)
  }

  for ((_, vec) <- df.io.Out.elements) {
    for (decoupled <- vec) {
      poke(decoupled.ready, false.B)
    }
  }

  println("Initial Inputs:")
  printInputs()

  step(1)

  for (lane <- 0 until NumLanes) {
    poke(df.io.LeftIO(lane).valid, true.B)
    poke(df.io.RightIO(lane).valid, true.B)
    poke(df.io.LeftIO(lane).bits.predicate, true.B)
    poke(df.io.RightIO(lane).bits.predicate, true.B)
  }

  for ((_, vec) <- df.io.Out.elements) {
    for (decoupled <- vec) {
      poke(decoupled.ready, true.B)
    }
  }

  println("Inputs After Enabling:")
  printInputs()
  println("Outputs After Enabling:")
  printOutputs()


}

// Test suite for the vectorized ComputeNodeWithVectorization.
class VectorCompTests extends FlatSpec with Matchers {
  implicit val p = new WithAccelConfig ++ new WithTestConfig

  it should "Test vectorized ComputeNodeWithVectorization" in {
    chisel3.iotesters.Driver.execute(
      Array("--target-dir", "generated_dut/", "--generate-vcd-output", "on", "-X", "verilog"),
      () => new ComputeNodeWithVectorization(
        NumOuts = Seq(1, 1, 1, 1),
        NumLanes = 4,
        ID = 0,
        opCode = "Add"
      )(sign = false, Debug = false)
    ) { c => new VectorComputeTester(c) } should be(true)
  }
}
