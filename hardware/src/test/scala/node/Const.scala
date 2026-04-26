package heteacc.node

import chisel3.iotesters.PeekPokeTester
import chipsalliance.rocketchip.config._
import heteacc.config._
import org.scalatest.{FlatSpec, Matchers}

class VectorConstTester(df: ConstFastNodeWithVectorization)
                       (implicit p: Parameters) extends PeekPokeTester(df) {
  val numLanes = df.io.Out.length

  private def printInput(): Unit = {
    val taskID = peek(df.io.enable.bits.taskID)
    val control = peek(df.io.enable.bits.control)
    println(s"Enable: taskID = $taskID, control = $control, valid = ${peek(df.io.enable.valid)}")
  }

  private def printOutputs(): Unit = {
    for (lane <- 0 until numLanes) {
      val outData = peek(df.io.Out(lane).bits.data)
      val outPred = peek(df.io.Out(lane).bits.predicate)
      val outValid = peek(df.io.Out(lane).valid)
      println(s"Output[$lane]: Data = $outData, Pred = $outPred, Valid = $outValid")
    }
  }

  poke(df.io.enable.valid, 0)
  poke(df.io.enable.bits.taskID, 0)
  poke(df.io.enable.bits.control, 0)
  for (lane <- 0 until numLanes) {
    poke(df.io.Out(lane).ready, 1)
  }

  println("=== Initial State ===")
  printInput()
  printOutputs()

  step(1)

  poke(df.io.enable.bits.taskID, 10)
  poke(df.io.enable.bits.control, 1)
  poke(df.io.enable.valid, 1)

  println("=== After Enabling Inputs and Outputs ===")
  printInput()
  printOutputs()

  step(1)
  println("=== Outputs After One Cycle ===")
  printOutputs()

  step(2)
  println("=== Outputs After Three Cycles ===")
  printOutputs()
}

class VectorConstTests extends FlatSpec with Matchers {
  implicit val p = new WithAccelConfig ++ new WithTestConfig

  it should "test vectorized ConstFastNodeWithVectorization" in {
    chisel3.iotesters.Driver.execute(
      Array("--target-dir", "generated_dut/",
        "--generate-vcd-output", "on",
        "-X", "verilog"),
      () => new ConstFastNodeWithVectorization(
        value = 42,
        NumLanes = 4,
        ID = 0
      )
    ) { c => new VectorConstTester(c) } should be(true)
  }
}
