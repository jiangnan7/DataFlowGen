package heteacc.node

import chisel3.iotesters.PeekPokeTester
import chipsalliance.rocketchip.config._
import heteacc.config._
import org.scalatest.{FlatSpec, Matchers}

class VectorGepTester(df: GepNodeWithVectorization)
                     (implicit p: Parameters) extends PeekPokeTester(df) {

  val numIns = df.io.idx.length

  private def outputVec(fieldIdx: Int) = df.io.Out.elements(s"field$fieldIdx")

  private def printInputs(): Unit = {
    println("[TEST] --- Inputs ---")
    println(f"[TEST] BaseAddress valid: ${peek(df.io.baseAddress.valid)}")
    println(f"[TEST] BaseAddress data: 0x${peek(df.io.baseAddress.bits.data)}%X")

    for (i <- 0 until numIns) {
      println(f"[TEST] Index[$i] valid: ${peek(df.io.idx(i).valid)}")
      println(f"[TEST] Index[$i] data: 0x${peek(df.io.idx(i).bits.data)}%X")
    }
  }

  private def printOutputs(): Unit = {
    println("[TEST] --- Outputs ---")
    for (fieldIdx <- 0 until df.io.Out.elements.size) {
      val vec = outputVec(fieldIdx)
      for (idx <- 0 until vec.length) {
        println(f"[TEST] Output[field$fieldIdx][$idx] valid: ${peek(vec(idx).valid)}")
        println(f"[TEST] Output[field$fieldIdx][$idx] data: 0x${peek(vec(idx).bits.data)}%X")
      }
    }
  }

  poke(df.io.baseAddress.valid, 0)
  poke(df.io.baseAddress.bits.data, 0)
  for (i <- 0 until numIns) {
    poke(df.io.idx(i).valid, 0)
    poke(df.io.idx(i).bits.data, 0)
  }
  for (fieldIdx <- 0 until df.io.Out.elements.size) {
    val vec = outputVec(fieldIdx)
    for (idx <- 0 until vec.length) {
      poke(vec(idx).ready, 1)
    }
  }

  println("=== Initial State ===")
  printInputs()
  printOutputs()

  poke(df.io.baseAddress.valid, 1)
  poke(df.io.baseAddress.bits.data, 0x1000)
  for (i <- 0 until numIns) {
    poke(df.io.idx(i).valid, 1)
    poke(df.io.idx(i).bits.data, 0x10 + i)
  }

  println("=== After Providing Inputs ===")
  printInputs()
  printOutputs()

  step(1)
  println("=== Outputs After One Cycle ===")
  printOutputs()

  step(2)
  println("=== Outputs After Three Cycles ===")
  printOutputs()
}

class VectorGepTests extends FlatSpec with Matchers {
  implicit val p = new WithAccelConfig ++ new WithTestConfig

  it should "test vectorized GepNodeWithVectorization" in {
    chisel3.iotesters.Driver.execute(
      Array("--target-dir", "generated_dut/",
        "--generate-vcd-output", "on",
        "-X", "verilog"),
      () => new GepNodeWithVectorization(
        NumIns = 1,
        NumOuts = Seq(1, 1, 1, 1),
        NumLanes = 4,
        ID = 0
      )(
        ElementSize = 1,
        ArraySize = List()
      )
    ) { c => new VectorGepTester(c) } should be(true)
  }
}
