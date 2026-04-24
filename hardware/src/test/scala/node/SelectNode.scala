package heteacc.node

import chisel3.iotesters.PeekPokeTester
import chipsalliance.rocketchip.config._
import heteacc.config._
import org.scalatest.{FlatSpec, Matchers}

class VectorSelectTester(df: SelectNodeWithVectorization)
                        (implicit p: Parameters) extends PeekPokeTester(df) {

  private def outputVec(fieldIdx: Int) = df.io.Out.elements(s"field$fieldIdx")

  private def printInputs(): Unit = {
    for (lane <- 0 until df.io.InData1.length) {
      println(f"[TEST] InData1[$lane] valid: ${peek(df.io.InData1(lane).valid)}")
      println(f"[TEST] InData1[$lane] data: 0x${peek(df.io.InData1(lane).bits.data)}%X")
      println(f"[TEST] InData2[$lane] valid: ${peek(df.io.InData2(lane).valid)}")
      println(f"[TEST] InData2[$lane] data: 0x${peek(df.io.InData2(lane).bits.data)}%X")
      println(f"[TEST] Select[$lane] valid: ${peek(df.io.Select(lane).valid)}")
      println(f"[TEST] Select[$lane] data: 0x${peek(df.io.Select(lane).bits.data)}%X")
    }
  }

  private def printOutputs(): Unit = {
    for (fieldIdx <- 0 until df.io.Out.elements.size) {
      val vec = outputVec(fieldIdx)
      for (idx <- 0 until vec.length) {
        println(f"[TEST] Output[field$fieldIdx][$idx] valid: ${peek(vec(idx).valid)}")
        println(f"[TEST] Output[field$fieldIdx][$idx] data: 0x${peek(vec(idx).bits.data)}%X")
      }
    }
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

  for (lane <- 0 until df.io.InData1.length) {
    poke(df.io.InData1(lane).valid, 1)
    poke(df.io.InData1(lane).bits.data, 0x10 + lane)

    poke(df.io.InData2(lane).valid, 1)
    poke(df.io.InData2(lane).bits.data, 0x20 + lane)

    poke(df.io.Select(lane).valid, 1)
    poke(df.io.Select(lane).bits.data, lane % 2)
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

class VectorSelectTests extends FlatSpec with Matchers {
  implicit val p = new WithAccelConfig ++ new WithTestConfig

  it should "test vectorized SelectNodeWithVectorization" in {
    chisel3.iotesters.Driver.execute(
      Array("--target-dir", "generated_dut/",
        "--generate-vcd-output", "on",
        "-X", "verilog"),
      () => new SelectNodeWithVectorization(
        NumOuts = Seq(1, 1, 1, 1),
        NumLanes = 4,
        ID = 0
      )
    ) { c => new VectorSelectTester(c) } should be(true)
  }
}
