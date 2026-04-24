package heteacc.node

import chisel3.iotesters.PeekPokeTester
import chipsalliance.rocketchip.config._
import heteacc.config._
import org.scalatest.{FlatSpec, Matchers}

class VectorBroadcastTester(df: BroadcastNodeWithVectorization)
                           (implicit p: Parameters) extends PeekPokeTester(df) {

  private def printInputs(): Unit = {
    println(f"[TEST] Input valid: ${peek(df.io.Input.valid)}")
    println(f"[TEST] Input data: 0x${peek(df.io.Input.bits.data)}%X")
    println(f"[TEST] Input ready: ${peek(df.io.Input.ready)}")
  }

  private def printOutputs(): Unit = {
    for (i <- 0 until df.io.Out.length) {
      println(f"[TEST] Output[$i] valid: ${peek(df.io.Out(i).valid)}")
      println(f"[TEST] Output[$i] data: 0x${peek(df.io.Out(i).bits.data)}%X")
      println(f"[TEST] Output[$i] ready: ${peek(df.io.Out(i).ready)}")
    }
  }

  for (i <- 0 until df.io.Out.length) {
    poke(df.io.Out(i).ready, 1)
  }
  poke(df.io.Input.valid, 0)

  println("=== Initial State ===")
  printInputs()
  printOutputs()

  poke(df.io.Input.valid, 1)
  poke(df.io.Input.bits.data, 0xDEADBEEF)

  println("=== After Providing Input ===")
  printInputs()
  printOutputs()

  step(1)
  println("=== Outputs After One Cycle ===")
  printOutputs()

  poke(df.io.Input.bits.data, 0xCAFEBABE)
  step(1)
  println("=== After Sending New Data ===")
  printOutputs()
}

class VectorBroadcastTests extends FlatSpec with Matchers {
  implicit val p = new WithAccelConfig ++ new WithTestConfig

  it should "test vectorized BroadcastNodeWithVectorization" in {
    chisel3.iotesters.Driver.execute(
      Array("--target-dir", "generated_dut/",
        "--generate-vcd-output", "on",
        "-X", "verilog"),
      () => new BroadcastNodeWithVectorization(
        NumOuts = 1,
        NumLanes = 4,
        ID = 0
      )(sign = false, Debug = false)
    ) { c => new VectorBroadcastTester(c) } should be(true)
  }
}
