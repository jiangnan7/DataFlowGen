package matrix

import chisel3._
import org.scalatest.{FlatSpec, Matchers}
import heteacc.config._
import chipsalliance.rocketchip.config._

class Systolic32x32Tester extends FlatSpec with Matchers {

  implicit val p = new WithAccelConfig(HeteaccAccelParams())

  it should "run systolic 32x32 compute test" in {
    chisel3.iotesters.Driver.execute(Array(
      "-tn", "matrix32x32",
      "-tbn", "verilator",
      "-td", s"test_run_dir/matrix32x32",
      "-tts", "0001",
      "--generate-vcd-output", "on"),
      () => new SystolicSquareBuffered(UInt(32.W), 32)) {
      c => new SystolicBaseTests(c)
    } should be(true)
  }
}
