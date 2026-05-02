package heteacc.node

import chisel3._
import chiseltest._
import chipsalliance.rocketchip.config._
import heteacc.config._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BitcastNodeTests extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  implicit val p: Parameters = new WithAccelConfig ++ new WithTestConfig

  behavior of "Bitcast nodes"

  it should "forward scalar input data" in {
    test(new BitCastNode(NumOuts = 1, ID = 0, Debug = false)) { c =>
      c.io.enable.valid.poke(false.B)
      c.io.enable.bits.control.poke(false.B)
      c.io.enable.bits.taskID.poke(0.U)
      c.io.enable.bits.debug.poke(false.B)
      c.io.Input.valid.poke(false.B)
      c.io.Out(0).ready.poke(true.B)

      c.clock.step()
      c.io.Input.ready.expect(true.B)

      c.io.Input.valid.poke(true.B)
      c.io.Input.bits.data.poke("h1234".U)
      c.io.Input.bits.predicate.poke(true.B)
      c.io.Input.bits.taskID.poke(0.U)
      c.io.enable.valid.poke(true.B)
      c.io.enable.bits.control.poke(true.B)
      c.io.enable.bits.taskID.poke(11.U)

      c.clock.step()
      c.io.Input.valid.poke(false.B)
      c.io.enable.valid.poke(false.B)

      c.io.Out(0).valid.expect(true.B)
      c.io.Out(0).bits.data.expect("h1234".U)
      c.io.Out(0).bits.taskID.expect(11.U)
      c.io.Out(0).bits.predicate.expect(true.B)

      c.clock.step()
      c.io.Out(0).valid.expect(false.B)
    }
  }

  it should "forward vector inputs lane by lane" in {
    test(new BitcastNodeWithVectorization(NumOuts = Seq(1, 1, 1, 1), NumLanes = 4, ID = 0)) { c =>
      for (lane <- c.io.Input.indices) {
        c.io.Input(lane).valid.poke(false.B)
      }
      c.io.Out.elements.values.foreach(_.foreach(_.ready.poke(true.B)))

      c.clock.step()

      for (lane <- c.io.Input.indices) {
        c.io.Input(lane).valid.poke(true.B)
        c.io.Input(lane).bits.data.poke((0x40 + lane).U)
        c.io.Input(lane).bits.predicate.poke(true.B)
        c.io.Input(lane).bits.taskID.poke(0.U)
      }

      c.clock.step()
      for (lane <- c.io.Input.indices) {
        c.io.Input(lane).valid.poke(false.B)
      }

      for (lane <- c.io.Input.indices) {
        val out = c.io.Out.elements(s"field$lane")(0)
        out.valid.expect(true.B)
        out.bits.data.expect((0x40 + lane).U)
        out.bits.taskID.expect(1.U)
        out.bits.predicate.expect(true.B)
      }

      c.clock.step()
    }
  }
}
