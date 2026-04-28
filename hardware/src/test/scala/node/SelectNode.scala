package heteacc.node

import chisel3._
import chiseltest._
import chipsalliance.rocketchip.config._
import heteacc.config._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SelectNodeTests extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  implicit val p: Parameters = new WithAccelConfig ++ new WithTestConfig

  behavior of "Select nodes"

  it should "select the scalar true branch" in {
    test(new SelectNode(NumOuts = 1, ID = 0)) { c =>
      c.io.enable.valid.poke(false.B)
      c.io.enable.bits.control.poke(false.B)
      c.io.enable.bits.taskID.poke(0.U)
      c.io.enable.bits.debug.poke(false.B)
      c.io.InData1.valid.poke(false.B)
      c.io.InData2.valid.poke(false.B)
      c.io.Select.valid.poke(false.B)
      c.io.Out(0).ready.poke(true.B)

      c.clock.step()

      c.io.enable.valid.poke(true.B)
      c.io.enable.bits.control.poke(true.B)
      c.io.enable.bits.taskID.poke(21.U)
      c.io.InData1.valid.poke(true.B)
      c.io.InData1.bits.data.poke("h11".U)
      c.io.InData1.bits.predicate.poke(true.B)
      c.io.InData1.bits.taskID.poke(0.U)
      c.io.InData2.valid.poke(true.B)
      c.io.InData2.bits.data.poke("h22".U)
      c.io.InData2.bits.predicate.poke(true.B)
      c.io.InData2.bits.taskID.poke(0.U)
      c.io.Select.valid.poke(true.B)
      c.io.Select.bits.data.poke(1.U)
      c.io.Select.bits.predicate.poke(true.B)
      c.io.Select.bits.taskID.poke(0.U)

      c.clock.step()
      c.io.enable.valid.poke(false.B)
      c.io.InData1.valid.poke(false.B)
      c.io.InData2.valid.poke(false.B)
      c.io.Select.valid.poke(false.B)

      c.io.Out(0).valid.expect(true.B)
      c.io.Out(0).bits.data.expect("h11".U)
      c.io.Out(0).bits.taskID.expect(21.U)
      c.io.Out(0).bits.predicate.expect(true.B)

      c.clock.step()
      c.io.Out(0).valid.expect(false.B)
    }
  }

  it should "select vector lane outputs independently" in {
    test(new SelectNodeWithVectorization(NumOuts = Seq(1, 1, 1, 1), NumLanes = 4, ID = 0)) { c =>
      c.io.Out.elements.values.foreach(_.foreach(_.ready.poke(true.B)))

      for (lane <- c.io.InData1.indices) {
        c.io.InData1(lane).valid.poke(false.B)
        c.io.InData2(lane).valid.poke(false.B)
        c.io.Select(lane).valid.poke(false.B)
      }

      c.clock.step()

      for (lane <- c.io.InData1.indices) {
        c.io.InData1(lane).valid.poke(true.B)
        c.io.InData1(lane).bits.data.poke((0x10 + lane).U)
        c.io.InData1(lane).bits.predicate.poke(true.B)
        c.io.InData1(lane).bits.taskID.poke(0.U)

        c.io.InData2(lane).valid.poke(true.B)
        c.io.InData2(lane).bits.data.poke((0x20 + lane).U)
        c.io.InData2(lane).bits.predicate.poke(true.B)
        c.io.InData2(lane).bits.taskID.poke(0.U)

        c.io.Select(lane).valid.poke(true.B)
        c.io.Select(lane).bits.data.poke((lane % 2).U)
        c.io.Select(lane).bits.predicate.poke(true.B)
        c.io.Select(lane).bits.taskID.poke(0.U)
      }

      c.clock.step()
      for (lane <- c.io.InData1.indices) {
        c.io.InData1(lane).valid.poke(false.B)
        c.io.InData2(lane).valid.poke(false.B)
        c.io.Select(lane).valid.poke(false.B)
      }

      for (lane <- c.io.InData1.indices) {
        val expected = if (lane % 2 == 0) 0x20 + lane else 0x10 + lane
        val out = c.io.Out.elements(s"field$lane")(0)
        out.valid.expect(true.B)
        out.bits.data.expect(expected.U)
        out.bits.taskID.expect(0.U)
        out.bits.predicate.expect(true.B)
      }

      c.clock.step()
    }
  }
}
