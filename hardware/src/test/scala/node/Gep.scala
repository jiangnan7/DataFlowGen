package heteacc.node

import chisel3._
import chiseltest._
import chipsalliance.rocketchip.config._
import heteacc.config._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GepNodeTests extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  implicit val p: Parameters = new WithAccelConfig ++ new WithTestConfig

  behavior of "Gep nodes"

  it should "compute scalar addresses" in {
    test(new GepNode(NumIns = 1, NumOuts = 1, ID = 0)(ElementSize = 4, ArraySize = List())) { c =>
      c.io.enable.valid.poke(false.B)
      c.io.enable.bits.control.poke(false.B)
      c.io.enable.bits.taskID.poke(0.U)
      c.io.enable.bits.debug.poke(false.B)
      c.io.baseAddress.valid.poke(false.B)
      c.io.idx(0).valid.poke(false.B)
      c.io.Out(0).ready.poke(true.B)

      c.clock.step()
      c.io.baseAddress.ready.expect(true.B)
      c.io.idx(0).ready.expect(true.B)

      c.io.enable.valid.poke(true.B)
      c.io.enable.bits.control.poke(true.B)
      c.io.enable.bits.taskID.poke(5.U)
      c.io.baseAddress.valid.poke(true.B)
      c.io.baseAddress.bits.data.poke("h1000".U)
      c.io.baseAddress.bits.predicate.poke(true.B)
      c.io.baseAddress.bits.taskID.poke(13.U)
      c.io.idx(0).valid.poke(true.B)
      c.io.idx(0).bits.data.poke(3.U)
      c.io.idx(0).bits.predicate.poke(true.B)
      c.io.idx(0).bits.taskID.poke(0.U)

      c.clock.step()
      c.io.enable.valid.poke(false.B)
      c.io.baseAddress.valid.poke(false.B)
      c.io.idx(0).valid.poke(false.B)

      c.clock.step()
      c.io.Out(0).valid.expect(true.B)
      c.io.Out(0).bits.data.expect("h100c".U)
      c.io.Out(0).bits.taskID.expect(13.U)
      c.io.Out(0).bits.predicate.expect(true.B)

      c.clock.step()
      c.io.Out(0).valid.expect(false.B)
    }
  }

  it should "compute vector field addresses" in {
    test(new GepNodeWithVectorization(NumIns = 1, NumOuts = Seq(1, 1, 1, 1), NumLanes = 4, ID = 0)(
      ElementSize = 1,
      ArraySize = List()
    )) { c =>
      c.io.Out.elements.values.foreach(_.foreach(_.ready.poke(true.B)))
      c.io.baseAddress.valid.poke(false.B)
      c.io.idx(0).valid.poke(false.B)

      c.clock.step()

      c.io.baseAddress.valid.poke(true.B)
      c.io.baseAddress.bits.data.poke("h1000".U)
      c.io.baseAddress.bits.predicate.poke(true.B)
      c.io.baseAddress.bits.taskID.poke(0.U)
      c.io.idx(0).valid.poke(true.B)
      c.io.idx(0).bits.data.poke("h10".U)
      c.io.idx(0).bits.predicate.poke(true.B)
      c.io.idx(0).bits.taskID.poke(0.U)

      c.clock.step()
      c.io.baseAddress.valid.poke(false.B)
      c.io.idx(0).valid.poke(false.B)

      for (fieldIdx <- 0 until c.io.Out.elements.size) {
        val out = c.io.Out.elements(s"field$fieldIdx")(0)
        out.valid.expect(true.B)
        out.bits.data.expect((0x1010 + fieldIdx).U)
        out.bits.taskID.expect(1.U)
        out.bits.predicate.expect(true.B)
      }

      c.clock.step()
    }
  }
}
