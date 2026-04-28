package heteacc.node

import chisel3._
import chiseltest._
import chipsalliance.rocketchip.config._
import heteacc.config._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ConstNodeTests extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  implicit val p: Parameters = new WithAccelConfig ++ new WithTestConfig

  behavior of "Const nodes"

  it should "emit the scalar constant with enable metadata" in {
    test(new ConstFastNode(value = 42, ID = 0)) { c =>
      c.io.enable.valid.poke(false.B)
      c.io.enable.bits.control.poke(false.B)
      c.io.enable.bits.taskID.poke(0.U)
      c.io.enable.bits.debug.poke(false.B)
      c.io.Out.ready.poke(false.B)

      c.clock.step()
      c.io.enable.ready.expect(true.B)
      c.io.Out.valid.expect(false.B)

      c.io.enable.valid.poke(true.B)
      c.io.enable.bits.control.poke(true.B)
      c.io.enable.bits.taskID.poke(7.U)
      c.io.Out.ready.poke(true.B)

      c.clock.step()
      c.io.Out.valid.expect(true.B)
      c.io.Out.bits.data.expect(42.U)
      c.io.Out.bits.taskID.expect(7.U)
      c.io.Out.bits.predicate.expect(true.B)

      c.io.enable.valid.poke(false.B)
      c.clock.step()
      c.io.Out.valid.expect(false.B)
    }
  }

  it should "broadcast the vector constant to all lanes" in {
    test(new ConstFastNodeWithVectorization(value = 42, NumLanes = 4, ID = 0)) { c =>
      c.io.enable.valid.poke(false.B)
      c.io.enable.bits.control.poke(false.B)
      c.io.enable.bits.taskID.poke(0.U)
      c.io.enable.bits.debug.poke(false.B)
      c.io.Out.foreach(_.ready.poke(false.B))

      c.clock.step()
      c.io.enable.ready.expect(true.B)
      c.io.Out.foreach(_.valid.expect(false.B))

      c.io.enable.valid.poke(true.B)
      c.io.enable.bits.control.poke(true.B)
      c.io.enable.bits.taskID.poke(9.U)
      c.io.Out.foreach(_.ready.poke(true.B))

      c.clock.step()
      c.io.Out.foreach { out =>
        out.valid.expect(true.B)
        out.bits.data.expect(42.U)
        out.bits.taskID.expect(9.U)
        out.bits.predicate.expect(true.B)
      }

      c.io.enable.valid.poke(false.B)
      c.clock.step()
      c.io.Out.foreach(_.valid.expect(false.B))
    }
  }
}
