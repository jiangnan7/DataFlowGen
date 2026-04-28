package heteacc.node

import chisel3._
import chiseltest._
import chipsalliance.rocketchip.config._
import heteacc.config._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ComputeNodeTests extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  implicit val p: Parameters = new WithAccelConfig ++ new WithTestConfig

  behavior of "Compute nodes"

  it should "compute scalar add results" in {
    test(new ComputeNode(NumOuts = 1, ID = 0, opCode = "Add")(sign = false, Debug = false)) { c =>
      c.io.enable.valid.poke(false.B)
      c.io.enable.bits.control.poke(false.B)
      c.io.enable.bits.taskID.poke(0.U)
      c.io.enable.bits.debug.poke(false.B)
      c.io.LeftIO.valid.poke(false.B)
      c.io.RightIO.valid.poke(false.B)
      c.io.Out(0).ready.poke(false.B)

      c.clock.step()
      c.io.LeftIO.ready.expect(true.B)
      c.io.RightIO.ready.expect(true.B)

      c.io.enable.valid.poke(true.B)
      c.io.enable.bits.control.poke(true.B)
      c.io.enable.bits.taskID.poke(3.U)
      c.io.LeftIO.valid.poke(true.B)
      c.io.LeftIO.bits.data.poke(10.U)
      c.io.LeftIO.bits.predicate.poke(true.B)
      c.io.LeftIO.bits.taskID.poke(0.U)
      c.io.RightIO.valid.poke(true.B)
      c.io.RightIO.bits.data.poke(32.U)
      c.io.RightIO.bits.predicate.poke(true.B)
      c.io.RightIO.bits.taskID.poke(0.U)

      c.clock.step()
      c.io.enable.valid.poke(false.B)
      c.io.LeftIO.valid.poke(false.B)
      c.io.RightIO.valid.poke(false.B)

      c.io.Out(0).valid.expect(true.B)
      c.clock.step()
      c.io.Out(0).valid.expect(true.B)
      c.io.Out(0).bits.data.expect(42.U)
      c.io.Out(0).bits.taskID.expect(3.U)
      c.io.Out(0).bits.predicate.expect(true.B)

      c.io.Out(0).ready.poke(true.B)
      c.clock.step()
      c.io.Out(0).valid.expect(false.B)
    }
  }

  it should "compute vector add results lane by lane" in {
    test(new ComputeNodeWithVectorization(NumOuts = Seq(1, 1, 1, 1), NumLanes = 4, ID = 0, opCode = "Add")(
      sign = false,
      Debug = false
    )) { c =>
      for (lane <- c.io.LeftIO.indices) {
        c.io.LeftIO(lane).valid.poke(false.B)
        c.io.RightIO(lane).valid.poke(false.B)
      }
      c.io.Out.elements.values.foreach(_.foreach(_.ready.poke(true.B)))

      c.clock.step()

      for (lane <- c.io.LeftIO.indices) {
        c.io.LeftIO(lane).valid.poke(true.B)
        c.io.LeftIO(lane).bits.data.poke((lane + 1).U)
        c.io.LeftIO(lane).bits.predicate.poke(true.B)
        c.io.LeftIO(lane).bits.taskID.poke(0.U)

        c.io.RightIO(lane).valid.poke(true.B)
        c.io.RightIO(lane).bits.data.poke(((lane + 1) * 2).U)
        c.io.RightIO(lane).bits.predicate.poke(true.B)
        c.io.RightIO(lane).bits.taskID.poke(0.U)
      }

      c.clock.step()
      for (lane <- c.io.LeftIO.indices) {
        c.io.LeftIO(lane).valid.poke(false.B)
        c.io.RightIO(lane).valid.poke(false.B)
      }

      for (lane <- c.io.LeftIO.indices) {
        val out = c.io.Out.elements(s"field$lane")(0)
        out.valid.expect(true.B)
        out.bits.data.expect(((lane + 1) * 3).U)
        out.bits.taskID.expect(0.U)
        out.bits.predicate.expect(true.B)
      }

      c.clock.step()
    }
  }
}
