package heteacc.node

import chisel3._
import chiseltest._
import chipsalliance.rocketchip.config._
import heteacc.config._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LSNodeTests extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  implicit val p: Parameters = new WithAccelConfig ++ new WithTestConfig

  behavior of "LS nodes"

  it should "forward load address and returned data" in {
    test(new Load(NumOuts = 1, ID = 1, RouteID = 0)) { c =>
      c.GepAddr.valid.poke(false.B)
      c.GepAddr.bits.data.poke(0.U)
      c.GepAddr.bits.predicate.poke(false.B)
      c.GepAddr.bits.taskID.poke(0.U)
      c.address_out.ready.poke(true.B)
      c.data_in.valid.poke(false.B)
      c.data_in.bits.data.poke(0.U)
      c.data_in.bits.predicate.poke(false.B)
      c.data_in.bits.taskID.poke(0.U)
      c.io.Out(0).ready.poke(true.B)

      c.clock.step()
      c.GepAddr.ready.expect(true.B)
      c.data_in.ready.expect(true.B)

      c.GepAddr.valid.poke(true.B)
      c.GepAddr.bits.data.poke(0x20.U)
      c.GepAddr.bits.predicate.poke(true.B)
      c.GepAddr.bits.taskID.poke(2.U)
      c.data_in.valid.poke(true.B)
      c.data_in.bits.data.poke(0xAB.U)
      c.data_in.bits.predicate.poke(true.B)
      c.data_in.bits.taskID.poke(2.U)

      c.address_out.valid.expect(true.B)
      c.address_out.bits.data.expect(0x20.U)
      c.io.Out(0).valid.expect(true.B)
      c.io.Out(0).bits.data.expect(0xAB.U)
      c.io.Out(0).bits.predicate.expect(true.B)
      c.io.Out(0).bits.taskID.expect(2.U)

      c.clock.step()
    }
  }

  it should "forward store address and payload together" in {
    test(new Store(NumOuts = 1, ID = 1, RouteID = 0)) { c =>
      c.GepAddr.valid.poke(false.B)
      c.GepAddr.bits.data.poke(0.U)
      c.GepAddr.bits.predicate.poke(false.B)
      c.GepAddr.bits.taskID.poke(0.U)
      c.inData.valid.poke(false.B)
      c.inData.bits.data.poke(0.U)
      c.inData.bits.predicate.poke(false.B)
      c.inData.bits.taskID.poke(0.U)
      c.address_out.ready.poke(true.B)
      c.io.Out(0).ready.poke(true.B)

      c.clock.step()

      c.GepAddr.valid.poke(true.B)
      c.GepAddr.bits.data.poke(0x34.U)
      c.GepAddr.bits.predicate.poke(true.B)
      c.GepAddr.bits.taskID.poke(4.U)
      c.inData.valid.poke(true.B)
      c.inData.bits.data.poke(0xCD.U)
      c.inData.bits.predicate.poke(true.B)
      c.inData.bits.taskID.poke(4.U)

      c.address_out.valid.expect(true.B)
      c.address_out.bits.data.expect(0x34.U)
      c.io.Out(0).valid.expect(true.B)
      c.io.Out(0).bits.data.expect(0xCD.U)
      c.io.Out(0).bits.predicate.expect(true.B)
      c.io.Out(0).bits.taskID.expect(4.U)

      c.clock.step()
    }
  }
}
