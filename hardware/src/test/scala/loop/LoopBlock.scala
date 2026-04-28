package heteacc.loop

import chisel3._
import chiseltest._
import chipsalliance.rocketchip.config._
import heteacc.config._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LoopBlockTests extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  implicit val p: Parameters = new WithAccelConfig ++ new WithTestConfig

  behavior of "LoopBlockNodeExperimental"

  it should "restart once and then exit with the last live-out value" in {
    test(new LoopBlockNodeExperimental(
      ID = 0,
      NumIns = List(1),
      NumCarry = List(1),
      NumOuts = List(1),
      NumExits = 1,
      LoopCounterMax = 1,
      LoopCounterStep = 1
    )) { c =>
      c.io.enable.valid.poke(false.B)
      c.io.enable.bits.control.poke(false.B)
      c.io.enable.bits.taskID.poke(0.U)
      c.io.enable.bits.debug.poke(false.B)
      c.io.InLiveIn(0).valid.poke(false.B)
      c.io.InLiveIn(0).bits.data.poke(0.U)
      c.io.InLiveIn(0).bits.predicate.poke(false.B)
      c.io.InLiveIn(0).bits.taskID.poke(0.U)
      c.io.InLiveOut(0).valid.poke(false.B)
      c.io.InLiveOut(0).bits.data.poke(0.U)
      c.io.InLiveOut(0).bits.predicate.poke(false.B)
      c.io.InLiveOut(0).bits.taskID.poke(0.U)
      c.io.CarryDepenIn(0).valid.poke(false.B)
      c.io.CarryDepenIn(0).bits.data.poke(0.U)
      c.io.CarryDepenIn(0).bits.predicate.poke(false.B)
      c.io.CarryDepenIn(0).bits.taskID.poke(0.U)

      c.io.activate_loop_start.ready.poke(false.B)
      c.io.activate_loop_back.ready.poke(false.B)
      c.io.OutLiveIn.elements("field0")(0).ready.poke(false.B)
      c.io.CarryDepenOut.elements("field0")(0).ready.poke(false.B)
      c.io.OutLiveOut.elements("field0")(0).ready.poke(false.B)
      c.io.loopExit(0).ready.poke(false.B)

      c.clock.step()

      c.io.enable.valid.poke(true.B)
      c.io.enable.bits.control.poke(true.B)
      c.io.enable.bits.taskID.poke(5.U)
      c.io.InLiveIn(0).valid.poke(true.B)
      c.io.InLiveIn(0).bits.data.poke(7.U)
      c.io.InLiveIn(0).bits.predicate.poke(true.B)
      c.io.InLiveIn(0).bits.taskID.poke(5.U)

      c.clock.step()
      c.io.enable.valid.poke(false.B)
      c.io.InLiveIn(0).valid.poke(false.B)

      c.clock.step()
      c.io.activate_loop_start.valid.expect(true.B)
      c.io.activate_loop_start.bits.control.expect(true.B)
      c.io.activate_loop_start.bits.taskID.expect(5.U)
      c.io.activate_loop_back.valid.expect(true.B)
      c.io.activate_loop_back.bits.control.expect(false.B)
      c.io.OutLiveIn.elements("field0")(0).valid.expect(true.B)
      c.io.OutLiveIn.elements("field0")(0).bits.data.expect(7.U)
      c.io.CarryDepenOut.elements("field0")(0).valid.expect(true.B)
      c.io.CarryDepenOut.elements("field0")(0).bits.data.expect(0.U)

      c.io.activate_loop_start.ready.poke(true.B)
      c.io.activate_loop_back.ready.poke(true.B)
      c.io.OutLiveIn.elements("field0")(0).ready.poke(true.B)
      c.io.CarryDepenOut.elements("field0")(0).ready.poke(true.B)
      c.io.InLiveOut(0).valid.poke(true.B)
      c.io.InLiveOut(0).bits.data.poke(99.U)
      c.io.InLiveOut(0).bits.predicate.poke(true.B)
      c.io.InLiveOut(0).bits.taskID.poke(5.U)
      c.io.CarryDepenIn(0).valid.poke(true.B)
      c.io.CarryDepenIn(0).bits.data.poke(5.U)
      c.io.CarryDepenIn(0).bits.predicate.poke(true.B)
      c.io.CarryDepenIn(0).bits.taskID.poke(5.U)

      c.clock.step()
      c.io.InLiveOut(0).valid.poke(false.B)
      c.io.CarryDepenIn(0).valid.poke(false.B)
      c.io.activate_loop_start.ready.poke(false.B)
      c.io.activate_loop_back.ready.poke(false.B)
      c.io.OutLiveIn.elements("field0")(0).ready.poke(false.B)
      c.io.CarryDepenOut.elements("field0")(0).ready.poke(false.B)

      c.clock.step()
      c.io.activate_loop_start.valid.expect(true.B)
      c.io.activate_loop_start.bits.control.expect(false.B)
      c.io.activate_loop_back.valid.expect(true.B)
      c.io.activate_loop_back.bits.control.expect(true.B)
      c.io.OutLiveIn.elements("field0")(0).valid.expect(true.B)
      c.io.OutLiveIn.elements("field0")(0).bits.data.expect(7.U)
      c.io.CarryDepenOut.elements("field0")(0).valid.expect(true.B)
      c.io.CarryDepenOut.elements("field0")(0).bits.data.expect(5.U)

      c.io.activate_loop_start.ready.poke(true.B)
      c.io.activate_loop_back.ready.poke(true.B)
      c.io.OutLiveIn.elements("field0")(0).ready.poke(true.B)
      c.io.CarryDepenOut.elements("field0")(0).ready.poke(true.B)
      c.io.InLiveOut(0).valid.poke(true.B)
      c.io.InLiveOut(0).bits.data.poke(123.U)
      c.io.InLiveOut(0).bits.predicate.poke(true.B)
      c.io.InLiveOut(0).bits.taskID.poke(5.U)
      c.io.CarryDepenIn(0).valid.poke(true.B)
      c.io.CarryDepenIn(0).bits.data.poke(8.U)
      c.io.CarryDepenIn(0).bits.predicate.poke(true.B)
      c.io.CarryDepenIn(0).bits.taskID.poke(5.U)

      c.clock.step()
      c.io.InLiveOut(0).valid.poke(false.B)
      c.io.CarryDepenIn(0).valid.poke(false.B)

      c.clock.step()
      c.io.OutLiveOut.elements("field0")(0).valid.expect(true.B)
      c.io.OutLiveOut.elements("field0")(0).bits.data.expect(123.U)
      c.io.loopExit(0).valid.expect(true.B)
      c.io.loopExit(0).bits.control.expect(true.B)

      c.io.OutLiveOut.elements("field0")(0).ready.poke(true.B)
      c.io.loopExit(0).ready.poke(true.B)
      c.clock.step()
      c.io.OutLiveOut.elements("field0")(0).valid.expect(false.B)
      c.io.loopExit(0).valid.expect(false.B)
    }
  }
}
