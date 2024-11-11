package heteacc.node

import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester, OrderedDecoupledHWIOTester}
import org.scalatest.{Matchers, FlatSpec}

import chipsalliance.rocketchip.config._
import heteacc.config._
import utility._

class TypLoadTests(c: UnTypLoadCache) extends PeekPokeTester(c) {
  poke(c.io.GepAddr.valid, false)
  poke(c.io.enable.valid, false)
  poke(c.io.MemReq.ready, false)
  poke(c.io.MemResp.valid, false)
  poke(c.io.Out(0).ready, true)

  var requestCycle = -1
  var responseCycle = -1

  for (t <- 0 until 20) {

    step(1)

    //IF ready is set
    // send address
    if (peek(c.io.GepAddr.ready) == 1) {
      poke(c.io.GepAddr.valid, true)
      poke(c.io.GepAddr.bits.data, 12)
      poke(c.io.GepAddr.bits.predicate, true)
      poke(c.io.enable.bits.control, true)
      poke(c.io.enable.valid, true)
      if (requestCycle == -1) {
        requestCycle = t
      }
    }

    printf(s"t: ${t}  c.io.memReq: ${peek(c.io.MemReq)} \n")
    if ((peek(c.io.MemReq.valid) == 1) && (t > 4)) {
      poke(c.io.MemReq.ready, true)
    }

    if (t > 8) {
      poke(c.io.MemResp.valid, true)
      poke(c.io.MemResp.bits.data, 0 + t)
      if (responseCycle == -1) {
        responseCycle = t
      }
    }
  }

  if (requestCycle != -1 && responseCycle != -1) {
    val loadCycles = responseCycle - requestCycle
    println(s"Load operation took $loadCycles cycles")
  } else {
    println("Load operation did not complete within the test cycles")
  }
}


import Constants._

class TypLoadTester extends FlatSpec with Matchers {
  implicit val p = new WithAccelConfig
  it should "Load Node tester" in {
    chisel3.iotesters.Driver.execute(
      Array(
        // "-ll", "Info",
        "-tn", "test03",
        "-tbn", "verilator",
        "-td", "test_run_dir/test03",
        "-tts", "0001"),
      () => new UnTypLoadCache(NumPredOps = 0, NumSuccOps = 0, NumOuts = 1, ID = 1, RouteID = 0)) { c =>
      new TypLoadTests(c)
    } should be(true)
  }
}
