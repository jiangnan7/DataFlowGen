package heteacc.generator

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester, OrderedDecoupledHWIOTester}
import org.scalatest.{Matchers, FlatSpec}
import heteacc.generator.doitgenTripleDF
import chipsalliance.rocketchip.config._
import heteacc.config._
import utility._
import heteacc.interfaces.NastiMemSlave
import heteacc.interfaces._
import heteacc.accel._
import heteacc.acctest._
import heteacc.memory._


class doitgenTriple_main(implicit p: Parameters) extends AccelIO(List(32, 32, 32), List())(p) {

  val cache = Module(new Cache) // Simple Nasti Cache
  val memModel = Module(new NastiMemSlave) // Model of DRAM to connect to Cache

  // // Connect the wrapper I/O to the memory model initialization interface so the
  // // test bench can write contents at start.
  memModel.io.nasti <> cache.io.nasti
  memModel.io.init.bits.addr := 0.U
  memModel.io.init.bits.data := 0.U
  memModel.io.init.valid := false.B
  cache.io.cpu.abort := false.B



  val test13 = Module(new doitgenTripleDF())

  //Put an arbiter infront of cache
  val CacheArbiter = Module(new MemArbiter(1))


  //Connect main module to cache arbiter
  CacheArbiter.io.cpu.MemReq(0) <> io.req
  io.resp <> CacheArbiter.io.cpu.MemResp(0)

  //Connect cache to the arbiter
  cache.io.cpu.req <> CacheArbiter.io.cache.MemReq
  CacheArbiter.io.cache.MemResp <> cache.io.cpu.resp

  //Connect in/out ports
  test13.io.in <> io.in
  io.out <> test13.io.out

  // Check if trace option is on or off
  if (log == false) {
    println(Console.RED + "****** Trace option is off. *********" + Console.RESET)
  }
  else
    println(Console.BLUE + "****** Trace option is on. *********" + Console.RESET)


}


class doitgenTripleDF01[T <: AccelIO](c: T)
                                      (inAddrVec: List[Int], inDataVec: List[Int],
                                       outAddrVec: List[Int], outDataVec: List[Int])
  extends AccelTesterLocal(c)(inAddrVec, inDataVec, outAddrVec, outDataVec) {

  // initMemory()


  poke(c.io.in.valid, false)
  poke(c.io.in.bits.data("field0").data, 1.U)
  // poke(c.io.in.bits.data("field0").taskID, 0.U)
  poke(c.io.in.bits.data("field0").predicate, false.B)
  poke(c.io.out.ready, false.B)
  poke(c.io.in.bits.data("field1").data, 1.U)//store
  // poke(c.io.in.bits.data("field1").taskID, 0)
  poke(c.io.in.bits.data("field1").predicate, false)

  poke(c.io.in.bits.data("field2").data, 17.U)
  // poke(c.io.in.bits.data("field1").taskID, 0)
  poke(c.io.in.bits.data("field2").predicate, false)


  step(1)
  poke(c.io.in.bits.enable.control, true)
  poke(c.io.in.valid, true)
  poke(c.io.in.bits.data("field0").data, 0.U) // Array a[] base address
  poke(c.io.in.bits.data("field0").predicate, true)
  poke(c.io.out.ready, true.B)
  poke(c.io.in.bits.data("field1").data, 1.U) // Array b[] base address
  poke(c.io.in.bits.data("field1").predicate, true)
  
  poke(c.io.in.bits.data("field2").data, 17.U) // Array b[] base address
  poke(c.io.in.bits.data("field2").predicate, true)




  var time = 0 //Cycle counter
  var result = false
  while (time < 18000 && !result) {
    time += 1
    step(1)

    if (peek(c.io.out.valid) == 1) {
      result = true
      println(Console.BLUE + s"*** BdoitgenTriple finished. Run time: $time cycles." + Console.RESET)
    }
  }


  if (!result) {
    println(Console.RED + "*** Timeout." + Console.RESET)
    fail
  }
}

class doitgenTripleDF_test extends FlatSpec with Matchers {

  val inDataVec = List()


  val inAddrVec = List.range(0, 32 * inDataVec.length, 32) //32

  // val outAddrVec = List.range(32 * inDataVec.length, 32 * inDataVec.length + (32 * 1), 32)
  val outAddrVec = List.range(32 * inDataVec.length, 32 * inDataVec.length + (32 * 1),32)
  val outDataVec = List()

  implicit val p = new WithAccelConfig(HeteaccAccelParams())
  // iotester flags:
  // -ll  = log level <Error|Warn|Info|Debug|Trace>
  // -tbn = backend <firrtl|verilator|vcs>
  // -td  = target directory
  // -tts = seed for RNG
  it should s"Test: direct connection" in {
    chisel3.iotesters.Driver.execute(
      Array(
        // "-ll", "Info",
        "-tn", "doitgenTriple",
        "-tbn", "verilator",
        "-td", s"test_run_dir/doitgenTriple",
        "-tts", "0001",
        "--generate-vcd-output", "on"),
        
      () => new doitgenTriple_main()(p)) {
      c => new doitgenTripleDF01(c)(inAddrVec, inDataVec, outAddrVec, outDataVec)
    } should be(true)
  }
}