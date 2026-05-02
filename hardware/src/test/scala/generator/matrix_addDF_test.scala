package heteacc.generator

import chisel3._
import chipsalliance.rocketchip.config._
import chisel3._
import chisel3.iotesters.Driver
import org.scalatest.{FlatSpec, Matchers}
import heteacc.config._
import heteacc.accel._
import heteacc.acctest._
import heteacc.interfaces._
import heteacc.interfaces.NastiMemSlave
import heteacc.memory._

class matrix_add_main(implicit p: Parameters) extends AccelIO(List(32, 32, 32, 32), List(32))(p) {

  val cache = Module(new Cache)
  val memModel = Module(new NastiMemSlave)

  memModel.io.nasti <> cache.io.nasti
  memModel.io.init.bits.addr := 0.U
  memModel.io.init.bits.data := 0.U
  memModel.io.init.valid := false.B
  cache.io.cpu.abort := false.B

  val test13 = Module(new matrix_addDF())
  val CacheArbiter = Module(new MemArbiter(1))

  CacheArbiter.io.cpu.MemReq(0) <> io.req
  io.resp <> CacheArbiter.io.cpu.MemResp(0)

  cache.io.cpu.req <> CacheArbiter.io.cache.MemReq
  CacheArbiter.io.cache.MemResp <> cache.io.cpu.resp

  test13.io.in <> io.in
  io.out <> test13.io.out
}

class matrix_add01[T <: AccelIO](c: T)(
  inAddrVec: List[Int],
  inDataVec: List[Int],
  outAddrVec: List[Int],
  outDataVec: List[Int]
)
  extends AccelTesterLocal(c)(inAddrVec, inDataVec, outAddrVec, outDataVec) {

  poke(c.io.in.valid, false)
  poke(c.io.in.bits.data("field0").data, 0.U)
  poke(c.io.in.bits.data("field0").predicate, false.B)
  poke(c.io.out.ready, false.B)
  poke(c.io.in.bits.data("field1").data, 50.U)
  poke(c.io.in.bits.data("field1").predicate, false.B)
  poke(c.io.in.bits.data("field2").data, 100.U)
  poke(c.io.in.bits.data("field2").predicate, false.B)
  poke(c.io.in.bits.data("field3").data, 150.U)
  poke(c.io.in.bits.data("field3").predicate, false.B)

  step(1)
  poke(c.io.in.bits.enable.control, true.B)
  poke(c.io.in.valid, true.B)
  poke(c.io.in.bits.data("field0").data, 0.U)
  poke(c.io.in.bits.data("field0").predicate, true.B)
  poke(c.io.out.ready, true.B)
  poke(c.io.in.bits.data("field1").data, 50.U)
  poke(c.io.in.bits.data("field1").predicate, true.B)
  poke(c.io.in.bits.data("field2").data, 100.U)
  poke(c.io.in.bits.data("field2").predicate, true.B)
  poke(c.io.in.bits.data("field3").data, 150.U)
  poke(c.io.in.bits.data("field3").predicate, true.B)

  var time = 0
  var result = false
  while (time < 40000 && !result) {
    time += 1
    step(1)
    if (peek(c.io.out.valid) == 1) {
      val data = peek(c.io.out.bits.data("field0").data)
      expect(c.io.out.bits.data("field0").data, 4900.U)
      result = true
      println(Console.BLUE + s"*** matrix_add finished. Result: $data, Run time: $time cycles." + Console.RESET)
    }
  }

  if (!result) {
    println(Console.RED + "*** Timeout." + Console.RESET)
    fail
  }
}

class matrix_addDF_test extends FlatSpec with Matchers {

  val inDataVec = List()
  val inAddrVec = List.range(0, 32 * inDataVec.length, 32)
  val outAddrVec = List.range(32 * inDataVec.length, 32 * inDataVec.length + 32, 32)
  val outDataVec = List(4900)

  implicit val p = new WithAccelConfig(HeteaccAccelParams())

  it should s"Test: direct connection" in {
    Driver.execute(
      Array(
        "-tn", "matrix_add",
        "-tbn", "verilator",
        "-td", s"test_run_dir/matrix_add",
        "-tts", "0001",
        "--generate-vcd-output", "on"
      ),
      () => new matrix_add_main()(p)) {
      c => new matrix_add01(c)(inAddrVec, inDataVec, outAddrVec, outDataVec)
    } should be(true)
  }
}
