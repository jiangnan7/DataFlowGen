package heteacc.generator

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester, OrderedDecoupledHWIOTester}
import org.scalatest.{Matchers, FlatSpec}
import heteacc.generator.gemmDF
import chipsalliance.rocketchip.config._
import heteacc.config._
import utility._
import heteacc.interfaces.NastiMemSlave
import heteacc.interfaces._
import heteacc.accel._
import heteacc.acctest._
import heteacc.memory._


// 测试模块
class gemm_main(implicit p: Parameters) extends AccelIO(List(32, 32, 32, 32, 32), List())(p) {

  val cache = Module(new Cache) // Simple Nasti Cache
  val memModel = Module(new NastiMemSlave) // Model of DRAM to connect to Cache

  // // Connect the wrapper I/O to the memory model initialization interface so the
  // // test bench can write contents at start.
  memModel.io.nasti <> cache.io.nasti
  memModel.io.init.bits.addr := 0.U
  memModel.io.init.bits.data := 0.U
  memModel.io.init.valid := false.B
  cache.io.cpu.abort := false.B


  // Wire up the cache and modules under test.
  //  val test04 = Module(new test04DF())
  val test13 = Module(new gemmDF())

  //Put an arbiter infront of cache
  val CacheArbiter = Module(new MemArbiter(2))

  // Connect input signals to cache
  CacheArbiter.io.cpu.MemReq(0) <> test13.io.MemReq
  test13.io.MemResp <> CacheArbiter.io.cpu.MemResp(0)

  //Connect main module to cache arbiter
  CacheArbiter.io.cpu.MemReq(1) <> io.req
  io.resp <> CacheArbiter.io.cpu.MemResp(1)

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


class gemmDF01[T <: AccelIO](c: T)
                                      (inAddrVec: List[Int], inDataVec: List[Int],
                                       outAddrVec: List[Int], outDataVec: List[Int])
  extends AccelTesterLocal(c)(inAddrVec, inDataVec, outAddrVec, outDataVec) {

  initMemory()

  // for(i <- 0 until inDataVec.length) {
  //   poke(c.io.req.bits.addr, inAddrVec(i))
  //   poke(c.io.req.bits.data, inDataVec(i))
  //   poke(c.io.req.bits.iswrite, true.B)
  //   step(1)
  // }
  // poke(c.io.req.bits.iswrite, false.B)
  // step(1)

  // 初始化输入信号
  // poke(c.io.in.bits.enable.control, false)
  // poke(c.io.in.bits.enable.taskID, 0)
  poke(c.io.in.valid, false)
  poke(c.io.in.bits.data("field0").data, 0.U)
  // poke(c.io.in.bits.data("field0").taskID, 0.U)
  poke(c.io.in.bits.data("field0").predicate, false.B)
  poke(c.io.out.ready, false.B)
  poke(c.io.in.bits.data("field1").data, 32.U)
  // poke(c.io.in.bits.data("field1").taskID, 0)
  poke(c.io.in.bits.data("field1").predicate, false)

  poke(c.io.in.bits.data("field2").data, 64.U)
  // poke(c.io.in.bits.data("field1").taskID, 0)
  poke(c.io.in.bits.data("field2").predicate, false)

  poke(c.io.in.bits.data("field3").data, 1088.U)
  // poke(c.io.in.bits.data("field1").taskID, 0)
  poke(c.io.in.bits.data("field3").predicate, false)

  poke(c.io.in.bits.data("field4").data, 2112.U)
  // poke(c.io.in.bits.data("field1").taskID, 0)
  poke(c.io.in.bits.data("field4").predicate, false)
  // poke(c.io.in.bits.data("field2").data, 0)
  // poke(c.io.in.bits.data("field2").taskID, 0)
  // poke(c.io.in.bits.data("field2").predicate, false)
  
  step(1)
  poke(c.io.in.bits.enable.control, true)
  poke(c.io.in.valid, true)
  poke(c.io.in.bits.data("field0").data, 0.U) // Array a[] base address
  poke(c.io.in.bits.data("field0").predicate, true)
  poke(c.io.out.ready, true.B)
  poke(c.io.in.bits.data("field1").data, 32) // Array b[] base address
  poke(c.io.in.bits.data("field1").predicate, true)
  
  poke(c.io.in.bits.data("field2").data, 64) // Array b[] base address
  poke(c.io.in.bits.data("field2").predicate, true)
  
  poke(c.io.in.bits.data("field3").data, 1088) // Array b[] base address
  poke(c.io.in.bits.data("field3").predicate, true)
  
  poke(c.io.in.bits.data("field4").data, 2112) // Array b[] base address
  poke(c.io.in.bits.data("field4").predicate, true)
  
  // poke(c.io.out.ready, true.B)
  // step(1)
  // poke(c.io.in.bits.enable.control, true.B)
  // poke(c.io.in.valid, true.B)
  // poke(c.io.in.bits.data("field0").data, 64.U)
  // poke(c.io.in.bits.data("field0").predicate, false)


  

  // poke(c.io.in.bits.data("field1").data, 0.U)
  // poke(c.io.in.bits.data("field1").predicate, false)

  var time = 0 //Cycle counter
  var result = false
  while (time < 5000 && !result) {
    time += 1
    step(1)
    // val data = peek(c.io.out.bits.data("field0").data)
    val dtat = peek(c.io.in.bits.data("field0").data)
    println(Console.RED + s"*** Incorrect result received. input $dtat Got. Hoping for 9870" + Console.RESET) 
    // if(data != 0){
    //   println(Console.BLUE + s"*** Bgemm finished. Run time: $time cycles." + Console.RESET)
    //   System.exit(0) // 退出程序
    // }
    
    if (peek(c.io.out.valid) == 1) {// && peek(c.io.out.bits.data("field0").predicate) == 1
      result = true
      // val data = peek(c.io.out.bits.data("field0").data)
      // println(Console.RED + s"*** Incorrect result received. Got $data. Hoping for 9870" + Console.RESET) 
      println(Console.BLUE + s"*** Bgemm finished. Run time: $time cycles." + Console.RESET)
    }
  }
  //  Peek into the CopyMem to see if the expected data is written back to the Cache

  // checkMemory()

  if (!result) {
    println(Console.RED + "*** Timeout." + Console.RESET)
    fail
  }
}

class gemm_test extends FlatSpec with Matchers {

  val inDataVec = List(2,2,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32


)


  val inAddrVec = List.range(0, 32 * inDataVec.length, 32) //32

  // val outAddrVec = List.range(32 * inDataVec.length, 32 * inDataVec.length + (32 * 1), 32)
  val outAddrVec = List.range(32 * inDataVec.length, 32 * inDataVec.length + (32 * 1),32)
  val outDataVec = List(9870)
  // val inAddrVec = List.range(0, (4 * 5), 4)
  // val inDataVec = List(1, 2, 3, 4, 5)
  // val outAddrVec = List.range(20, 20 + (4 * 5), 4)
  // val outDataVec = List(2, 4, 6, 8, 10)

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
        "-tn", "gemm",
        "-tbn", "verilator",
        "-td", s"test_run_dir/gemm",
        "-tts", "0001",
        "--generate-vcd-output", "on"),
        
      () => new gemm_main()(p)) {
      c => new gemmDF01(c)(inAddrVec, inDataVec, outAddrVec, outDataVec)
    } should be(true)
  }
}