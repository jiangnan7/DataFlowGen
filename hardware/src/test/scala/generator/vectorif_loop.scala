package heteacc.generator

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester, OrderedDecoupledHWIOTester}
import org.scalatest.{Matchers, FlatSpec}
import heteacc.generator.if_loop_1DF
import chipsalliance.rocketchip.config._
import heteacc.config._
import utility._
import heteacc.interfaces._
import heteacc.accel._
import heteacc.acctest._
import heteacc.memory._


// 测试模块
class if_loop_main(implicit p: Parameters) extends AccelIO(List(32), List(32))(p) {

  val cache = Module(new Cache) // Simple Nasti Cache
  val memModel = Module(new NastiVMemSlave) // Model of DRAM to connect to Cache

  // // Connect the wrapper I/O to the memory model initialization interface so the
  // // test bench can write contents at start.
  memModel.io.nasti <> cache.io.nasti
  memModel.io.init.bits.addr := 0.U
  memModel.io.init.bits.data := 0.U
  memModel.io.init.valid := false.B
  cache.io.cpu.abort := false.B
  // cache.io.cpu.flush := DontCare
  


  // Wire up the cache and modules under test.
  //  val test04 = Module(new test04DF())
  val test13 = Module(new if_loop_1DF())//if_loop_1DF_unroll  if_loop_1DF

  //Put an arbiter infront of cache
  val CacheArbiter = Module(new MemArbiter(1))

  // Connect input signals to cache
  // CacheArbiter.io.cpu.MemReq(0) <> test13.io.MemReq
  // test13.io.MemResp <> CacheArbiter.io.cpu.MemResp(0)

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


class if_looptest01[T <: AccelIO](c: T)
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
  poke(c.io.in.bits.data("field0").data, 1.U)
  // poke(c.io.in.bits.data("field0").taskID, 0.U)
  poke(c.io.in.bits.data("field0").predicate, false.B)
  poke(c.io.out.ready, false.B)
  // poke(c.io.in.bits.data("field1").data, 0)
  // poke(c.io.in.bits.data("field1").taskID, 0)
  // poke(c.io.in.bits.data("field1").predicate, false)
  // poke(c.io.in.bits.data("field2").data, 0)
  // poke(c.io.in.bits.data("field2").taskID, 0)
  // poke(c.io.in.bits.data("field2").predicate, false)
  
  step(1)
  poke(c.io.in.bits.enable.control, true)
  poke(c.io.in.valid, true)
  poke(c.io.in.bits.data("field0").data, 1.U) // Array a[] base address
  poke(c.io.in.bits.data("field0").predicate, true)
  poke(c.io.out.ready, true.B)
  // poke(c.io.in.bits.data("field1").data, 1024) // Array b[] base address
  // poke(c.io.in.bits.data("field1").predicate, true)
  
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
  while (time < 2000 && !result) {
    time += 1
    step(1)
    val data = peek(c.io.out.bits.data("field0").data)
    val dtat = peek(c.io.in.bits.data("field0").data)
    println(Console.RED + s"*** Incorrect result received. input $dtat Got $data. Hoping for 9870" + Console.RESET) 
    // if(data != 0){
    //   println(Console.BLUE + s"*** Bgemm finished. Run time: $time cycles." + Console.RESET)
    //   System.exit(0) // 退出程序
    // }
    
    if (peek(c.io.out.valid) == 1) {
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

class vectorif_loop extends FlatSpec with Matchers {

  val inDataVec = List(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99)//, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199)
  val inAddrVec = List.range(0, 32 * inDataVec.length, 32)

  val outAddrVec = List.range(32 * inDataVec.length, 32 * inDataVec.length + (32 * 1), 32)
  val outDataVec = List(39770)
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
        "-tn", "if_loop_1DF",
        "-tbn", "verilator",
        "-td", s"test_run_dir/if_loop_2DF",
        "-tts", "0001",
        "--generate-vcd-output", "on"),
        
      () => new if_loop_main()(p)) {
      c => new if_looptest01(c)(inAddrVec, inDataVec, outAddrVec, outDataVec)
    } should be(true)
  }
}
