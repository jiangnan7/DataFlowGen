// package heteacc.memory

// import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester, OrderedDecoupledHWIOTester}
// import org.scalatest.{Matchers, FlatSpec}
// import heteacc.interfaces._
// import chipsalliance.rocketchip.config._
// import heteacc.config._

// import chisel3._
// import chisel3.util._
// import chisel3.util.experimental.loadMemoryFromFileInline

// trait InitSyncMem {

//   def mem: SyncReadMem[UInt]

//   def initMem(memoryFile: String) = loadMemoryFromFileInline(mem, memoryFile)
// }

// class InitMem(size: Int) extends MultiIOModule with InitSyncMem {

//   val mem = SyncReadMem(size, UInt(32.W))

// }

// class ReadMemoryControllerTests(c: ReadMemController)
// 	(implicit p: Parameters)
// 	extends PeekPokeTester(c) {

//     val mem= Module(new InitMem(1024))
//     mem.initMem("dataset/memory/in_0.txt")     
// // 	var readidx = 0
// 	poke(c.io.ReadIn(0).bits.address, 9)
// 	poke(c.io.ReadIn(0).bits.RouteID, 0)
// 	poke(c.io.ReadIn(0).bits.Typ,3)
// 	poke(c.io.ReadIn(0).valid,1)
// 	poke(c.io.MemReq.ready,true)
// 	poke(c.io.MemResp.valid,false)

// 	var req  = false
// 	var tag  = peek(c.io.MemReq.bits.tag)
// 	var reqT = 0
//     // in_arb.io.in(0).bits.RouteID := 0.U
//     // in_arb.io.in(0).bits.Typ := MT_W
//     // in_arb.io.in(0).valid := true.B
// 	poke(c.io.MemReq.ready,1)
// 	poke(c.io.MemResp.valid,false)
// 	for (t <- 0 until 12) {
//         printf(s"current_t: ${t} ---------------------------- \n")
// 		if((peek(c.io.MemReq.valid) == 1) && (peek(c.io.MemReq.ready) == 1)) {
// 			printf(s"t: ${t} ---------------------------- \n")
// 			req  = true
// 			tag  = peek(c.io.MemReq.bits.tag)//这里的标签似乎是不断增加，0，1，2，
// 			reqT = t
// 		}
// 		if ((req == true) && (t > reqT))
// 		{
// 			poke(c.io.MemResp.valid,true)
// 			poke(c.io.MemResp.bits.data, 0xdeadbeefL)
// 			printf("Tag:%d ",tag.U)
// 			poke(c.io.MemResp.bits.tag,peek(c.io.MemReq.bits.tag))
// 			req = false
// 		}
// 		if (req == true) {
// 			poke(c.io.MemReq.ready,false)
// 		} else {
// 			poke(c.io.MemReq.ready,true)
// 		}
//     step(1)
//   }
// }


// class ReadMemoryControllerTester extends  FlatSpec with Matchers {
//   implicit val p = new WithAccelConfig ++ new WithTestConfig
//   it should "Memory Controller tester" in {
//     chisel3.iotesters.Driver.execute(
//         Array("--target-dir", "generated_dut/", "--generate-vcd-output", "on","-X", "verilog"),
//        () => new ReadMemController(NumOps=1,BaseSize=2,NumEntries=2)(p)) {
//       c => new ReadMemoryControllerTests(c)
//     } should be(true)
//   }
// }
