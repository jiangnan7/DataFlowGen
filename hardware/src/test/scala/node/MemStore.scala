// package heteacc.node

// import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester, OrderedDecoupledHWIOTester}
// import org.scalatest.{Matchers, FlatSpec}

// import chipsalliance.rocketchip.config._
// import heteacc.config._
// import utility._

// class StoreNodeTests(c: MemStore) extends PeekPokeTester(c) {
//   poke(c.io.GepAddr.valid, false)
//   poke(c.io.enable.valid, false)
//   poke(c.io.inData.valid, false)
//   poke(c.io.PredOp(0).valid, true)
//   poke(c.io.MemReq.ready, false)
//   poke(c.io.MemResp.valid, false)


//   poke(c.io.SuccOp(0).ready, true)
//   poke(c.io.Out(0).ready, false)


//   for (t <- 0 until 20) {

//     step(1)

//     //IF ready is set
//     // send address
//     if (peek(c.io.GepAddr.ready) == 1) {
//       poke(c.io.GepAddr.valid, true)
//       poke(c.io.GepAddr.bits.data, 12)
//       poke(c.io.GepAddr.bits.predicate, true)
//       poke(c.io.inData.valid, true)
//       poke(c.io.inData.bits.data, t + 1)
//       poke(c.io.inData.bits.predicate, true)
//       // //         poke(c.io.inData.bits.valid,true)
//       poke(c.io.enable.bits.control, true)
//       poke(c.io.enable.valid, true)
//     }

//     if ((peek(c.io.MemReq.valid) == 1) && (t > 4)) {
//       poke(c.io.MemReq.ready, true)
//     }

//     if (t > 5 && peek(c.io.MemReq.ready) == 1) {
//       // poke(c.io.MemReq.ready,false)
//       // poke(c.io.MemResp.data,t)
//       poke(c.io.MemResp.valid, true)
//     }
//     printf(s"t: ${t}  io.Out: ${peek(c.io.Out(0))} \n")

//   }


// }


// import Constants._

// class StoreNodeTester extends FlatSpec with Matchers {
//   implicit val p = new WithAccelConfig ++ new WithTestConfig
//   it should "Store Node tester" in {
//     chisel3.iotesters.Driver.execute(
//       Array(
//         // "-ll", "Info",
//         "-tbn", "firrtl",
//         "-td", "test_run_dir/StoreNodeTester",
//         "-tts", "0001"),
//       () => new MemStore(NumPredOps = 1, NumSuccOps = 1, NumOuts = 1, Typ = MT_W, ID = 1, RouteID = 0)) { c =>
//       new StoreNodeTests(c)
//     } should be(true)
//   }
// }
