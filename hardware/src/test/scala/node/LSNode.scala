// package heteacc.node

// import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester, OrderedDecoupledHWIOTester}
// import org.scalatest.{Matchers, FlatSpec}

// import chipsalliance.rocketchip.config._
// import heteacc.config._
// import utility._



// class LoadTests(c: Load) extends PeekPokeTester(c) {

//   poke(c.io.address_in.valid, false)
//   poke(c.io.enable.valid, false)

//   poke(c.io.data_in.valid, false)
//   poke(c.io.Out(0).ready, true)

//   for (t <- 0 until 20) {

//     step(1)
    
//     // printf(s"t: ${t} c.io.address_in.ready: ${peek(c.io.address_in.ready)} \n")
//     //IF ready is set
//     // send address
//     if (peek(c.io.address_in.ready) == 1) {

//       poke(c.io.address_in.valid, true)
//       poke(c.io.address_in.bits.data, 12)
//       poke(c.io.address_out.valid, true)  
//       poke(c.io.enable.bits.control, true)
//       poke(c.io.enable.valid, true)
//     }
//     // printf(s"t: ${t} c.io.enable.valid: ${peek(c.io.enable.valid)} \n\n")

//     printf(s"t: ${t}  c.io.data_in: ${peek(c.io.Out(0))} \n")
//     printf(s"t: ${t}  c.io.address_out: ${peek(c.io.address_out.bits)} \n")
//     // if ((peek(c.io.data_in.valid) == 1) && (t > 4)) {
//     //   poke(c.io.MemReq.ready, true)
//     // }

//     if (t > 8) {
//       poke(c.io.data_in.valid, true)
//       poke(c.io.data_in.bits, 0 + t)
//     }
//   }
// }


// import Constants._

// class LoadTester extends FlatSpec with Matchers {
//   implicit val p = new WithAccelConfig
//   it should "Load Node tester" in {
//     chisel3.iotesters.Driver.execute(
//       Array(
//         // "-ll", "Info",
//         "-tn", "load01",
//         "-tbn", "verilator",
//         "-td", "test_run_dir/load01",
//         "-tts", "0001"),
//       () => new Load(NumOuts = 1, ID = 1, RouteID = 0)) { c =>
//       new LoadTests(c)
//     } should be(true)
//   }
// }



// class StoreTests(c: Store) extends PeekPokeTester(c) {

//   poke(c.io.address_in.valid, false)
//   poke(c.io.enable.valid, false)
//   poke(c.io.data_in.valid, false)

//   poke(c.io.Out(0).ready, true)

//   for (t <- 0 until 20) {

//     step(1)
    
//     // printf(s"t: ${t} c.io.address_in.ready: ${peek(c.io.address_in.ready)} \n")
//     //IF ready is set
//     // send address
//     if (peek(c.io.address_in.ready) == 1) {

//       poke(c.io.address_in.valid, true)
//       poke(c.io.address_in.bits, 12)

//       poke(c.io.data_in.valid, true)
//       poke(c.io.data_in.bits, t + 1)  

//       poke(c.io.address_out.ready, true)  
//       poke(c.io.enable.bits.control, true)
//       poke(c.io.enable.valid, true)
//     }
//     // printf(s"t: ${t} c.io.enable.valid: ${peek(c.io.enable.valid)} \n\n")

//     printf(s"t: ${t}  c.io.done: ${peek(c.io.Out(0))} \n")

//     // if ((peek(c.io.data_in.valid) == 1) && (t > 4)) {
//     //   poke(c.io.MemReq.ready, true)
//     // }

//     // if (t > 8) {
//     //   poke(c.io.data_in.valid, true)
//     //   poke(c.io.data_in.bits, 0 + t)
//     // }
//   }
// }


// import Constants._

// class StoreTester extends FlatSpec with Matchers {
//   implicit val p = new WithAccelConfig
//   it should "Store Node tester" in {
//     chisel3.iotesters.Driver.execute(
//       Array(
//         // "-ll", "Info",
//         "-tn", "store01",
//         "-tbn", "verilator",
//         "-td", "test_run_dir/store01",
//         "-tts", "0001"),
//       () => new Store(NumOuts = 1, ID = 1, RouteID = 0)) { c =>
//       new StoreTests(c)
//     } should be(true)
//   }
// }