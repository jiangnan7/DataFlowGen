package heteacc.node

import chisel3._
import chisel3.util._
import chipsalliance.rocketchip.config._
import chisel3.util.experimental.BoringUtils
import heteacc.interfaces._
import utility.Constants._

import org.scalatest.{FlatSpec, Matchers}
import heteacc.config._
import chisel3.Module
import util._
import chipsalliance.rocketchip.config._
// class MemLoadTest(size: Int, width: Int, portNum: Int,
//                 NumPredOps: Int,
//                 NumSuccOps: Int,
//                 NumOuts: Int,
//                 Typ: UInt = MT_D,
//                 ID: Int,
//                 RouteID: Int,
//                 Debug : Boolean =false,
//                 GuardVal : Int = 0) (implicit p: Parameters) 
//                 extends MultiIOModule with InitSyncMem {

  
//   val memload = Module(new MemLoad(NumPredOps, NumSuccOps, NumOuts,
//                 Typ, ID, RouteID))
                
//   val r_data = IO(Output(UInt(width.W)))

//   override lazy val io = IO(new LoadIO(NumPredOps, NumSuccOps, NumOuts, Debug))  
//   val mem = SyncReadMem(size, UInt(width.W))
//   initMem("dataset/memory/in_0.txt")
//   io.GepAddr.valid := true.B
//   io.GepAddr.bits := mem.read(io.GepAddr.bits.data, io.MemReq.valid)

//   memload.io.GepAddr := io.GepAddr


// }

// class LoadIO(NumPredOps: Int,
//              NumSuccOps: Int,
//              NumOuts: Int,
//              Debug : Boolean =false)(implicit p: Parameters)
//   extends HandShakingIOPS(NumPredOps, NumSuccOps, NumOuts, Debug)(new DataBundle) {
//   // GepAddr: The calculated address comming from GEP node
//   val GepAddr = Flipped(Decoupled(new DataBundle))
//   // Memory request
//   val MemReq = Decoupled(new ReadReq())
//   // Memory response.
//   val MemResp = Input(Valid(new ReadResp()))
// }

class LoadIO(NumPredOps: Int,
             NumSuccOps: Int,
             NumOuts: Int,
             Debug : Boolean =false)(implicit p: Parameters)
  extends HandShakingIOPS(NumPredOps, NumSuccOps, NumOuts, Debug)(new DataBundle) {
  // GepAddr: The calculated address comming from GEP node
  val GepAddr = Flipped(Decoupled(new DataBundle))
  // Memory request
  val MemReq = Decoupled(new ReadReq())
  // Memory response.
  val MemResp = Input(Valid(new ReadResp()))//val MemResp = Input(Valid(new ReadResp()))
}

/**
  * @brief Load Node. Implements load operations
  * @details [load operations can either reference values in a scratchpad or cache]
  * @param NumPredOps [Number of predicate memory operations]
  */
class MemLoad(NumPredOps: Int,
                NumSuccOps: Int,
                NumOuts: Int,
                Typ: UInt = MT_D,
                ID: Int,
                RouteID: Int
               , Debug : Boolean =false
               , GuardVal : Int = 0)
               (implicit p: Parameters,
                name: sourcecode.Name,
                file: sourcecode.File)
  extends HandShaking(NumPredOps, NumSuccOps, NumOuts, ID, Debug)(new DataBundle)(p) {

  override lazy val io = IO(new LoadIO(NumPredOps, NumSuccOps, NumOuts, Debug))
  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)
  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "


  /*=============================================
  =            Registers                        =
  =============================================*/
  // OP Inputs
  val addr_R = RegInit(DataBundle.default)
  val addr_valid_R = RegInit(false.B)

  // Memory Response
  val data_R = RegInit(DataBundle.default)
  val data_valid_R = RegInit(false.B)

  // State machine
  val s_idle :: s_RECEIVING :: s_Done :: Nil = Enum(3)
  val state = RegInit(s_idle)


  /*================================================
  =            Latch inputs. Wire up output            =
  ================================================*/

  //Initialization READY-VALIDs for GepAddr and Predecessor memory ops
  io.GepAddr.ready := ~addr_valid_R
  when(io.GepAddr.fire) {
    addr_R := io.GepAddr.bits
    addr_valid_R := true.B
  }

  //**********************************************************************
  var log_id = WireInit(ID.U((6).W))
  var GuardFlag = WireInit(0.U(1.W))

  var log_out_reg = RegInit(0.U((xlen-7).W))
  val writeFinish = RegInit(false.B)
  //log_id := ID.U
  //test_value := Cat(GuardFlag,log_id, log_out)
  val log_value = WireInit(0.U(xlen.W))
  log_value := Cat(GuardFlag, log_id, log_out_reg)



  /*============================================
  =            Predicate Evaluation            =
  ============================================*/

  val complete = IsSuccReady() && IsOutReady()
  val predicate = addr_R.predicate && enable_R.control
  val mem_req_fire = addr_valid_R && IsPredValid()


  // Wire up Outputs
  for (i <- 0 until NumOuts) {
    io.Out(i).bits := data_R
    io.Out(i).bits.predicate := predicate
    io.Out(i).bits.taskID := addr_R.taskID | enable_R.taskID
  }

  io.MemReq.valid := false.B
  io.MemReq.bits.address := addr_R.data
  io.MemReq.bits.Typ := Typ
  io.MemReq.bits.RouteID := RouteID.U
  io.MemReq.bits.taskID := addr_R.taskID

  // Connect successors outputs to the enable status
  when(io.enable.fire) {
    succ_bundle_R.foreach(_ := io.enable.bits)
  }




  /*=============================================
  =            ACTIONS (possibly dangerous)     =
  =============================================*/


  switch(state) {
    is(s_idle) {
      when(enable_valid_R && mem_req_fire) {
        when(enable_R.control && predicate) {
          io.MemReq.valid := true.B
          when(io.MemReq.ready) {
            state := s_RECEIVING
          }
        }.otherwise {
          data_R.predicate := false.B
          ValidSucc()
          ValidOut()
          state := s_Done
        }
      }
    }
    is(s_RECEIVING) {
      when(io.MemResp.valid) {

        // Set data output registers
        data_R.data := io.MemResp.bits.data

        // if (Debug) {
        //   when(data_R.data =/= GuardVal.U) {
        //     GuardFlag := 1.U
        //     log_out_reg :=  data_R.data
        //     data_R.data := GuardVal.U

        //   }.otherwise {
        //     GuardFlag := 0.U
        //     log_out_reg :=  data_R.data
        //   }
        // }


        data_R.predicate := true.B
        // out_valid_R.foreach(_ := true.B)
        ValidSucc()
        ValidOut()
        // Completion state.
        state := s_Done

      }
    }
    is(s_Done) {
      when(complete) {
        // Clear all the valid states.
        // Reset address
        // addr_R := DataBundle.default
        addr_valid_R := false.B
        // Reset data
        // data_R := DataBundle.default
        data_valid_R := false.B
        // Reset state.
        Reset()
        // Reset state.
        state := s_idle
        if (log) {
          printf("[LOG] " + "[" + module_name + "] [TID->%d] [LOAD] " + node_name + ": Output fired @ %d, Address:%d, Value: %d\n",
            enable_R.taskID, cycleCount, addr_R.data, data_R.data)
        }
      }
    }
  }
}



class LoadCacheIO(NumPredOps: Int,
                  NumSuccOps: Int,
                  NumOuts: Int,
                  Debug: Boolean = false)(implicit p: Parameters)
  extends HandShakingIOPS(NumPredOps, NumSuccOps, NumOuts, Debug)(new DataBundle) {
  
  val GepAddr = Flipped(Decoupled(new DataBundle))
  val MemReq = Decoupled(new MemReq)
  val MemResp = Flipped(Valid(new MemResp))
}

/**
 * @brief Load Node. Implements load operations
 * @details [load operations can either reference values in a scratchpad or cache]
 * @param NumPredOps [Number of predicate memory operations]
 */
class UnTypLoadCache(NumPredOps: Int,
                     NumSuccOps: Int,
                     NumOuts: Int,
                     ID: Int,
                     RouteID: Int,
                     Debug: Boolean = false,
                     GuardAddress: Seq[Int] = List(),
                     GuardData: Seq[Int] = List())
                    (implicit p: Parameters,
                     name: sourcecode.Name,
                     file: sourcecode.File)
  extends HandShaking(NumPredOps, NumSuccOps, NumOuts, ID, Debug)(new DataBundle)(p)
    // with HasAccelShellParams  with HasDebugCodes {
  {

  override lazy val io = IO(new LoadCacheIO(NumPredOps, NumSuccOps, NumOuts, Debug))

  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)
  /**
   * Registers
   */
  // OP Inputs
  val addr_R = RegInit(DataBundle.default)
  val addr_valid_R = RegInit(false.B)

  // Memory Response
  val data_R = RegInit(DataBundle.default)
  val data_valid_R = RegInit(false.B)

  // State machine
  val s_idle :: s_RECEIVING :: s_Done :: Nil = Enum(3)
  val state = RegInit(s_idle)


  /*================================================
  =            Latch inputs. Wire up output            =
  ================================================*/

  //Initialization READY-VALIDs for GepAddr and Predecessor memory ops
  io.GepAddr.ready := ~addr_valid_R
  when(io.GepAddr.fire) {
    addr_R := io.GepAddr.bits
    addr_valid_R := true.B
  }

  /**
   * Debug signals
   */
  // val address_value_valid = WireInit(false.B)
  // val address_value_ready = WireInit(true.B)

  // val data_value_valid = WireInit(false.B)
  // val data_value_ready = WireInit(true.B)

  // val arb = Module(new Arbiter(UInt(dbgParams.packetLen.W), 2))
  // val data_queue = Module(new Queue(UInt(dbgParams.packetLen.W), entries = 20))


  def isAddrFire(): Bool = {
    enable_valid_R && addr_valid_R && enable_R.control && state === s_idle && io.MemReq.ready //&& address_value_ready
  }

  def complete(): Bool = {
    IsSuccReady() && IsOutReady()
  }

  def isRespValid(): Bool = {
    state === s_Done && complete()// && data_value_ready
  }


  val predicate = enable_R.control
  val mem_req_fire = addr_valid_R && IsPredValid()



  // Wire up Outputs
  for (i <- 0 until NumOuts) {
    // io.Out(i).bits := Mux(is_data_buggy, correct_data_val, data_R)
   io.Out(i).bits :=  data_R
  }

  // Initilizing the MemRequest bus
  io.MemReq.valid := false.B
  io.MemReq.bits.data := 0.U
  io.MemReq.bits.addr := addr_R.data
  io.MemReq.bits.tag := RouteID.U
  io.MemReq.bits.taskID := addr_R.taskID
  io.MemReq.bits.mask := 0.U
  io.MemReq.bits.iswrite := false.B

  // Connect successors outputs to the enable status
  when(io.enable.fire) {
    succ_bundle_R.foreach(_ := io.enable.bits)
  }


  /*=============================================
  =            ACTIONS (possibly dangerous)     =
  =============================================*/


  switch(state) {
    is(s_idle) {
      when(enable_valid_R && mem_req_fire) {// && address_value_ready
        when(enable_R.control && predicate) {
          io.MemReq.valid := true.B
          when(io.MemReq.ready) {
            state := s_RECEIVING

          }
        }.otherwise {
          data_R.predicate := false.B
          ValidSucc()
          ValidOut()
          state := s_Done
        }
      }
    }
    is(s_RECEIVING) {
      when(io.MemResp.valid) {
        data_R.data := io.MemResp.bits.data
        data_R.predicate := true.B

        ValidSucc()
        ValidOut()
        // out_valid_R.foreach(_ := true.B)
        // Completion state.
        state := s_Done
      }
    }
    is(s_Done) {
      when(complete) {// && data_value_ready
        // Clear all the valid states.
        // addr_R := DataBundle.default
        addr_valid_R := false.B
        // Reset data
        // data_R := DataBundle.default
        data_valid_R := false.B
        // Reset state.
        Reset()
        // Reset state.
        state := s_idle
        if (log) {
          printf("[LOG] " + "[" + module_name + "] [TID->%d] [LOAD] " + node_name + ": Output fired @ %d, Address:%d, Value: %d\n",
            enable_R.taskID, cycleCount, addr_R.data, data_R.data)
        }
      }
    }
  }
}
