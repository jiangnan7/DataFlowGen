package heteacc.node

import chisel3._
import heteacc.config._
import chisel3.Module
import heteacc.interfaces._
import util._
import chipsalliance.rocketchip.config._


class BitCastNodeIO(NumOuts: Int, Debug:Boolean)
                (implicit p: Parameters)
  extends HandShakingIONPS(NumOuts, Debug )(new DataBundle) {
  // LeftIO: Left input data for computation
  val Input = Flipped(Decoupled(new DataBundle()))

}

class BitCastNode(NumOuts: Int, ID: Int, Debug: Boolean= false)
              (implicit p: Parameters,
               name: sourcecode.Name,
               file: sourcecode.File)
  extends HandShakingNPS(NumOuts, ID, Debug)(new DataBundle())(p) {
  override lazy val io = IO(new BitCastNodeIO(NumOuts, Debug))

   // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize

  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  /*===========================================*
   *            Registers                      *
   *===========================================*/
  // Left Input
  val data_R = RegInit(DataBundle.default)
  val data_valid_R = RegInit(false.B)

  val s_IDLE :: s_COMPUTE :: Nil = Enum(2)
  val state = RegInit(s_IDLE)


  //Output register
  val out_data_R = RegNext(Mux(enable_R.control, data_R.data, 0.U), init = 0.U)
  val predicate = Mux(enable_valid_R, enable_R.control ,io.enable.bits.control)
  val taskID = Mux(enable_valid_R, enable_R.taskID ,io.enable.bits.taskID)

  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/

  io.Input.ready := ~data_valid_R
  when(io.Input.fire) {
    data_R <> io.Input.bits
    data_valid_R := true.B
  }


  // Wire up Outputs
  // The taskID's should be identical except in the case
  // when one input is tied to a constant.  In that case
  // the taskID will be zero.  Logical OR'ing the IDs
  // Should produce a valid ID in either case regardless of
  // which input is constant.
  io.Out.foreach(_.bits := DataBundle(out_data_R, taskID, predicate))

  /*============================================*
   *            State Machine                   *
   *============================================*/
  switch(state) {
    is(s_IDLE) {
      when(enable_valid_R) {//cast_2.io.enable
        io.Out.foreach(_.bits := DataBundle(data_R.data, taskID, predicate))
        io.Out.foreach(_.valid := true.B)
        ValidOut()
        state := s_COMPUTE
        if (log) {
          printf("[LOG] " + "[" + module_name + "] " + "[TID->%d] [CMP] " +
            node_name + ": Output fired @ %d, Value: %d\n", taskID, cycleCount, data_R.data)
        }
      }
    }
    is(s_COMPUTE) {
      when(IsOutReady()) {
        // Reset data
        data_valid_R := false.B

        out_data_R := 0.U
        //Reset state
        state := s_IDLE
        Reset()


      }
    }
  }
  def isDebug(): Boolean = {
    Debug
  }
}


class BitCastNodeWithoutStateIO(NumOuts: Int, Debug:Boolean)
                (implicit p: Parameters)
  extends HandShakingIONPS(NumOuts, Debug )(new DataBundle) {
  // LeftIO: Left input data for computation
  val Input = Flipped(Decoupled(new DataBundle()))

}

class BitCastNodeWithoutState(NumOuts: Int, ID: Int, Debug: Boolean= false)
              (implicit p: Parameters,
               name: sourcecode.Name,
               file: sourcecode.File)
  extends HandShakingNPS(NumOuts, ID, Debug)(new DataBundle())(p) {
  override lazy val io = IO(new BitCastNodeWithoutStateIO(NumOuts, Debug))

   // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize

  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "
  val (cycleCount, _) = Counter(true.B, 32 * 1024)
  
  // io.Input.ready := true.B
  io.Out.foreach(_.bits := io.Input.bits)//DataBundle(out_data_R, io.enable.bits.taskID, io.enable.bits.control))
  // ValidOut()
  
  private val join = Module(new Join(1))
  // private val buff = Module(new DelayBuffer(2 - 1, 1))
  private val oehb = Module(new OEHB(0))

  join.pValid(0) := io.Input.valid
  // join.pValid(1) := io.RightIO.valid
  io.Input.ready := join.ready(0)
  // io.RightIO.ready := join.ready(1)
  join.nReady := oehb.dataIn.ready

  // buff.valid_in := join.valid
  // buff.ready_in := oehb.dataIn.ready

  oehb.dataIn.bits := DontCare
  // oehb.dataOut.ready := io.Out(0).ready
  oehb.dataIn.valid := join.valid
  // io.Out(0).valid := oehb.dataOut.valid

  oehb.dataOut.ready := io.Out.map(_.ready).reduce(_ && _)
  for (i <- 0 until NumOuts) {
    // oehb.dataOut.ready := io.Out(i).ready
    io.Out(i).valid := oehb.dataOut.valid
  }
  when(IsOutReady()) {
    if (log) {
      printf("[LOG] " + "[" + module_name + "] " + "[TID->%d] [CMP] " +
      node_name + ": Output fired @ %d, Value: %d\n", 
      cycleCount, io.Input.bits.data, io.Out(0).bits.data)

    }
    Reset()
  }
 
}