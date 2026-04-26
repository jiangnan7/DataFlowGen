package heteacc.node

import chisel3._
import heteacc.config._
import chisel3.Module
import heteacc.interfaces._
import util._
import chipsalliance.rocketchip.config._
import utility.UniformPrintfs

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
  extends HandShakingDynIO(NumOuts, Debug )(new DataBundle) {
  // LeftIO: Left input data for computation
  val Input = Flipped(Decoupled(new DataBundle()))

}

class BitCastNodeWithoutState(NumOuts: Int, ID: Int, Debug: Boolean= false)
              (implicit p: Parameters,
               name: sourcecode.Name,
               file: sourcecode.File)
  extends HandShakingDyn(NumOuts, ID, Debug)(new DataBundle())(p) {
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


class BroadcastNodeWithVectorizationIO(NumOuts: Int, Debug: Boolean = false)
                                     (implicit p: Parameters)
  extends HandShakingDynIO(NumOuts, Debug)(new DataBundle) {
  val Input = Flipped(Decoupled(new DataBundle()))

}

class BroadcastNodeWithVectorization(NumOuts: Int, NumLanes: Int, ID: Int)
                                  (sign: Boolean, Debug: Boolean = false)
                                  (implicit  p: Parameters,
                                   name: sourcecode.Name,
                                   file: sourcecode.File)
                                   extends HandShakingDyn(NumOuts, ID, Debug)(new DataBundle())(p)
    with HasAccelShellParams{

   override lazy val io = IO(new BroadcastNodeWithVectorizationIO(NumOuts*NumLanes,  Debug))

  val node_name = name.value
  val module_name = file.value.split("/").last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  /*===========================================*
   *            Registers                      *
   *===========================================*/

  private val join = Module(new Join(1))
  private val oehb = Module(new OEHB(0))

  join.pValid(0) := io.Input.valid
  io.Input.ready := join.ready(0)

  join.nReady := oehb.dataIn.ready
  oehb.dataIn.bits := DontCare
  oehb.dataIn.valid := join.valid

  /*===========================================*
   *            Validity and Output Logic      *
   *===========================================*/

  // val nReadyReg = RegNext(io.Out.map(_.ready).reduce(_ && _), false.B)
  val nReadyReg = RegEnable(io.Out.map(_.ready).reduce(_ && _), false.B, io.Input.valid)

  oehb.dataOut.ready := nReadyReg

  val validReg = RegNext(oehb.dataOut.valid, false.B)
  for (i <- 0 until NumOuts*NumLanes) {
    io.Out(i).valid := validReg
  }
  for (i <- 0 until NumOuts*NumLanes) {
    io.Out(i).bits := Mux(io.Input.valid, io.Input.bits, 0.U.asTypeOf(new DataBundle()))
  }
  io.Out.foreach(_.bits := DataBundle(io.Input.bits.data, true.B, true.B))


  when(IsOutReady()) {
    Reset()
    if (log) {
      for (lane <- 0 until NumLanes) {
        printf(p"[LOG] [${module_name}] [TID: 0] [Broadcast] [${node_name}] [Lane: ${lane}] " +
          p"[InData: 0x${Hexadecimal(io.Input.bits.data)}] " +
          p"[Out: 0x${Hexadecimal(io.Out(lane).bits.data)}] " +
          p"[Cycle: ${cycleCount}]\n")
      }
    }
  }
}


class BitcastNodeWithVectorizationIO(NumOuts: Seq[Int], NumLanes: Int)
                                     (implicit p: Parameters)  extends AccelBundle {
  val Input = Vec(NumLanes, Flipped(Decoupled(new DataBundle())))
  val Out = new VariableDecoupledVec(NumOuts)
}

class BitcastNodeWithVectorization(NumOuts: Seq[Int], NumLanes: Int, ID: Int)
                                  (implicit val  p: Parameters,
                                   name: sourcecode.Name,
                                   file: sourcecode.File)
                                   extends Module with HasAccelParams with UniformPrintfs {

  val io = IO(new BitcastNodeWithVectorizationIO(NumOuts, NumLanes))

  val node_name = name.value
  val module_name = file.value.split("/").last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  /*===========================================*
   *            Registers                      *
   *===========================================*/

  private val join = Module(new Join(NumLanes))
  private val oehb = Module(new OEHB(0))

  for (lane <- 0 until NumLanes) {
    join.pValid(lane) := io.Input(lane).valid
    io.Input(lane).ready := join.ready(lane)
  }

  join.nReady := oehb.dataIn.ready

  oehb.dataIn.bits := DontCare
  oehb.dataIn.valid := join.valid

  /*===========================================*
   *            Validity and Output Logic      *
   *===========================================*/

  def ValidOut(): Bool = {
    io.Out.elements.map { case (_, vec) =>
      vec.map(_.ready).reduce(_ && _)
    }.reduce(_ && _)
  }

  oehb.dataOut.ready := ValidOut()


  for (i <- NumOuts.indices) {
    for (j <- 0 until NumOuts(i)) {
        io.Out.elements(s"field$i")(j).valid := oehb.dataOut.valid
    }
  }


  for (i <- NumOuts.indices) {
    for (j <- 0 until NumOuts(i)) {
      io.Out.elements(s"field$i")(j).bits := DataBundle(io.Input(i).bits.data, 1.U, 1.U)
      io.Out.elements(s"field$i")(j).valid := oehb.dataOut.valid
    }
  }

  def IsOutReady(): Bool = {
    if (NumOuts.isEmpty) {
      true.B
    } else {
      val out_ready_R = RegInit(VecInit(Seq.fill(NumOuts.sum)(false.B)))
      val fire_mask = (out_ready_R zip io.Out.elements.flatMap { case (_, vec) =>
        vec.map(out => out.valid && out.ready)
      }).map { case (a, b) => a | b }
      fire_mask.reduce(_ && _)
    }
  }

  when(IsOutReady()) {
    Reset()
    if (log) {
      for (lane <- 0 until NumLanes) {
        printf(p"[LOG] [${module_name}] [TID: 0] [Bitcast] [${node_name}] [Lane: ${lane}] " +
          p"[InData: 0x${Hexadecimal(io.Input(lane).bits.data)}] " +
          // p"[Out: 0x${Hexadecimal(io.Out.elements("filed$lane")(0).bits.data)}] " +
          p"[Cycle: ${cycleCount}]\n")
      }
    }
  }
}
