package heteacc.execution

import chisel3._
import chisel3.Module
import utility.UniformPrintfs
import chipsalliance.rocketchip.config._
import heteacc.config._
import heteacc.interfaces._
import util._


/**
 * ExecutionBlockNode
 *
 * @param BID
 * @param NumOuts
 * @param p
 * @param name
 * @param file
 */

class BasicBlockNoMaskFastIO(val NumOuts: Int)(implicit p: Parameters)
  extends AccelBundle()(p) {
  // Output IO
  val predicateIn = Flipped(Decoupled(new ControlBundle()))
  val Out = Vec(NumOuts, Decoupled(new ControlBundle))

}


class ExecutionBlockVecIO(val NumInputs: Int, val NumOuts: Int)(implicit p: Parameters)
  extends AccelBundle()(p) {
  // Output IO
  val predicateIn = Vec(NumInputs, Flipped(Decoupled(new ControlBundle())))
  val Out = Vec(NumOuts, Decoupled(new ControlBundle))

}

/**
 * ExecutionBlockNode details:
 * 1) Node can one one or multiple predicates as input and their type is controlBundle
 * 2) State machine inside the node waits for all the inputs to arrive and then fire.
 * 3) The ouput value is OR of all the input values
 * 4) Node can fire outputs at the same cycle if all the inputs. Since, basic block node
 * is only for circuit simplification therefore, in case that we know output is valid
 * we don't want to waste one cycle for latching purpose. Therefore, output can be zero
 * cycle.
 *
 * @param BID
 * @param NumInputs
 * @param NumOuts
 * @param p
 * @param name
 * @param file
 */

class ExecutionBlockNode(BID: Int, val NumInputs: Int = 1, val NumOuts: Int, val fast: Boolean = true)
                              (implicit val p: Parameters,
                               name: sourcecode.Name,
                               file: sourcecode.File)
  extends Module with HasAccelParams with UniformPrintfs {

  val io = IO(new ExecutionBlockVecIO(NumInputs, NumOuts)(p))

  // Defining IO latches

  // Data Inputs
  val in_data_R = Seq.fill(NumInputs)(RegInit(ControlBundle.default))
  val in_data_valid_R = Seq.fill(NumInputs)(RegInit(false.B))

  val output_valid_R = Seq.fill(NumOuts)(RegInit(false.B))
  val output_fire_R = Seq.fill(NumOuts)(RegInit(false.B))

  //Make sure whenever output is fired we latch it
  for (i <- 0 until NumInputs) {
    io.predicateIn(i).ready := ~in_data_valid_R(i)
    when(io.predicateIn(i).fire) {
      in_data_R(i) <> io.predicateIn(i).bits
      in_data_valid_R(i) := true.B
    }
  }

  val in_task_ID = 0.U(1.W)
  //Output connections
  for (i <- 0 until NumOuts) {
    when(io.Out(i).fire) {
      output_fire_R(i) := true.B
      output_valid_R(i) := false.B
    }
  }




  val out_fire_mask = (output_fire_R zip io.Out.map(_.fire)) map { case (a, b) => a | b }


  // val output_valid_map = for (i <- 0 until NumInputs) yield {
  //   val ret = Mux(io.predicateIn(i).fire, true.B, in_data_valid_R(i))
  //   ret
  // }
  //Connecting output signals
  for (i <- 0 until NumOuts) {
    io.Out(i).bits := ControlBundle.default
    io.Out(i).valid <> output_valid_R(i)
  }


  val s_idle :: s_fire :: Nil = Enum(2)
  val state = RegInit(s_idle)


  switch(state) {
    is(s_idle) {
      if (fast) {
          io.Out.foreach(_.valid := true.B)
        }

        (output_valid_R zip io.Out.map(_.fire)).foreach { case (a, b) => a := b ^ true.B }

        state := s_fire
    }
    is(s_fire) {
      //Restart the states
      when(out_fire_mask.reduce(_ & _)) {
        in_data_R foreach (_ := ControlBundle.default)
        in_data_valid_R foreach (_ := false.B)

        output_fire_R foreach (_ := false.B)

        state := s_idle
      }
    }
  }
}

class BasicBlockIO(NumInputs: Int,
                   NumOuts: Int,
                   NumPhi: Int)
                  (implicit p: Parameters)
  extends HandShakingCtrlMaskIO(NumInputs, NumOuts, NumPhi) {

  val predicateIn = Vec(NumInputs, Flipped(Decoupled(new ControlBundle())))

}

class BasicBlockNode(NumInputs: Int,
                     NumOuts: Int,
                     NumPhi: Int,
                     BID: Int)
                    (implicit p: Parameters,
                     name: sourcecode.Name,
                     file: sourcecode.File)
  extends HandShakingCtrlMask(NumInputs, NumOuts, NumPhi, BID)(p) {

  override lazy val io = IO(new BasicBlockIO(NumInputs, NumOuts, NumPhi))
  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  override val printfSigil = node_name + BID + " "
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  //Assertion
  /*===========================================*
   *            Registers                      *
   *===========================================*/
  // OP Inputs
  val predicate_control_R = Seq.fill(NumInputs)(RegInit(false.B))
  val predicate_task_R = Seq.fill(NumInputs)(RegInit(0.U(tlen.W)))
  val predicate_debug_R = Seq.fill(NumInputs)(RegInit(false.B))
  val predicate_valid_R = Seq.fill(NumInputs)(RegInit(false.B))

  val s_IDLE :: s_LATCH :: Nil = Enum(2)
  val state = RegInit(s_IDLE)

  /*===========================================*
   *            Valids                         *
   *===========================================*/

  val predicate = predicate_control_R.reduce(_ | _)
  val predicate_task = predicate_task_R.reduce(_ | _)
  val predicate_debug = predicate_debug_R.reduce(_ | _)

  val start = (io.predicateIn.map(_.fire) zip predicate_valid_R) map { case (a, b) => a | b } reduce (_ & _)
  val fast_predicate_control = predicate_control_R.indices.map { i =>
    Mux(predicate_valid_R(i), predicate_control_R(i), io.predicateIn(i).bits.control)
  }
  val fast_predicate = fast_predicate_control.reduce(_ | _)
  val fast_predicate_task = predicate_task_R.indices.map { i =>
    Mux(predicate_valid_R(i), predicate_task_R(i), io.predicateIn(i).bits.taskID)
  }.reduce(_ | _)
  val fast_predicate_debug = predicate_debug_R.indices.map { i =>
    Mux(predicate_valid_R(i), predicate_debug_R(i), io.predicateIn(i).bits.debug)
  }.reduce(_ | _)
  val out_ready_now = if (NumOuts == 0) true.B else io.Out.map(_.ready).reduce(_ && _)
  val mask_ready_now = if (NumPhi == 0) true.B else io.MaskBB.map(_.ready).reduce(_ && _)
  val fire_now = state === s_IDLE && start && out_ready_now && mask_ready_now

  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/


  for (i <- 0 until NumInputs) {
    io.predicateIn(i).ready := ~predicate_valid_R(i)
    when(io.predicateIn(i).fire) {
      predicate_control_R(i) := io.predicateIn(i).bits.control
      predicate_task_R(i) := io.predicateIn(i).bits.taskID
      predicate_debug_R(i) := io.predicateIn(i).bits.debug
      predicate_valid_R(i) := true.B
    }
  }

  // Wire up Outputs
  for (i <- 0 until NumOuts) {
    io.Out(i).bits.control := true.B//predicate
    io.Out(i).bits.taskID := Mux(fire_now, fast_predicate_task, predicate_task)
    io.Out(i).bits.debug := Mux(fire_now, fast_predicate_debug, predicate_debug)
    io.Out(i).valid := out_valid_R(i) || fire_now
  }

  // Wire up mask output
  for (i <- 0 until NumPhi) {
    io.MaskBB(i).bits := Mux(fire_now, VecInit(fast_predicate_control).asUInt, VecInit(predicate_control_R).asUInt)
    io.MaskBB(i).valid := mask_valid_R(i) || fire_now
  }


  /*============================================*
   *            ACTIONS (possibly dangerous)    *
   *============================================*/

  switch(state) {
    is(s_IDLE) {
      when(start) {
        when(fire_now) {
          predicate_valid_R.foreach(_ := false.B)
        }.otherwise {
          ValidOut()
          state := s_LATCH
        }
      }
    }
    is(s_LATCH) {
      when(IsOutReady()) {
        predicate_valid_R.foreach(_ := false.B)
        Reset()
        state := s_IDLE

        when(predicate) {
          if (log) {
            printf(p"[LOG] [${module_name}] [TID: ${predicate_task}] [BB] " +
              p"${node_name}] [Mask: 0x${Hexadecimal(VecInit(predicate_control_R).asUInt)}]\n")
          }
        }.otherwise {
          if (log) {
            printf("[LOG] " + "[" + module_name + "] " + node_name + ": Output fired @ %d -> 0 predicate\n", cycleCount)
          }
        }
      }
    }

  }
}
