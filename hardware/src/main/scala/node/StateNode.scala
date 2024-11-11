package heteacc.node

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, Driver, OrderedDecoupledHWIOTester, PeekPokeTester}
import chisel3.Module
import chisel3.testers._
import chisel3.util._
import org.scalatest.{FlatSpec, Matchers}
import chipsalliance.rocketchip.config._
import heteacc.config._
import heteacc.interfaces._
import muxes._
import util._
import utility.UniformPrintfs




class UBranchNode(NumPredOps: Int = 0,
                  NumOuts: Int = 1,
                  ID: Int)
                 (implicit p: Parameters,
                  name: sourcecode.Name,
                  file: sourcecode.File)
  extends HandShaking(NumPredOps, 0, NumOuts, ID)(new ControlBundle)(p) {
  override lazy val io = IO(new HandShakingIOPS(NumPredOps, 0, NumOuts)(new ControlBundle)(p))
  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)
  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "

  /*===========================================*
   *            Registers                      *
   *===========================================*/

  val s_idle :: s_OUTPUT :: Nil = Enum(2)
  val state = RegInit(s_idle)

  /*==========================================*
   *           Predicate Evaluation           *
   *==========================================*/

  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/

  /**
   * Combination of bits and valid signal from CmpIn whill result the output value:
   * valid == 0  ->  output = 0
   * valid == 1  ->  cmp = true  then 1
   * valid == 1  ->  cmp = false then 2
   *
   * @note data_R value is equale to predicate bit
   */
  // Wire up Outputs
  io.Out.foreach(_.bits := enable_R)

  switch(state) {
    is(s_idle) {
      when(IsEnableValid() && IsPredValid()) {
        state := s_OUTPUT
        ValidOut()
        io.Out.foreach(_.valid := true.B)
        if (log) {
          printf(p"[LOG] [${module_name}] [TID: ${enable_R.taskID}] " +
            p"[IDLE] " +
            p"[${node_name}] " +
            p"[Out: ${enable_R.control}] " +
            p"[Cycle: ${cycleCount}]\n")
        }
      }
    }

    is(s_OUTPUT) {
      when(IsOutReady()) {
        state := s_idle
        Reset()
        enable_R := ControlBundle.default
        if (log) {
          printf(p"[LOG] [${module_name}] [TID: ${enable_R.taskID}] " +
            p"[OUTPUT] " +
            p"[${node_name}] " +
            p"[Out: ${enable_R.control}] " +
            p"[Cycle: ${cycleCount}]\n")
        }
      }
    }
  }

}


/**
 * This class is the fast version of CBranch which the IO supports
 * a vector of output for each side True/False
 *
 */

class CBranchIO(val NumTrue: Int, val NumFalse: Int, val NumPredecessor: Int = 0)(implicit p: Parameters)
  extends AccelBundle()(p) {
  //Control signal
  val enable = Flipped(Decoupled(new ControlBundle))

  //Comparision result
  val CmpIO = Flipped(Decoupled(new DataBundle))

  // Control dependencies
  val PredOp = Vec(NumPredecessor, Flipped(Decoupled(new ControlBundle)))

  //Output
  val TrueOutput = Vec(NumTrue, Decoupled(new ControlBundle))
  val FalseOutput = Vec(NumFalse, Decoupled(new ControlBundle))
}

/**
  * @note
  * For Conditional Branch output is always equal to two!
  * Since your branch output wire to two different basic block only
  */

class CBranchNodeIO(NumOuts: Int = 1)
                   (implicit p: Parameters)
  extends HandShakingIONPS(NumOuts)(new ControlBundle) {

  // RightIO: Right input data for computation
  val CmpIO = Flipped(Decoupled(new DataBundle))

}

class CBranchNode(ID: Int)
                 (implicit p: Parameters,
                  name: sourcecode.Name,
                  file: sourcecode.File)
  extends HandShakingCtrlNPS(1, ID)(p) {
  override lazy val io = IO(new CBranchNodeIO())
  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)
  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "

  /*===========================================*
   *            Registers                      *
   *===========================================*/
  // OP Inputs
  val cmp_R = RegInit(DataBundle.default)
  val cmp_valid_R = RegInit(false.B)


  // Output wire
  //  val data_out_w = WireInit(VecInit(Seq.fill(2)(false.B)))
  val data_out_R = RegInit(VecInit(Seq.fill(1)(false.B)))

  //  val s_IDLE :: s_LATCH :: s_COMPUTE :: Nil = Enum(3)
  val s_IDLE :: s_COMPUTE :: Nil = Enum(2)
  val state = RegInit(s_IDLE)

  /*==========================================*
   *           Predicate Evaluation           *
   *==========================================*/

  val start = cmp_valid_R & IsEnableValid()

  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/

  // Predicate register
  //val pred_R = RegInit(init = false.B)


  io.CmpIO.ready := ~cmp_valid_R
  when(io.CmpIO.fire()) {
    cmp_R := io.CmpIO.bits
    cmp_valid_R := true.B
  }

  // Wire up Outputs
  io.Out(0).bits.control := data_out_R(0)
  io.Out(0).bits.taskID := enable_R.taskID
  io.Out(0).bits.debug <> DontCare

  /*============================================*
   *            STATE MACHINE                   *
   *============================================*/

  /**
    * Combination of bits and valid signal from CmpIn whill result the output value:
    * valid == 0  ->  output = 0
    * valid == 1  ->  cmp = true  then 1
    * valid == 1  ->  cmp = false then 2
    */

  switch(state) {
    is(s_IDLE) {
      when(IsEnableValid() && cmp_valid_R) {
        state := s_COMPUTE
        ValidOut()
        when(IsEnable()) {
          data_out_R(0) := cmp_R.data.asUInt.orR

        }.otherwise {
          data_out_R := VecInit(Seq.fill(1)(false.B))
        }
      }
    }
    is(s_COMPUTE) {
      when(IsOutReady()) {
        // Restarting
        //cmp_R := DataBundle.default
        cmp_valid_R := false.B

        // Reset output
        data_out_R := VecInit(Seq.fill(1)(false.B))
        //Reset state
        state := s_IDLE

        Reset()
        if (log) {
          printf("[LOG] " + "[" + module_name + "] [TID->%d] " +
            node_name + ": Output fired @ %d, Value: %d\n", enable_R.taskID, cycleCount, data_out_R.asUInt())
        }
      }
    }
  }

}
/**
 * This class is the fast version of CBranch which the IO supports
 * a vector of output for each side True/False
 *
 * @param ID Node id
 */

class CBranchNodeVariable(val NumTrue: Int = 1, val NumFalse: Int = 1, val NumPredecessor: Int = 0, val ID: Int)
                         (implicit val p: Parameters,
                          name: sourcecode.Name,
                          file: sourcecode.File)
  extends Module with HasAccelParams with UniformPrintfs {

  val io = IO(new CBranchIO(NumTrue = NumTrue, NumFalse = NumFalse, NumPredecessor = NumPredecessor))

  // Printf debugging
  override val printfSigil = "Node (CBR) ID: " + ID + " "
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize

  //Latching input comparision result
  val cmp_R = RegInit(ControlBundle.default)
  val cmp_valid = RegInit(false.B)

  //Latching control signal
  val enable_R = RegInit(ControlBundle.default)
  val enable_valid_R = RegInit(false.B)


  val predecessor_R = Seq.fill(NumPredecessor)(RegInit(ControlBundle.default))
  val predecessor_valid_R = Seq.fill(NumPredecessor)(RegInit(false.B))

  val output_true_R = RegInit(ControlBundle.default)
  val output_true_valid_R = Seq.fill(NumTrue)(RegInit(false.B))
  val fire_true_R = Seq.fill(NumTrue)(RegInit(false.B))

  val output_false_R = RegInit(ControlBundle.default)
  val output_false_valid_R = Seq.fill(NumFalse)(RegInit(false.B))
  val fire_false_R = Seq.fill(NumFalse)(RegInit(false.B))

  val task_id = enable_R.taskID | cmp_R.taskID


  // Latching CMP input
  io.CmpIO.ready := ~cmp_valid
  when(io.CmpIO.fire) {
    cmp_R.control := io.CmpIO.bits.data.orR()
    cmp_R.taskID := io.CmpIO.bits.taskID
    cmp_valid := true.B
  }

  for (i <- 0 until NumPredecessor) {
    io.PredOp(i).ready := ~predecessor_valid_R(i)
    when(io.PredOp(i).fire) {
      predecessor_R(i) := io.PredOp(i).bits
      predecessor_valid_R(i) := true.B
    }
  }

  def IsPredecessorValid(): Bool = {
    if (NumPredecessor == 0) {
      true.B
    }
    else {
      predecessor_valid_R.reduce(_ & _)
    }
  }


  // Latching enable signal
  io.enable.ready := ~enable_valid_R
  when(io.enable.fire) {
    enable_R <> io.enable.bits
    enable_valid_R := true.B
  }

  // Output for true and false sides
  val true_output = enable_R.control & cmp_R.control
  val false_output = enable_R.control && (~cmp_R.control)

  output_true_R.control := true_output
  output_true_R.taskID := task_id

  for (i <- 0 until NumTrue) {
    io.TrueOutput(i).bits <> output_true_R
    io.TrueOutput(i).valid <> output_true_valid_R(i)
  }

  for (i <- 0 until NumTrue) {
    when(io.TrueOutput(i).fire) {
      fire_true_R(i) := true.B
      output_true_valid_R(i) := false.B
    }
  }


  output_false_R.control := false_output
  output_false_R.taskID := task_id

  for (i <- 0 until NumFalse) {
    io.FalseOutput(i).bits <> output_false_R
    io.FalseOutput(i).valid <> output_false_valid_R(i)
  }

  for (i <- 0 until NumFalse) {
    when(io.FalseOutput(i).fire) {
      fire_false_R(i) := true.B
      output_false_valid_R(i) := false.B
    }
  }

  val fire_true_mask = fire_true_R.reduce(_ & _)
  val fire_false_mask = fire_false_R.reduce(_ & _)


  //Output register
  val s_idle :: s_fire :: Nil = Enum(2)
  val state = RegInit(s_idle)


  switch(state) {
    is(s_idle) {
      when(enable_valid_R && cmp_valid && IsPredecessorValid()) {

        output_true_valid_R.foreach(_ := true.B)
        output_false_valid_R.foreach(_ := true.B)

        state := s_fire

        when(enable_R.control) {
          when(cmp_R.control) {
            if (log) {
              printf(p"[LOG] [${module_name}] [TID: ${task_id}] [CBR] " +
                p"[${node_name}] [Out: T:1 - F:0] [Cycle: ${cycleCount}]\n")
            }
          }.otherwise {
            if (log) {
              printf(p"[LOG] [${module_name}] [TID: ${task_id}] [CBR] " +
                p"[${node_name}] [Out: T:0 - F:1] [Cycle: ${cycleCount}]\n")
            }
          }
        }.otherwise {
          if (log) {
            printf(p"[LOG] [${module_name}] [TID: ${task_id}] [CBR] " +
              p"[${node_name}] [Out: T:0 - F:0] [Cycle: ${cycleCount}]\n")
          }
        }
      }
    }
    is(s_fire) {

      //Now we can restart the states
      when(fire_true_mask && fire_false_mask) {
        //Latching input comparision result
        cmp_R := ControlBundle.default
        cmp_valid := false.B

        //Latching control signal
        enable_R := ControlBundle.default
        enable_valid_R := false.B
        predecessor_valid_R foreach (_ := false.B)

        output_true_R := ControlBundle.default
        output_true_valid_R.foreach(_ := false.B)
        fire_true_R.foreach(_ := false.B)

        output_false_R := ControlBundle.default
        output_false_valid_R.foreach(_ := false.B)
        fire_false_R.foreach(_ := false.B)

        state := s_idle
      }
    }
  }
}

