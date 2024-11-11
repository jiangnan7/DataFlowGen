package heteacc.fpu

import chisel3._
import chisel3.util._
import heteacc.interfaces._
import chipsalliance.rocketchip.config._
import heteacc.config._
import utility._

/**
 * [FPComputeNodeIO description]
 */
class FPComputeNodeIO(NumOuts: Int)
                     (implicit p: Parameters)
  extends HandShakingIONPS(NumOuts)(new DataBundle) {
  // LeftIO: Left input data for computation
  val LeftIO = Flipped(Decoupled(new DataBundle()))

  // RightIO: Right input data for computation
  val RightIO = Flipped(Decoupled(new DataBundle()))

}


/**
 * [FPComputeNode description]
 */
class FPComputeNode(NumOuts: Int, ID: Int, opCode: String)
                   (t: FType)
                   (implicit p: Parameters,
                    name: sourcecode.Name,
                    file: sourcecode.File)
  extends HandShakingNPS(NumOuts, ID)(new DataBundle())(p) {
  override lazy val io = IO(new FPComputeNodeIO(NumOuts))

  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize

  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "
  //  override val printfSigil = "Node (COMP - " + opCode + ") ID: " + ID + " "
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  val iter_counter = Counter(32 * 1024)

  /*===========================================*
   *            Registers                      *
   *===========================================*/
  // Left Input
  val left_R = RegInit(DataBundle.default)
  val left_valid_R = RegInit(false.B)

  // Right Input
  val right_R = RegInit(DataBundle.default)
  val right_valid_R = RegInit(false.B)

  val task_ID_R = RegNext(next = enable_R.taskID)

  //Output register
  val s_IDLE :: s_COMPUTE :: Nil = Enum(2)
  val state = RegInit(s_IDLE)

  val FU = Module(new FPUALU(64, opCode, t))

  val out_data_R = RegNext(Mux(enable_R.control, FU.io.out, 0.U), init = 0.U)
  val predicate = Mux(enable_valid_R, enable_R.control, io.enable.bits.control)
  val taskID = Mux(enable_valid_R, enable_R.taskID, io.enable.bits.taskID)

  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/

  //Instantiate ALU with selected code. IEEE ALU. IEEE in/IEEE out
  FU.io.in1 := left_R.data
  FU.io.in2 := right_R.data

  io.LeftIO.ready := ~left_valid_R
  when(io.LeftIO.fire) {
    left_R <> io.LeftIO.bits
    left_valid_R := true.B
  }

  io.RightIO.ready := ~right_valid_R
  when(io.RightIO.fire) {
    right_R <> io.RightIO.bits
    right_valid_R := true.B
  }

  // Wire up Outputs

  io.Out.foreach(_.bits := DataBundle(out_data_R, taskID, predicate))

  /*============================================*
   *            State Machine                   *
   *============================================*/
  switch(state) {
    is(s_IDLE) {
      when(enable_valid_R && left_valid_R && right_valid_R) {
        io.Out.foreach(_.bits := DataBundle(FU.io.out, enable_R.taskID, predicate))
        io.Out.foreach(_.valid := true.B)
        ValidOut()
        left_valid_R := false.B
        right_valid_R := false.B
        state := s_COMPUTE
        iter_counter.inc()
        if (log) {
          printf(p"[LOG] [${module_name}] [TID: ${task_ID_R}] [FPCompute] [${node_name}] " +
            p"[Pred: ${enable_R.control}] " +
            p"[Iter: ${iter_counter.value}] " +
            p"[In(0): ${Decimal(left_R.data)}] " +
            p"[In(1) ${Decimal(right_R.data)}] " +
            p"[Out: ${Decimal(FU.io.out)}] " +
            p"[OpCode: ${opCode}] " +
            p"[Cycle: ${cycleCount}]\n")
        }
      }
    }
    is(s_COMPUTE) {
      when(IsOutReady()) {
        state := s_IDLE
        out_data_R := 0.U
        //Reset output
        Reset()
      }
    }
  }

}