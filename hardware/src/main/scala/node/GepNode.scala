package heteacc.node

import chisel3._
import chisel3.util._
import chisel3.util.experimental.BoringUtils
import org.scalatest.{FlatSpec, Matchers}
import heteacc.config._
import chisel3.Module
import heteacc.interfaces._
import util._
import chipsalliance.rocketchip.config._


class GepNodeIO(NumIns: Int, NumOuts: Int)
               (implicit p: Parameters)
  extends HandShakingIONPS(NumOuts)(new DataBundle) {

  // Inputs should be fed only when Ready is HIGH
  // Inputs are always latched.
  // If Ready is LOW; Do not change the inputs as this will cause a bug
  val baseAddress = Flipped(Decoupled(new DataBundle()))
  val idx = Vec(NumIns, Flipped(Decoupled(new DataBundle())))

}



class GepNode(NumIns: Int, NumOuts: Int, ID: Int)
             (ElementSize: Int, ArraySize: List[Int])
             (implicit p: Parameters,
              name: sourcecode.Name,
              file: sourcecode.File)
  extends HandShakingNPS(NumOuts, ID)(new DataBundle)(p) {
  override lazy val io = IO(new GepNodeIO(NumIns, NumOuts))
  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)
  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "

  /*===========================================*
   *            Registers                      *
   *===========================================*/
  // Addr Inputs
  val base_addr_R = RegInit(DataBundle.default)
  val base_addr_valid_R = RegInit(false.B)

  // Index 1 input
  val idx_R = Seq.fill(NumIns)(RegInit(DataBundle.default))
  val idx_valid_R = Seq.fill(NumIns)(RegInit(false.B))


  val s_IDLE :: s_COMPUTE :: Nil = Enum(2)
  val state = RegInit(s_IDLE)

  //We support only geps with 1 or 2 inputs
  assert(NumIns <= 2)

  /*==========================================*
   *           Predicate Evaluation           *
   *==========================================*/

  val predicate = IsEnable()

  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/

  io.baseAddress.ready := ~base_addr_valid_R
  when(io.baseAddress.fire) {
    base_addr_R <> io.baseAddress.bits
    base_addr_valid_R := true.B
  }

  for (i <- 0 until NumIns) {
    io.idx(i).ready := ~idx_valid_R(i)
    when(io.idx(i).fire) {
      idx_R(i) <> io.idx(i).bits
      idx_valid_R(i) := true.B
    }
  }

  val seek_value =
    if (ArraySize.isEmpty) {
      idx_R(0).data * ElementSize.U
    } else if (ArraySize.length == 1) {
      (idx_R(0).data * ArraySize(0).U) + (idx_R(1).data * ElementSize.U)
    }
    else {
      0.U
    }

  val data_out = base_addr_R.data + seek_value
  // val data_out = base_addr_R.data + idx_R(0).data
  // Wire up Outputs
  for (i <- 0 until NumOuts) {
    io.Out(i).bits.data := data_out
    // io.Out(i).bits.predicate := predicate
    // io.Out(i).bits.taskID := base_addr_R.taskID
    io.Out(i).bits.predicate := predicate
    io.Out(i).bits.taskID := base_addr_R.taskID
  }


  /*============================================*
   *            STATES                          *
   *============================================*/

  switch(state) {
    is(s_IDLE) {
      when(enable_valid_R && base_addr_valid_R && idx_valid_R.reduce(_ & _)) {
        ValidOut()
        state := s_COMPUTE
      }
    }
    is(s_COMPUTE) {
      when(IsOutReady()) {
        // Reset output
        idx_R.foreach(_ := DataBundle.default)
        base_addr_R := DataBundle.default

        idx_valid_R.foreach(_ := false.B)
        base_addr_valid_R := false.B

        // Reset state
        state := s_IDLE

        // Reset output
        Reset()
        if (log) {
          printf(p"[LOG] [${module_name}] [TID: ${enable_R.taskID}] [GEP] [${node_name}] " +
            p"[Pred: ${enable_R.control}][Out: 0x${Hexadecimal(data_out)}] [Cycle: ${cycleCount}]\n")
        }
      }
    }
  }
}



class GepNodeWithoutStateIO(NumIns: Int, NumOuts: Int)
               (implicit p: Parameters)
  extends HandShakingIONPS(NumOuts)(new DataBundle) {

  // Inputs should be fed only when Ready is HIGH
  // Inputs are always latched.
  // If Ready is LOW; Do not change the inputs as this will cause a bug
  val baseAddress = Flipped(Decoupled(new DataBundle()))
  val idx = Vec(NumIns, Flipped(Decoupled(new DataBundle())))

}



class GepNodeWithoutState(NumIns: Int, NumOuts: Int, ID: Int)
             (ElementSize: Int, ArraySize: List[Int])
             (implicit p: Parameters,
              name: sourcecode.Name,
              file: sourcecode.File)
  extends HandShakingNPS(NumOuts, ID)(new DataBundle)(p) {
  override lazy val io = IO(new GepNodeWithoutStateIO(NumIns, NumOuts))
  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)
  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "

  /*===========================================*
   *            Registers                      *
   *===========================================*/
  // Addr Inputs
  private val join = Module(new Join())
  private val oehb = Module(new OEHB(0))

  join.pValid(0) := io.baseAddress.valid
  join.pValid(1) := io.idx(0).valid
  io.baseAddress.ready := join.ready(0)
  io.idx(0).ready := join.ready(1)
  join.nReady := oehb.dataIn.ready
  

  oehb.dataIn.bits := DontCare
  oehb.dataIn.valid := join.valid
  

  oehb.dataOut.ready := io.Out.map(_.ready).reduce(_ && _)
  for (i <- 0 until NumOuts) {
    // oehb.dataOut.ready := io.Out(i).ready
    io.Out(i).valid := oehb.dataOut.valid
  }
  

  val seek_value =
    if (ArraySize.isEmpty) {
      io.idx(0).bits.data * ElementSize.U
    } else if (ArraySize.length == 1) {
      (io.idx(0).bits.data * ArraySize(0).U) + (io.idx(1).bits.data * ElementSize.U)
    }
    else {
      0.U
    }

  val data_out = io.baseAddress.bits.data + seek_value
  // val data_out = io.baseAddress.bits.data + io.idx(0).bits.data
  val predicate = io.enable.bits.control
  val taskID = io.enable.bits.taskID
  
  io.Out.foreach(_.bits := DataBundle(data_out, taskID, predicate))
  
  when(IsOutReady()) {
    Reset()
    if (log) {
      printf(p"[LOG] [${module_name}] [TID: ${enable_R.taskID}] [GEP] [${node_name}] " +
        p"[Pred: ${enable_R.control}][Out: 0x${Hexadecimal(data_out)}] [Cycle: ${cycleCount}]\n")
    }
  }
 

}