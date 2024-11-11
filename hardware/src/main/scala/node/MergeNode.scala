package heteacc.node

import chisel3._
import chisel3.Module
import chipsalliance.rocketchip.config._
import heteacc.interfaces._
import util._
import utility.UniformPrintfs
import heteacc.config._
import chisel3.util.experimental.BoringUtils


abstract class MergeNodeIO(val NumInputs: Int = 2, val NumOutputs: Int = 1, val ID: Int, Debug: Boolean = false, GuardVal: Int = 0)
                            (implicit val p: Parameters)
  extends MultiIOModule with HasAccelParams with UniformPrintfs {

  val io = IO(new Bundle {
    //Control signal
    val enable = Flipped(Decoupled(new ControlBundle))

    // Vector input
    val InData = Vec(NumInputs, Flipped(Decoupled(new DataBundle)))

    // Predicate mask comming from the basic block
    val Mask = Flipped(Decoupled(UInt(NumInputs.W)))

    //Output
    val Out = Vec(NumOutputs, Decoupled(new DataBundle))
  })
}


 
class MergeNode(NumInputs: Int = 2, NumOutputs: Int = 1, ID: Int, Res: Boolean = false, Induction: Boolean = false,
                  Debug: Boolean = false, val GuardVals: Seq[Int] = List())
                 (implicit p: Parameters,
                  name: sourcecode.Name,
                  file: sourcecode.File)
  extends MergeNodeIO(NumInputs, NumOutputs, ID)(p)
    with HasAccelShellParams
    {

  // Printf debugging
  override val printfSigil = "Node (PHIFast) ID: " + ID + " "
  val (cycleCount, _) = Counter(true.B, 32 * 1024)


  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize

  // Data Inputs
  val in_data_R = RegInit(VecInit(Seq.fill(NumInputs)(DataBundle.default)))
  val in_data_valid_R = RegInit(VecInit(Seq.fill(NumInputs)(false.B)))

  // Enable Inputs
  val enable_R = RegInit(ControlBundle.default)
  val enable_valid_R = RegInit(false.B)

  // Mask Input
  val mask_R = RegInit(0.U(NumInputs.W))
  val mask_valid_R = RegInit(false.B)

  //Output register
  val s_idle :: s_fire :: s_not_predicated :: Nil = Enum(3)
  val state = RegInit(s_idle)


  // Latching output data
  val out_valid_R = Seq.fill(NumOutputs)(RegInit(false.B))

  val fire_R = Seq.fill(NumOutputs)(RegInit(false.B))


  // Latching Mask value
  io.Mask.ready := ~mask_valid_R
  when(io.Mask.fire()) {
    mask_R := io.Mask.bits
    mask_valid_R := true.B
  }

  // Latching enable value
  io.enable.ready := ~enable_valid_R
  when(io.enable.fire()) {
    enable_R <> io.enable.bits
    enable_valid_R := true.B
  }


  for (i <- 0 until NumInputs) {
    io.InData(i).ready := ~in_data_valid_R(i)
    when(io.InData(i).fire) {
      in_data_R(i) <> io.InData(i).bits
      in_data_valid_R(i) := true.B
    }
  }

  val sel =
    if (Res == false) {
      OHToUInt(mask_R)
    }
    else {
      OHToUInt(Reverse(mask_R))
    }

  // when(sel === 0.U){
  //   io.InData(1).ready := true.B
  //   in_data_R(1).data := 0.U
  //   in_data_valid_R(1) := true.B
  // }
  val select_input = in_data_R(sel).data
  val select_predicate = in_data_R(sel).predicate

  val enable_input = enable_R.control

  val task_input = (io.enable.bits.taskID | enable_R.taskID)

  for (i <- 0 until NumOutputs) {
    when(io.Out(i).fire) {
      fire_R(i) := true.B
      out_valid_R(i) := false.B
    }
  }

  //Getting mask for fired nodes
  val fire_mask = (fire_R zip io.Out.map(_.fire)).map { case (a, b) => a | b }

  def IsInputValid(): Bool = {
    in_data_valid_R.reduce(_ & _)
  }

  def isInFire(): Bool = {
    enable_valid_R && IsInputValid() && enable_R.control && state === s_idle
  }

  //***************************BORE Connection*************************************



  for (i <- 0 until NumOutputs) {
    //TODO: enable for comapring

    io.Out(i).bits := in_data_R(sel)
    io.Out(i).valid := out_valid_R(i)
  }


  switch(state) {
    is(s_idle) {
      when(enable_valid_R && IsInputValid()) {
        //Make outputs valid
        out_valid_R.foreach(_ := true.B)
        when(enable_R.control) {
          //*********************************
          state := s_fire
          //********************************
          //Print output


          //*****************************************************************
          if (log) {
            printf(p"[LOG] [${module_name}] [sel: ${sel}] [PHI] " +
              p"[${node_name}] [Pred: ${enable_R.control}] [Out: ${in_data_R(sel).data}] [Cycle: ${cycleCount}]\n")
          }
        }.otherwise {
          state := s_not_predicated
          //Print output
          if (log) {
            printf(p"[LOG] [${module_name}] [TID: ${io.InData(sel).bits.taskID}] [PHI] " +
              p"[${node_name}] [Pred: ${enable_R.control}] [Out: ${in_data_R(sel).data}] [Cycle: ${cycleCount}]\n")
          }
        }
      }
    }
    is(s_fire) {
      when(fire_mask.reduce(_ & _)) {

        /**
         * @note: In this case whenever all the GEP is fired we
         *        restart all the latched values. But it may be cases
         *        that because of pipelining we have latched an interation ahead
         *        and if we may reset the latches values we lost the value.
         *        I'm not sure when this case can happen!
         */
        in_data_R.foreach(_ := DataBundle.default)
        in_data_valid_R.foreach(_ := false.B)

        mask_R := 0.U
        mask_valid_R := false.B

        enable_R := ControlBundle.default
        enable_valid_R := false.B

        fire_R.foreach(_ := false.B)


        state := s_idle
        if (log) {
            printf(p"[LOG] [${module_name}] [TID: ${io.InData(sel).bits.taskID}] [PHI] " +
              p"[${node_name}] [Pred: ${enable_R.control}] [Out: ${in_data_R(sel).data}] [Cycle: ${cycleCount}]\n")
        }
      }

    }
    is(s_not_predicated) {
      io.Out.map(_.bits) foreach (_.data := 0.U)
      io.Out.map(_.bits) foreach (_.predicate := false.B)
      io.Out.map(_.bits) foreach (_.taskID := task_input)

      when(fire_mask.reduce(_ & _)) {
        in_data_R.foreach(_ := DataBundle.default)
        in_data_valid_R.foreach(_ := false.B)

        mask_R := 0.U
        mask_valid_R := false.B

        enable_R := ControlBundle.default
        enable_valid_R := false.B

        fire_R.foreach(_ := false.B)

        state := s_idle

      }
    }
  }

}
class MergeNodeWithMaskIO(NumInputs: Int, NumOutputs: Int, Debug: Boolean)
                         (implicit p: Parameters)
  extends HandShakingIONPS(NumOutputs, Debug)(new DataBundle) {

  override val enable = Flipped(Decoupled(new ControlBundle))
  override val Out = Vec(NumOutputs, Decoupled(new DataBundle))

  val Mask = Flipped(Decoupled(UInt(NumInputs.W)))
  val InData = Vec(NumInputs, Flipped(Decoupled(new DataBundle)))
}

class MergeNodeWithMask(NumInputs: Int = 2, 
                        NumOutputs: Int = 1, 
                        ID: Int, 
                        Debug: Boolean = false)
                       (implicit p: Parameters, 
                        name: sourcecode.Name, 
                        file: sourcecode.File)
  extends HandShakingNPS(NumOutputs, ID, Debug)(new DataBundle)(p)
    with HasAccelShellParams {

  override lazy val io = IO(new MergeNodeWithMaskIO(NumInputs, NumOutputs, Debug))

  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  private val join = Module(new Join(NumInputs))
  private val oehb = Module(new OEHB(0))

  val mask_R = RegInit(0.U(NumInputs.W))
  val mask_valid_R = RegInit(false.B)

  io.Mask.ready := ~mask_valid_R
  when(io.Mask.fire()) {
    mask_R := io.Mask.bits
    mask_valid_R := true.B
  }

  for (i <- 0 until NumInputs) {
    join.pValid(i) := io.InData(i).valid
    io.InData(i).ready := join.ready(i)
  }

  join.nReady := oehb.dataIn.ready

  // 使用 Mask 选择输入数据，并传递数据部分给 OEHB
  val selectedInputIndex = OHToUInt(mask_R)
  val selectedInput = io.InData(selectedInputIndex).bits.data

  oehb.dataIn.bits := selectedInput  // 仅传递数据部分
  oehb.dataIn.valid := join.valid

  oehb.dataOut.ready := io.Out.map(_.ready).reduce(_ && _)

  for (i <- 0 until NumOutputs) {
    io.Out(i).valid := oehb.dataOut.valid
    io.Out(i).bits := DataBundle(oehb.dataOut.bits, taskID = 0.U, predicate = true.B)
  }

  override def IsOutReady(): Bool = io.Out.map(_.ready).reduce(_ && _)

  override def Reset(): Unit = {
    oehb.dataIn.valid := false.B
    join.nReady := false.B
    mask_valid_R := false.B
    mask_R := 0.U
  }

  when(IsOutReady()) {
    if (Debug) {
      printf(p"[LOG] [${module_name}] [TID: ${io.enable.bits.taskID}] [MERGE] [Name: ${node_name}] " +
        p"[ID: ${ID}] [Mask: ${mask_R}] [Cycle: ${cycleCount}]\n")
    }
    Reset()
  }
}