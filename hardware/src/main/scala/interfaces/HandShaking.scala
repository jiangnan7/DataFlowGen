package heteacc.interfaces

import chisel3._
import chisel3.util._
import chipsalliance.rocketchip.config._
//import heteacc.node.{IsAlias}
import utility._
import Constants._
import utility.UniformPrintfs
import chipsalliance.rocketchip.config._
import heteacc.config._


/*===========================================================
=            Handshaking IO definitions                     =
===========================================================*/

/**
  * @note
  * There are three types of handshaking:
  * 1)   There is no ordering -> (No PredOp/ No SuccOp)
  * it has only vectorized output
  * @note HandshakingIONPS
  * @todo Put special case for singl output vs two outputs
  *       2)  There is ordering    -> (PredOp/ SuccOp)
  *       vectorized output/succ/pred
  * @note HandshakingIOPS
  *       3)  There is vectorized output + vectorized input
  *       No ordering
  * @todo needs to be implimented
  * @note HandshakingFusedIO
  *       4)  Control handshaking -> The only input is enable signal
  * @note HandshakingCtrl
  *       5) Control handshaking (PHI) -> There is mask and enable signal
  * @note HandshakingCtrlPhi
  *
  */

/**
  * @note Type1
  *       Handshaking IO with no ordering.
  * @note IO Bundle for Handshaking
  * @param NumOuts Number of outputs
  *
  */
class HandShakingIONPS[T <: Data](val NumOuts: Int, val Debug: Boolean = false)(gen: T)(implicit p: Parameters)
  extends AccelBundle with HasAccelParams {
  // Predicate enable
  val enable = Flipped(Decoupled(new ControlBundle))
  // Output IO
  val Out    = Vec(NumOuts, Decoupled(gen))
  /*hs
  val LogCheck = if (Debug) Some (Decoupled(new CustomDataBundle(UInt (32.W)))) else None
  val LogCheckAddr = if (Debug) Some (Decoupled(new CustomDataBundle(UInt (32.W)))) else None
  hs*/

}



/**
  * @note Type2
  *       Handshaking IO.
  * @note IO Bundle for Handshaking
  *       PredOp: Vector of RvAckIOs
  *       SuccOp: Vector of RvAckIOs
  *       Out      : Vector of Outputs
  * @param NumPredOps Number of parents
  * @param NumSuccOps Number of successors
  * @param NumOuts    Number of outputs
  *
  *
  */
class HandShakingIOPS[T <: Data](val NumPredOps: Int,
                                 val NumSuccOps: Int,
                                 val NumOuts: Int, val Debug: Boolean = false)(gen: T)(implicit p: Parameters)
  extends AccelBundle( )(p) {
  // Predicate enable
  val enable = Flipped(Decoupled(new ControlBundle))
  // Predecessor Ordering
  val PredOp = Vec(NumPredOps, Flipped(Decoupled(new ControlBundle)))
  // Successor Ordering
  val SuccOp = Vec(NumSuccOps, Decoupled(new ControlBundle( )))
  // Output IO
  val Out    = Vec(NumOuts, Decoupled(gen))
  // Logging port
  /*hs
  val LogCheck = if (Debug) Some(Decoupled(new CustomDataBundle(UInt(2.W)))) else None
  hs*/

}

/**
  * @note Type3
  *       Handshaking IO with no ordering.
  * @note IO Bundle for Handshaking
  * @param NumIns  Number of Inputs
  * @param NumOuts Number of outputs
  *
  */
class HandShakingFusedIO[T <: Data](val NumIns: Int, val NumOuts: Int,
					val Debug: Boolean = false)(gen: T)(implicit p: Parameters)
  extends AccelBundle( )(p) {
  // Predicate enable
  val enable = Flipped(Decoupled(new ControlBundle))
  // Input IO
  val In     = Flipped(Vec(NumIns, Decoupled(gen)))
  // Output IO
  val Out    = Vec(NumOuts, Decoupled(gen))
  /*hs
  val LogCheck = if (Debug) Some(Decoupled(new CustomDataBundle(UInt(2.W)))) else None
  hs*/

}

class HandShakingCtrlNPS(val NumOuts: Int,
                         val ID: Int)(implicit val p: Parameters)
  extends Module with HasAccelParams with UniformPrintfs {

  lazy val io = IO(new HandShakingIONPS(NumOuts)(new ControlBundle))

  /*=================================
  =            Registers            =
  =================================*/
  // Extra information
  val token    = RegInit(0.U)
  val nodeID_R = RegInit(ID.U)

  // Enable
  val enable_R       = RegInit(ControlBundle.default)
  val enable_valid_R = RegInit(false.B)

  // Output Handshaking
  val out_ready_R = RegInit(VecInit(Seq.fill(NumOuts)(false.B)))
  val out_valid_R = RegInit(VecInit(Seq.fill(NumOuts)(false.B)))

  // Wire
  // val out_ready_W   = Wire(Vec(Seq.fill(NumOuts)(false.B)))

  /*============================*
   *           Wiring           *
   *============================*/

  // Wire up OUT READYs and VALIDs
  for (i <- 0 until NumOuts) {
    io.Out(i).valid := out_valid_R(i)
    when(io.Out(i).fire( )) {
      // Detecting when to reset
      out_ready_R(i) := io.Out(i).ready
      // Propagating output
      out_valid_R(i) := false.B
    }
  }

  // Wire up enable READY and VALIDs
  io.enable.ready := ~enable_valid_R
  when(io.enable.fire( )) {
    enable_valid_R := io.enable.valid
    enable_R := io.enable.bits
  }

  /*===================================*
   *            Helper Checks          *
   *===================================*/
  def IsEnable(): Bool = {
    enable_R.control
  }

  def IsEnableValid(): Bool = {
    enable_valid_R
  }

  def ResetEnable(): Unit = {
    enable_valid_R := false.B
  }

  // OUTs
  def IsOutReady(): Bool = {
    out_ready_R.asUInt.andR
  }

  def IsOutValid(): Bool = {
    //    out_valid_R.asUInt.andR
    if (NumOuts == 0) {
      return true.B
    } else {
      out_valid_R.reduceLeft(_ && _)
    }
  }

  def ValidOut(): Unit = {
    out_valid_R.foreach(_ := true.B)
  }

  def InvalidOut(): Unit = {
    out_valid_R.foreach(_ := false.B)
  }

  def Reset(): Unit = {
    out_ready_R.foreach(_ := false.B)
    enable_valid_R := false.B
  }
}
class HandShakingCtrlMaskIO(val NumInputs: Int,
                            val NumOuts: Int,
                            val NumPhi: Int, val Debug: Boolean = false)(implicit p: Parameters)
  extends AccelBundle( )(p) {

  // Output IO
  val MaskBB = Vec(NumPhi, Decoupled(UInt(NumInputs.W)))
  val Out    = Vec(NumOuts, Decoupled(new ControlBundle))
  /*hs
  val LogCheck = if (Debug) Some(Decoupled(new CustomDataBundle(UInt(2.W)))) else None
  hs*/
}

class HandShakingCtrlMask(val NumInputs: Int,
                          val NumOuts: Int,
                          val NumPhi: Int,
                          val BID: Int,
			  val Debug: Boolean = false)(implicit val p: Parameters)
  extends Module with HasAccelParams with UniformPrintfs {

  lazy val io = IO(new HandShakingCtrlMaskIO(NumInputs, NumOuts, NumPhi,Debug))

  /*=================================
  =            Registers            =
  =================================*/
  // Extra information
  val token    = RegInit(0.U)
  val nodeID_R = RegInit(BID.U)

  // Output Handshaking
  val out_ready_R = RegInit(VecInit(Seq.fill(NumOuts)(false.B)))
  val out_valid_R = RegInit(VecInit(Seq.fill(NumOuts)(false.B)))

  // Mask handshaking
  val mask_ready_R = Seq.fill(NumPhi)(RegInit(false.B))
  val mask_valid_R = Seq.fill(NumPhi)(RegInit(false.B))

  /*============================*
   *           Wiring           *
   *============================*/

  // Wire up OUT READYs and VALIDs
  for (i <- 0 until NumOuts) {
    io.Out(i).valid := out_valid_R(i)
    when(io.Out(i).fire( )) {
      // Detecting when to reset
      out_ready_R(i) := io.Out(i).ready
      // Propagating output
      out_valid_R(i) := false.B
    }
  }

  // Wire up MASK Readys and Valids
  for (i <- 0 until NumPhi) {
    io.MaskBB(i).valid := mask_valid_R(i)
    when(io.MaskBB(i).fire( )) {
      // Detecting when to reset
      mask_ready_R(i) := io.MaskBB(i).ready
      // Propagating mask
      mask_valid_R(i) := false.B
    }

  }

  /*===================================*
   *            Helper Checks          *
   *===================================*/

  /*hs
  if(Debug){
    io.LogCheck.get.valid := false.B
    io.LogCheck.get.bits := DataBundle.default
  }

  def CaptureLog(data: UInt): Unit = {
    if (Debug) {
      io.LogCheck.get.bits := DataBundle(data)
      io.LogCheck.get.valid := true.B
    }
  }
  hs*/

  // OUTs
  def IsOutReady(): Bool = {
    out_ready_R.asUInt.andR
  }

  def IsMaskReady(): Bool = {
    if (NumPhi == 0) {
      return true.B
    } else {
      VecInit(mask_ready_R).asUInt.andR
    }
  }

  def IsOutValid(): Bool = {
    out_valid_R.asUInt.andR
  }

  def IsMaskValid(): Bool = {
    if (NumPhi == 0) {
      return true.B
    } else {
      VecInit(mask_valid_R).asUInt.andR
    }
  }

  def ValidOut(): Unit = {
    out_valid_R := VecInit(Seq.fill(NumOuts)(true.B))
    mask_valid_R.foreach {
      _ := true.B
    }
  }

  def InvalidOut(): Unit = {
    out_valid_R := VecInit(Seq.fill(NumOuts)(false.B))
    mask_valid_R.foreach {
      _ := false.B
    }
  }

  def Reset(): Unit = {
    out_ready_R := VecInit(Seq.fill(NumOuts)(false.B))
    mask_ready_R.foreach {
      _ := false.B
    }
  }
}











/*==============================================================
=            Handshaking Implementations                       =
==============================================================*/

/**
  * @brief Handshaking between data nodes with no ordering.
  * @note Sets up base registers and hand shaking registers
  * @param NumOuts Number of outputs
  * @param ID      Node id
  * @return Module
  */

class HandShakingNPS[T <: Data](val NumOuts: Int,
                                val ID: Int, val Debug: Boolean = false)(gen: T)(implicit val p: Parameters)
  extends Module with HasAccelParams with UniformPrintfs {

  lazy val io = IO(new HandShakingIONPS(NumOuts, Debug)(gen))

  /*=================================
  =            Registers            =
  =================================*/
  // Extra information
  val token    = RegInit(0.U)
  val nodeID_R = RegInit(ID.U)

  // Enable
  val enable_R       = RegInit(ControlBundle.default)
  val enable_valid_R = RegInit(false.B)

  // Output Handshaking
  val out_ready_R = Seq.fill(NumOuts)(RegInit(false.B))
  val out_valid_R = Seq.fill(NumOuts)(RegInit(false.B))

  /*============================*
   *           Wiring           *
   *============================*/

  // Wire up OUT READYs and VALIDs
  for (i <- 0 until NumOuts) {
    io.Out(i).valid := out_valid_R(i)
    when(io.Out(i).fire( )) {
      // Detecting when to reset
      out_ready_R(i) := io.Out(i).ready
      // Propagating output
      out_valid_R(i) := false.B
    }
  }


  // Wire up enable READY and VALIDs
  io.enable.ready := ~enable_valid_R
  when(io.enable.fire( )) {
    enable_valid_R := io.enable.valid
    enable_R <> io.enable.bits
  }



  def IsEnable(): Bool = {
    enable_R.control
  }

  def IsEnableValid(): Bool = {
    enable_valid_R
  }

  def ResetEnable(): Unit = {
    enable_valid_R := false.B
  }

  // OUTs
  def IsOutReady(): Bool = {
    if (NumOuts == 0) {
      return true.B
    } else {
      val fire_mask = (out_ready_R zip io.Out.map(_.fire)).map { case (a, b) => a | b }
      fire_mask reduce {_ & _}
    }
  }
  /*hs
  def IsDebugReady(): Bool = {
    if(Debug){
      io.LogCheck.get.ready
    } else {
      return true.B
    }
  }
  hs*/
  def IsOutValid(): Bool = {
    //    out_valid_R.asUInt.andR
    if (NumOuts == 0) {
      return true.B
    } else {
      out_valid_R.reduceLeft(_ && _)
    }
  }

  def ValidOut(): Unit = {
    (out_valid_R zip io.Out.map(_.fire)).foreach{ case (a,b) => a := b ^ true.B}
  }

  def InvalidOut(): Unit = {
    out_valid_R.foreach(_ := false.B)
  }

  def Reset(): Unit = {
    out_ready_R.foreach(_ := false.B)
    enable_valid_R := false.B
  }
}



class HandShakingFused[T <: PredicateT](val NumIns: Int, val NumOuts: Int,
                                        val ID: Int, val Debug: Boolean = false)(gen: T)(implicit val p: Parameters)
  extends Module with HasAccelParams with UniformPrintfs {

  lazy val io = IO(new HandShakingFusedIO(NumIns, NumOuts ,Debug)(new DataBundle))

  /*=================================
  =            Registers            =
  =================================*/
  // Extra information
  val token    = RegInit(0.U)
  val nodeID_R = RegInit(ID.U)

  // Enable
  val enable_R       = RegInit(ControlBundle.default)
  val enable_valid_R = RegInit(false.B)

  // Input Handshaking
  val in_predicate_W = WireInit(VecInit(Seq.fill(NumIns) {
    false.B
  }))
  val in_valid_R     = RegInit(VecInit(Seq.fill(NumIns) {
    false.B
  }))

  // Seq of registers. This has to be an array and not a vector
  // When vector it will try to instantiate registers; not possible since only
  // type description available here.
  // Do not try to dynamically dereference ops.
  val InRegs = for (i <- 0 until NumIns) yield {
    val InReg = Reg(gen)
    InReg
  }


  // Wire
  val out_valid_W = WireInit(VecInit(Seq.fill(NumOuts) {
    false.B
  }))
  val out_ready_W = WireInit(VecInit(Seq.fill(NumOuts) {
    false.B
  }))

  /*============================*
   *           Wiring           *
   *============================*/

  // Wire up OUT READYs and VALIDs
  for (i <- 0 until NumOuts) {
    io.Out(i).valid := out_valid_W(i)
    out_ready_W(i) := io.Out(i).ready
  }


  // Wire up enable READY and VALIDs
  for (i <- 0 until NumIns) {
    io.In(i).ready := ~in_valid_R(i)
    in_predicate_W(i) := InRegs(i).predicate
    when(io.In(i).fire( )) {
      in_valid_R(i) := io.In(i).valid
      InRegs(i) := io.In(i).bits
      //InRegs(i).valid := io.In(i).valid
      InRegs(i).predicate := io.In(i).bits.predicate
    }
  }

  // Wire up enable READY and VALIDs
  io.enable.ready := ~enable_valid_R
  when(io.enable.fire( )) {
    enable_valid_R := io.enable.valid
    enable_R <> io.enable.bits
  }

  /*===================================*
   *            Helper Checks          *
   *===================================*/

  /*hs
  if(Debug){
    io.LogCheck.get.valid := false.B
    io.LogCheck.get.bits := DataBundle.default
  }

  def CaptureLog(data: UInt): Unit = {
    if (Debug) {
      io.LogCheck.get.bits := DataBundle(data)
      io.LogCheck.get.valid := true.B
    }
  }

  hs*/

  def IsEnable(): Bool = {
    enable_R.control
  }

  def IsEnableValid(): Bool = {
    enable_valid_R
  }

  def ResetEnable(): Unit = {
    enable_valid_R := false.B
  }

  // Predicate.
  def IsInPredicate(): Bool = {
    in_predicate_W.asUInt.andR
  }

  // Ins
  def IsInValid(): Bool = {
    in_valid_R.asUInt.andR
  }

  def ValidIn(): Unit = {
    in_valid_R := VecInit(Seq.fill(NumOuts) {
      true.B
    })
  }

  def printInValid(): Unit = {
    for (i <- 0 until NumIns) yield {
      if (i != (NumIns - 1)) {
        printf("\"In(%x)\" : %x ,", i.U, in_valid_R(i))
      } else {
        printf("\"In(%x)\" : %x ", i.U, in_valid_R(i))
      }
    }
  }

  def InvalidIn(): Unit = {
    in_valid_R := VecInit(Seq.fill(NumOuts) {
      false.B
    })
  }

  // OUTs
  def IsOutReady(): Bool = {
    out_ready_W.asUInt.andR
  }

  def IsOutValid(): Bool = {
    out_valid_W.asUInt.andR
  }

  def ValidOut(): Unit = {
    out_valid_W := VecInit(Seq.fill(NumOuts) {
      true.B
    })
  }

  def InvalidOut(): Unit = {
    out_valid_W := VecInit(Seq.fill(NumOuts) {
      false.B
    })
  }

  def Reset(): Unit = {
    enable_valid_R := false.B
    in_valid_R := VecInit(Seq.fill(NumIns) {
      false.B
    })
  }
}