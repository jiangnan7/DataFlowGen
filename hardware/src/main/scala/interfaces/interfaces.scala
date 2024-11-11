package heteacc.interfaces


import chisel3._
import chisel3.util.Decoupled
import utility.Constants._
import heteacc.config._
import chipsalliance.rocketchip.config._
import utility.UniformPrintfs

import scala.collection.immutable.ListMap


trait RouteID extends AccelBundle {
  val RouteID = UInt(glen.W)
}

trait TaskID extends AccelBundle {
  val taskID = UInt(tlen.W)
}
trait PredicateT extends AccelBundle {
  val predicate = Bool()
}

class NewDataBundle(implicit p: Parameters) extends AccelBundle {
  // Data packet
  val data = UInt(xlen.W)

}

object NewDataBundle {

  def apply(data: UInt = 0.U)(implicit p: Parameters): NewDataBundle = {
    val wire = Wire(new NewDataBundle)
    wire.data := data
    wire
  }


  def default(implicit p: Parameters): NewDataBundle = {
    val wire = Wire(new NewDataBundle)
    wire.data := 0.U
    wire
  }

  def active(data: UInt = 0.U)(implicit p: Parameters): NewDataBundle = {
    val wire = Wire(new NewDataBundle)
    wire.data := data
    wire
  }

  def deactivate(data: UInt = 0.U)(implicit p: Parameters): NewDataBundle = {
    val wire = Wire(new NewDataBundle)
    wire.data := data
    wire
  }
}

object DataBundle {

  def apply(data: UInt = 0.U, taskID: UInt = 0.U)(implicit p: Parameters): DataBundle = {
    val wire = Wire(new DataBundle)
    wire.data := data
    wire.predicate := true.B
    wire.taskID := taskID
    wire
  }

  def apply(data: UInt, taskID: UInt, predicate: UInt)(implicit p: Parameters): DataBundle = {
    val wire = Wire(new DataBundle)
    wire.data := data
    wire.predicate := predicate
    wire.taskID := taskID
    wire
  }


  def default(implicit p: Parameters): DataBundle = {
    val wire = Wire(new DataBundle)
    wire.data := 0.U
    wire.predicate := false.B
    wire.taskID := 0.U
    wire
  }

  def active(data: UInt = 0.U)(implicit p: Parameters): DataBundle = {
    val wire = Wire(new DataBundle)
    wire.data := data
    wire.predicate := true.B
    wire.taskID := 0.U
    wire
  }

  def deactivate(data: UInt = 0.U)(implicit p: Parameters): DataBundle = {
    val wire = Wire(new DataBundle)
    wire.data := data
    wire.predicate := false.B
    wire.taskID := 0.U
    wire
  }
}

class TypBundle(implicit p: Parameters) extends PredicateT with TaskID {
  // Type Packet
  val data = UInt(typeSize.W)
}


object TypBundle {
  def default(implicit p: Parameters): TypBundle = {
    val wire = Wire(new TypBundle)
    wire.data := 0.U
    wire.predicate := false.B
    wire.taskID := 0.U
    wire
  }
}


/**
  * Data bundle between dataflow nodes.
  *
  * @note 2 fields
  *       data : U(xlen.W)
  *       predicate : Bool
  * @param p : implicit
  * @return
  */
class DataBundle(implicit p: Parameters) extends PredicateT with TaskID {
  // Data packet
  val data = UInt(xlen.W)

  def asControlBundle(): ControlBundle = {
    val wire = Wire(new ControlBundle)
    wire.control := this.predicate
    wire.taskID := this.taskID
    wire
  }
}


//It defines a hardware module for transferring control signals.
/**
  * Control bundle between branch and
  * basicblock nodes
  *
  * control  : Bool
  */
class ControlBundle(implicit p: Parameters) extends AccelBundle()(p) {
  //Control packet
  val taskID = UInt(tlen.W)
  val control = Bool()
  val debug = Bool()

  def asDataBundle(): DataBundle = {
    val wire = Wire(new DataBundle)
    wire.data := this.control.asUInt
    wire.predicate := this.control
    wire.taskID := this.taskID
    wire
  }
}

object ControlBundle {
  def default(implicit p: Parameters): ControlBundle = {
    val wire = Wire(new ControlBundle)
    wire.control := false.B
    wire.taskID := 0.U
    wire.debug := false.B
    wire
  }

  def default(control: Bool, task: UInt)(implicit p: Parameters): ControlBundle = {
    val wire = Wire(new ControlBundle)
    wire.control := control
    wire.taskID := task
    wire.debug := false.B
    wire
  }

  def default(control: Bool, task: UInt, debug: Bool)(implicit p: Parameters): ControlBundle = {
    val wire = Wire(new ControlBundle)
    wire.control := control
    wire.taskID := task
    wire.debug := debug
    wire
  }

  def active(taskID: UInt = 0.U)(implicit p: Parameters): ControlBundle = {
    val wire = Wire(new ControlBundle)
    wire.control := true.B
    wire.taskID := taskID
    wire.debug := false.B
    wire
  }

  def debug(taskID: UInt = 0.U)(implicit p: Parameters): ControlBundle = {
    val wire = Wire(new ControlBundle)
    wire.control := true.B
    wire.taskID := taskID
    wire.debug := true.B
    wire
  }


  def deactivate(taskID: UInt = 0.U)(implicit p: Parameters): ControlBundle = {
    val wire = Wire(new ControlBundle)
    wire.control := false.B
    wire.taskID := taskID
    wire.debug := false.B
    wire
  }


  def apply(control: Bool = false.B, taskID: UInt = 0.U, debug: Bool = false.B)(implicit p: Parameters): ControlBundle = {
    val wire = Wire(new ControlBundle)
    wire.control := control
    wire.taskID := taskID
    wire.debug := debug
    wire
  }

}


/**
  * @brief Handshaking between data nodes.
  * @note Sets up base registers and hand shaking registers
  * @param NumPredOps Number of parents
  * @param NumSuccOps Number of successors
  * @param NumOuts    Number of outputs
  * @param ID         Node id
  * @return Module
  */

class HandShaking[T <: Data](val NumPredOps: Int,
                             val NumSuccOps: Int,
                             val NumOuts: Int,
                             val ID: Int,
			     val Debug: Boolean = false)(gen: T)(implicit val p: Parameters)
  extends MultiIOModule with HasAccelParams with UniformPrintfs {

  lazy val io = IO(new HandShakingIOPS(NumPredOps, NumSuccOps, NumOuts ,Debug)(gen))

  /*=================================
  =            Registers            =
  =================================*/
  // Extra information
  val token    = RegInit(0.U)
  val nodeID_R = RegInit(ID.U)

  // Enable
  val enable_R       = RegInit(ControlBundle.default)
  val enable_valid_R = RegInit(false.B)

  // Predecessor Handshaking
  val pred_valid_R  = Seq.fill(NumPredOps)(RegInit(false.B))
  val pred_bundle_R = Seq.fill(NumPredOps)(RegInit(ControlBundle.default))

  // Successor Handshaking. Registers needed
  val succ_ready_R  = Seq.fill(NumSuccOps)(RegInit(false.B))
  val succ_valid_R  = Seq.fill(NumSuccOps)(RegInit(false.B))
  val succ_bundle_R = Seq.fill(NumSuccOps)(RegInit(ControlBundle.default))

  // Output Handshaking
  val out_ready_R = RegInit(VecInit(Seq.fill(NumOuts)(false.B)))
  val out_valid_R = RegInit(VecInit(Seq.fill(NumOuts)(false.B)))

  // Wire
  val out_ready_W  = WireInit(VecInit(Seq.fill(NumOuts) {
    false.B
  }))
  val succ_ready_W = Seq.fill(NumSuccOps)(WireInit(false.B))

  /*==============================
  =            Wiring            =
  ==============================*/
  // Wire up Successors READYs and VALIDs
  for (i <- 0 until NumSuccOps) {
    io.SuccOp(i).valid := succ_valid_R(i)
    io.SuccOp(i).bits := succ_bundle_R(i)
    succ_ready_W(i) := io.SuccOp(i).ready
    when(io.SuccOp(i).fire( )) {
      succ_ready_R(i) := io.SuccOp(i).ready
      succ_valid_R(i) := false.B
    }
  }

  // Wire up OUT READYs and VALIDs
  for (i <- 0 until NumOuts) {
    io.Out(i).valid := out_valid_R(i)
    out_ready_W(i) := io.Out(i).ready
    when(io.Out(i).fire( )) {
      // Detecting when to reset
      out_ready_R(i) := io.Out(i).ready
      // Propagating output
      out_valid_R(i) := false.B
    }
  }
  // Wire up Predecessor READY and VALIDs
  for (i <- 0 until NumPredOps) {
    io.PredOp(i).ready := ~pred_valid_R(i)
    when(io.PredOp(i).fire( )) {
      pred_valid_R(i) := io.PredOp(i).valid
      pred_bundle_R(i) := io.PredOp(i).bits
    }
  }

  //Enable is an input
  // Wire up enable READY and VALIDs
  io.enable.ready := ~enable_valid_R
  when(io.enable.fire( )) {
    enable_valid_R := io.enable.valid
    enable_R := io.enable.bits
  }

  /*=====================================
  =            Helper Checks            =
  =====================================*/


  def IsEnable(): Bool = {
    return enable_R.control
  }

  def IsEnableValid(): Bool = {
    enable_valid_R
  }

  def ResetEnable(): Unit = {
    enable_valid_R := false.B
  }

  // Check if Predecssors have fired
  def IsPredValid(): Bool = {
    if (NumPredOps == 0) {
      return true.B
    } else {
      VecInit(pred_valid_R).asUInt.andR
    }
  }

  // Fire Predecessors
  def ValidPred(): Unit = {
    pred_valid_R.map {
      _ := true.B
    }
    // pred_valid_R := Seq.fill(NumPredOps) {
    //   true.B
    // }
  }

  // Clear predessors
  def InvalidPred(): Unit = {
    pred_valid_R.foreach {
      _ := false.B
    }
    // pred_valid_R := Vec(Seq.fill(NumPredOps) {
    //   false.B
    // })
  }

  // Successors
  def IsSuccReady(): Bool = {
    if (NumSuccOps == 0) {
      return true.B
    } else {
      VecInit(succ_ready_R).asUInt.andR | VecInit(succ_ready_W).asUInt.andR
    }
  }

  def ValidSucc(): Unit = {
    succ_valid_R.foreach {
      _ := true.B
    }
  }

  def InvalidSucc(): Unit = {
    succ_valid_R.foreach {
      _ := false.B
    }
  }

  // OUTs
  def IsOutReady(): Bool = {
    out_ready_R.asUInt.andR | out_ready_W.asUInt.andR
  }

  def ValidOut(): Unit = {
    (out_valid_R zip io.Out.map(_.fire)).foreach{ case (a,b) => a := b ^ true.B}
    //out_valid_R := VecInit(Seq.fill(NumOuts)(true.B))
  }

  def InvalidOut(): Unit = {
    out_valid_R := VecInit(Seq.fill(NumOuts)(false.B))
  }

  def Reset(): Unit = {
    pred_valid_R.foreach {
      _ := false.B
    }

    succ_ready_R.foreach {
      _ := false.B
    }

    out_ready_R := VecInit(Seq.fill(NumOuts) {
      false.B
    })
    enable_valid_R := false.B
  }
}

// Bundle of Decoupled DataBundles with data width specified by the argTypes parameter
class VariableDecoupledData(val argTypes: Seq[Int])(implicit p: Parameters) extends Record {
  var elts = Seq.tabulate(argTypes.length) {
    i =>
      s"field$i" -> Decoupled(new DataBundle()(
        p.alterPartial({ case HeteaccConfigKey => p(HeteaccConfigKey).copy(dataLen = argTypes(i)) })
      )
      )
  }
  override val elements = ListMap(elts map { case (field, elt) => field -> elt.cloneType }: _*)

  def apply(elt: String) = elements(elt)
  override def cloneType = new VariableDecoupledData(argTypes).asInstanceOf[this.type]

}

// Call type that wraps an enable and variable DataBundle together
class Call(val argTypes: Seq[Int])(implicit p: Parameters) extends AccelBundle() {
  val enable = new ControlBundle
  val data = new VariableData(argTypes)

}

// Bundle of DataBundles with data width specified by the argTypes parameter
class VariableData(val argTypes: Seq[Int])(implicit p: Parameters) extends Record {

  var elts = Seq.tabulate(argTypes.length) {
    i =>
      s"field$i" -> new DataBundle()(
        p.alterPartial({ case HeteaccConfigKey => p(HeteaccConfigKey).copy(dataLen = argTypes(i)) })
      )
  }
  override val elements = ListMap(elts map { case (field, elt) => field -> elt.cloneType }: _*)

  def apply(elt: String) = elements(elt)
  override def cloneType = new VariableData(argTypes).asInstanceOf[this.type]

}

// Bundle of Decoupled DataBundle Vectors. Data width is default. Intended for use on outputs
// of a block (i.e. configurable number of output with configurable number of copies of each output)
class VariableDecoupledVec(val argTypes: Seq[Int])(implicit p: Parameters) extends Record {
  var elts = Seq.tabulate(argTypes.length) {
    i => s"field$i" -> Vec(argTypes(i), Decoupled(new DataBundle()(p)))
  }
  override val elements = ListMap(elts map { case (field, elt) => field -> elt.cloneType }: _*)

  def apply(elt: String) = elements(elt)
  override def cloneType = new VariableDecoupledVec(argTypes).asInstanceOf[this.type]

}

// Call type that wraps a decoupled enable and decoupled variable data bundle together
class CallDecoupled(val argTypes: Seq[Int])(implicit p: Parameters) extends AccelBundle() {
  val enable = Decoupled(new ControlBundle)
  val data = new VariableDecoupledData(argTypes)

}
class OutputBundle(implicit p: Parameters) extends AccelBundle() {
  val enable = (Decoupled(new ControlBundle))
  val data = (Decoupled(UInt(32.W)))
}

// Call type that wraps a decoupled enable and decoupled vector DataBundle together
class CallDecoupledVec(val argTypes: Seq[Int])(implicit p: Parameters) extends AccelBundle() {
  val enable = Decoupled(new ControlBundle)
  val data = new VariableDecoupledVec(argTypes)

}
