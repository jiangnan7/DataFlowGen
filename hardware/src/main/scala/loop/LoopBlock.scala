package heteacc.loop

import chisel3.{Flipped, Module, _}
import chisel3.util.{Decoupled, _}
import chipsalliance.rocketchip.config._
import heteacc.config._
import heteacc.interfaces._
import utility.Constants._
import heteacc.junctions._
import heteacc.node._
import utility.UniformPrintfs
import heteacc.config._
import chisel3.util.experimental.BoringUtils


/**
  * @brief LoopBlockIO class definition
  * @note Implimentation of BasickBlockIO
  * @param NumIns   Number of liveIn DataBundles
  * @param NumOuts  Number of liveOut DataBundles
  * @param NumExits Number exit control bundles for loop (e.g. number of exit points)
  *                 I/O:
  *                 In          = Connect each element to an upstream liveIn source for the loop
  *                 activate    = Connect to the activate input of the LoopHeader
  *                 latchEnable = Connect to the enable for the loop feedback path
  *                 liveIn      = Connect each element to the corresponding liveIn termination inside the loop
  *                 loopExit    = Connect to the exit control bundle(s) for the loop
  *                 liveOut     = Connect each element to the corresponding liveOut source inside the loop
  *                 endEnable   = Connect to the enable input of the downstream loopEnd block
  *                 Out         = Connect each element to a downstream liveOut termination
  *                 Operation:
  *                 The In values are latched when they are valid.  Their values are connected to the
  *                 liveIN outputs for the loop logic. The liveIN values will be validated at the start of a loop and each time
  *                 the latchEnable signal is active and valid.
  *                 The liveOut values are latched when an active and valid loopExit signal is asserted. They are connected
  *                 the Out outputs for the downstream logic.  The endEnable is also driven to trigger the end block downstream
  *                 from the loop.
  */

class LoopBlockNodeIO(NumIns: Seq[Int], NumCarry: Seq[Int], NumOuts: Seq[Int],
                      NumBackEdge: Int = 1, NumLoopFinish: Int = 1, NumExits: Int, NumStore: Int = 0, Debug: Boolean)
                     (implicit p: Parameters) extends AccelBundle {

  // INPUT from outside of the loop head
  // Predicate enable
  val enable = Flipped(Decoupled(new ControlBundle))
  // Live-in values
  val InLiveIn = Vec(NumIns.length, Flipped(Decoupled(new DataBundle())))

  // OUTPUT to internal part of the loop
  // Ouput live-in values
  val OutLiveIn = new VariableDecoupledVec(NumIns)
  // Output control signal to fire loop
  val activate_loop_start = Decoupled(new ControlBundle())
  val activate_loop_back = Decoupled(new ControlBundle())

  // Loop input control signals
  val loopBack = Vec(NumBackEdge, Flipped(Decoupled(new ControlBundle())))
  val loopFinish = Vec(NumLoopFinish, Flipped(Decoupled(new ControlBundle())))

  // Carry dependencies
  val CarryDepenIn = Vec(NumCarry.length, Flipped(Decoupled(new DataBundle())))
  val CarryDepenOut = new VariableDecoupledVec(NumCarry)

  // Live-out values
  val InLiveOut = Vec(NumOuts.length, Flipped(Decoupled(new DataBundle())))
  val OutLiveOut = new VariableDecoupledVec(NumOuts)

  // val StoreDepen = Vec(NumStore, Flipped(Decoupled(new ControlBundle())))

  // OUTPUT to outside of the loop
  //Output control signal
  val loopExit = Vec(NumExits, Decoupled(new ControlBundle()))

}

class LoopBlockNode(ID: Int, NumIns: Seq[Int], NumCarry: Seq[Int], NumOuts: Seq[Int],
                    NumBackEdge: Int = 1, NumLoopFinish: Int = 1, NumExits: Int, NumStore: Int = 0, Debug: Boolean = false)
                   (implicit val p: Parameters,
                    name: sourcecode.Name,
                    file: sourcecode.File) extends Module with HasAccelParams with UniformPrintfs {

  // Instantiate TaskController I/O signals
  val io = IO(new LoopBlockNodeIO(NumIns, NumCarry, NumOuts, NumBackEdge, NumLoopFinish, NumExits, NumStore, Debug))

  // Printf debugging
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)
  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "
  val loopcounter=RegInit(0.U(32.W))

  /**
    * Input signals and their latches
    */
  val enable_R = RegInit(ControlBundle.default)
  val enable_valid_R = RegInit(false.B)

  val loop_back_R = Seq.fill(NumBackEdge)(RegInit(ControlBundle.default))
  val loop_back_valid_R = Seq.fill(NumBackEdge)(RegInit(false.B))

  val loop_finish_R = Seq.fill(NumLoopFinish)(RegInit(ControlBundle.default))
  val loop_finish_valid_R = Seq.fill(NumLoopFinish)(RegInit(false.B))

  val in_live_in_R = Seq.fill(NumIns.length)(RegInit(DataBundle.default))
  val in_live_in_valid_R = Seq.fill(NumIns.length)(RegInit(false.B))

  val in_carry_in_R = Seq.fill(NumCarry.length)(RegInit(DataBundle.active()))
  val in_carry_in_valid_R = Seq.fill(NumCarry.length)(RegInit(false.B))

  val in_live_out_R = Seq.fill(NumOuts.length)(RegInit(DataBundle.default))
  val in_live_out_valid_R = Seq.fill(NumOuts.length)(RegInit(false.B))

  // val store_depn_R = Seq.fill(NumStore)(RegInit(ControlBundle.default))
  // val store_depen_valid_R = Seq.fill(NumStore)(RegInit(false.B))

  /**
    * Output signals and their valid and latch signals
    */
  val out_live_in_valid_R = for (i <- NumIns.indices) yield {
    val validReg = Seq.fill(NumIns(i))(RegInit(false.B))
    validReg
  }
  val out_live_in_fire_R = for (i <- NumIns.indices) yield {
    val validReg = Seq.fill(NumIns(i))(RegInit(false.B))
    validReg
  }


  val out_live_out_valid_R = for (i <- NumOuts.indices) yield {
    val liveout = Seq.fill(NumOuts(i))(RegInit(false.B))
    liveout
  }
  val out_live_out_fire_R = for (i <- NumOuts.indices) yield {
    val liveout = Seq.fill(NumOuts(i))(RegInit(false.B))
    liveout
  }


  val out_carry_out_valid_R = for (i <- NumCarry.indices) yield {
    val Reg = Seq.fill(NumCarry(i))(RegInit(false.B))
    Reg
  }
  val out_carry_out_fire_R = for (i <- NumCarry.indices) yield {
    val Reg = Seq.fill(NumCarry(i))(RegInit(false.B))
    Reg
  }

  val active_loop_start_R = RegInit(ControlBundle.default)
  val active_loop_start_valid_R = RegInit(false.B)

  val active_loop_back_R = RegInit(ControlBundle.default)
  val active_loop_back_valid_R = RegInit(false.B)

  val loop_exit_R = Seq.fill(NumExits)(RegInit(ControlBundle.default))
  val loop_exit_valid_R = Seq.fill(NumExits)(RegInit(false.B))
  val loop_exit_fire_R = Seq.fill(NumExits)(RegInit(false.B))
  
  /**
    * Latch all the inputs to the circuite
    */
  io.enable.ready := ~enable_valid_R
  when(io.enable.fire) {
    enable_R <> io.enable.bits
    enable_valid_R := true.B
  }

  for (i <- 0 until NumBackEdge) {
    io.loopBack(i).ready := ~loop_back_valid_R(i)
    when(io.loopBack(i).fire) {
      loop_back_R(i) <> io.loopBack(i).bits
      loop_back_valid_R(i) := true.B
    }

  }

  for (i <- 0 until NumLoopFinish) {
    io.loopFinish(i).ready := ~loop_finish_valid_R(i)
    when(io.loopFinish(i).fire) {
      loop_finish_R(i) <> io.loopFinish(i).bits
      loop_finish_valid_R(i) := true.B
    }

  }


  // Latch the block inputs when they fire to drive the liveIn I/O.
  for (i <- 0 until NumIns.length) {
    io.InLiveIn(i).ready := ~in_live_in_valid_R(i)
    when(io.InLiveIn(i).fire) {
      in_live_in_R(i) <> io.InLiveIn(i).bits
      in_live_in_valid_R(i) := true.B
    }
  }

  // Latch the liveOut inputs when they fire to drive the Out I/O
  for (i <- 0 until NumOuts.length) {
    io.InLiveOut(i).ready := ~in_live_out_valid_R(i)
    when(io.InLiveOut(i).fire) {
      in_live_out_R(i) <> io.InLiveOut(i).bits
      in_live_out_valid_R(i) := true.B
    }
  }

  // Latch the exit signals
  for (i <- 0 until NumCarry.length) {
    io.CarryDepenIn(i).ready := ~in_carry_in_valid_R(i)
    when(io.CarryDepenIn(i).fire) {
      in_carry_in_R(i) <> io.CarryDepenIn(i).bits
      in_carry_in_valid_R(i) := true.B
    }
  }

  // if (NumStore > 0) {
  //   for (i <- 0 until NumStore) {
  //     io.StoreDepen(i).ready := ~store_depen_valid_R(i)
  //     when(io.StoreDepen(i).fire) {
  //       store_depn_R(i) <> io.StoreDepen(i).bits
  //       store_depen_valid_R(i) := true.B
  //     }
  //   }
  // }


  /**
    * Connecting outputs
    */
  // Connect LiveIn registers to I/O
  for (i <- NumIns.indices) {
    for (j <- 0 until NumIns(i)) {
      io.OutLiveIn.elements(s"field$i")(j).bits <> in_live_in_R(i)
      io.OutLiveIn.elements(s"field$i")(j).valid := out_live_in_valid_R(i)(j)
    }
  }

  // Connect LiveIn registers to I/O
  for (i <- NumOuts.indices) {
    for (j <- 0 until NumOuts(i)) {
      io.OutLiveOut.elements(s"field$i")(j).bits <> in_live_out_R(i) //this apperantly is the output
      io.OutLiveOut.elements(s"field$i")(j).valid := out_live_out_valid_R(i)(j)
    }
  }

  // Connect LiveIn registers to I/O
  for (i <- NumCarry.indices) {
    for (j <- 0 until NumCarry(i)) {
      io.CarryDepenOut.elements(s"field$i")(j).bits <> in_carry_in_R(i)
      io.CarryDepenOut.elements(s"field$i")(j).valid := out_carry_out_valid_R(i)(j)
    }
  }

  /**
    * Connecting control output signals
    */
  io.activate_loop_start.bits <> active_loop_start_R
  io.activate_loop_start.valid := active_loop_start_valid_R

  io.activate_loop_back.bits <> active_loop_back_R
  io.activate_loop_back.valid := active_loop_back_valid_R

  for (i <- 0 until NumExits) {
    io.loopExit(i).bits <> loop_exit_R(i)
    io.loopExit(i).valid <> loop_exit_valid_R(i)
  }


  /**
    * Connecting output handshake signals
    */

  when(io.activate_loop_start.fire) {
    active_loop_start_valid_R := false.B
  }

  when(io.activate_loop_back.fire) {
    active_loop_back_valid_R := false.B
  }

  for (i <- 0 until NumExits) {
    when(io.loopExit(i).fire) {
      loop_exit_valid_R(i) := false.B
      loop_exit_fire_R(i) := true.B
    }
  }

  // Connect LiveIn registers to I/O
  for (i <- NumIns.indices) {
    for (j <- 0 until NumIns(i)) {
      when(io.OutLiveIn.elements(s"field$i")(j).fire) {
        out_live_in_valid_R(i)(j) := false.B
        out_live_in_fire_R(i)(j) := true.B
      }
    }
  }

  // Connect LiveOut registers to I/O
  for (i <- NumOuts.indices) {
    for (j <- 0 until NumOuts(i)) {
      when(io.OutLiveOut.elements(s"field$i")(j).fire) {
        out_live_out_valid_R(i)(j) := false.B
        out_live_out_fire_R(i)(j) := true.B
      }
    }
  }

  // Connect LiveIn registers to I/O
  for (i <- NumCarry.indices) {
    for (j <- 0 until NumCarry(i)) {
      when(io.CarryDepenOut.elements(s"field$i")(j).fire) {
        out_carry_out_valid_R(i)(j) := false.B
        out_carry_out_fire_R(i)(j) := true.B
      }
    }
  }

  /**
    * Helper functions
    */

  def IsEnableValid(): Bool = {
    enable_valid_R
  }

  def IsEnable(): Bool = {
    enable_R.control
  }

  def IsLiveInValid(): Bool = {
    if (NumIns.length == 0) {
      return true.B
    } else {
      in_live_in_valid_R.reduce(_ && _)
    }
  }

  // Live outs are ready if all have fired
  def IsLiveOutValid(): Bool = {
    if (NumOuts.length == 0) {
      return true.B
    } else {
      in_live_out_valid_R.reduceLeft(_ && _)
    }
  }

  // Live outs are ready if all have fired
  def IsCarryDepenValid(): Bool = {
    if (NumOuts.length == 0) {
      return true.B
    } else {
      in_carry_in_valid_R.reduceLeft(_ && _)
    }
  }

  def IsLoopBackValid(): Bool = {
    return loop_back_valid_R.reduce(_ & _)
  }

  def IsLoopFinishValid(): Bool = {
    return loop_finish_valid_R.reduce(_ & _)
  }

  def ValidOut(): Unit = {
    for (i <- NumOuts.indices) {
      out_live_out_valid_R(i).foreach(_ := true.B)
    }
  }

  def IsExitsFired(): Bool = {
    loop_exit_fire_R.reduce(_ & _)
  }

  def IsLiveOutFired(): Bool = {
    if (NumOuts.length == 0) {
      return true.B
    }
    else {
      val fire_mask = for (i <- NumOuts.indices) yield {
        val fire_mask_live_out = out_live_out_fire_R(i) reduce (_ & _)
        fire_mask_live_out
      }
      fire_mask.reduce(_ & _)
    }

  }

  def IsLiveInFired(): Bool = {
    if (NumIns.length == 0) {
      return true.B
    }
    else {
      val fire_mask = for (i <- NumIns.indices) yield {
        val fire_mask_live_in = out_live_in_fire_R(i) reduce (_ & _)
        fire_mask_live_in
      }
      fire_mask.reduce(_ & _)
    }

  }

  def IsCarryOutFired(): Bool = {
    if (NumOuts.length == 0) {
      return true.B
    }
    else {
      val fire_mask = for (i <- NumOuts.indices) yield {
        val fire_mask_live_out = out_carry_out_fire_R(i) reduce (_ & _)
        fire_mask_live_out
      }
      fire_mask.reduce(_ & _)
    }

  }

  // def IsStoreDepnValid(): Bool = {
  //   if (NumStore == 0) {
  //     true.B
  //   }
  //   else {
  //     store_depen_valid_R.reduce(_ & _)
  //   }
  // }

  /**
    * State machines
    */

  val s_idle :: s_active :: s_end :: Nil = Enum(3)
  val state = RegInit(s_idle)


  switch(state) {
    is(s_idle) {
      /**
        * Init values for registers
        **/
      //Wait for all the inputs and enable signal to latch
      when(IsLiveInValid() && IsEnableValid()) {
        when(IsEnable()) {
          //If loop is in the if(true) path go to active state
          // Set the loop liveIN data as valid

          in_carry_in_R.foreach(_ := DataBundle.default)
          out_live_in_valid_R.foreach(_.foreach(_ := true.B))
          out_carry_out_valid_R.foreach(_.foreach(_ := true.B))

          active_loop_start_R := ControlBundle.active(enable_R.taskID)
          active_loop_start_valid_R := true.B

          active_loop_back_R := ControlBundle.deactivate(enable_R.taskID)
          active_loop_back_valid_R := true.B

          //Change state
          state := s_active
        }.otherwise {
          //If loop is in the if(false) path, put some garbage value on handshaking
          // Fire live-outs
          in_live_out_R.foreach(_ := DataBundle.deactivate())
          out_live_out_valid_R.foreach(_.foreach(_ := true.B))

          // Fire Loop exists
          loop_exit_R.foreach(_ := ControlBundle.deactivate())
          loop_exit_valid_R.foreach(_ := true.B)

          //Change state
          state := s_end
        }
      }
    }
    is(s_active) {
      when(IsLoopBackValid() && IsLoopFinishValid()
        && IsLiveOutValid() && IsLiveInFired()
        && IsCarryDepenValid()) {// && IsStoreDepnValid()
        //When loop needs to repeat itself
        when(loop_back_R.map(_.control).reduce(_ | _)) { //in case of multiple backwardedges being valid
          //Drive loop internal output signals
          loopcounter := loopcounter + 1.U
          active_loop_start_R := ControlBundle.deactivate(loop_back_R(0).taskID) //outer input
          active_loop_start_valid_R := true.B

          active_loop_back_R := ControlBundle.active(loop_back_R(0).taskID)
          active_loop_back_valid_R := true.B

          out_live_in_fire_R.foreach(_.foreach(_ := false.B))
          out_carry_out_fire_R.foreach(_.foreach(_ := false.B))

          out_live_in_valid_R.foreach(_.foreach(_ := true.B))
          out_carry_out_valid_R.foreach(_.foreach(_ := true.B))

          loop_back_R.foreach(_ := ControlBundle.default)
          loop_back_valid_R.foreach(_ := false.B)

          loop_finish_R.foreach(_ := ControlBundle.default)
          loop_finish_valid_R.foreach(_ := false.B)

          in_live_out_valid_R.foreach(_ := false.B)
          in_carry_in_valid_R.foreach(_ := false.B)

          // store_depen_valid_R.foreach(_ := false.B)
          // store_depn_R.foreach(_ := ControlBundle.default)

          state := s_active

          if (log) {
            printf(p"[LOG] [${module_name}] [TID: ${io.activate_loop_start.bits.taskID}]" +
              p" [LOOP] [${node_name}] [RESTARTED] [Cycle: ${loopcounter}] [LoopCycle: ${cycleCount}]\n")
          }

        }.elsewhen(loop_finish_R.map(_.control).reduce(_ | _)) { //last iteration
          // Fire live-outs and loop exit control signal
          out_live_out_valid_R.foreach(_.foreach(_ := true.B))
          loop_exit_valid_R.foreach(_ := true.B)

          active_loop_start_R := ControlBundle.default
          active_loop_back_R := ControlBundle.default
          // in_carry_in_R.foreach(_ := DataBundle.default)
          // Fire Loop exists
          loop_exit_R.foreach(_ := ControlBundle.active(loop_back_R(0).taskID))
          loop_exit_valid_R.foreach(_ := true.B)

          //Change state
          if (log) {
            printf(p"[LOG] [${module_name}] [TID: ${io.activate_loop_start.bits.taskID}] [LOOP]" +
              p" [${node_name}] [FIRED] [Cycle: ${cycleCount}]\n")
            for (i <- 0 until NumOuts.size) {
              printf(p"\tOut[${i.U}] [Val: 0x${Hexadecimal(in_live_out_R(i).data)}]\n")
            }
            //if(NumOuts.size > 0) printf("\n")
          }
          state := s_end
          //in live out should be dumped
          // if (Debug) {
          //   log_out_reg := in_live_out_R(NumOuts.length - 1).asUInt
          // }


          if (log) {
            printf(p"[LOG] [${module_name}] [TID: ${io.activate_loop_start.bits.taskID}] [LOOP]" +
              p" [${node_name}] [FINAL] [Cycle: ${cycleCount}]\n")
          }
        }

      }
    }
    is(s_end) {
      when(IsExitsFired() && IsLiveOutFired()) {

        //Restart to initial state

        enable_R := ControlBundle.default
        enable_valid_R := false.B

        loop_back_R foreach (_ := ControlBundle.default)
        loop_back_valid_R foreach (_ := false.B)

        loop_finish_R foreach (_ := ControlBundle.default)
        loop_finish_valid_R foreach (_ := false.B)

        in_live_in_R.foreach(_ := DataBundle.default)
        in_live_in_valid_R.foreach(_ := false.B)

        in_live_out_valid_R.foreach(_ := false.B)

        in_carry_in_valid_R.foreach(_ := false.B)


        state := s_idle
      }
    }
  }
}

//sbt "test:runMain heteacc.loop.LoopGen"
import java.io.PrintWriter
object LoopGen extends App {
    implicit val p = new WithAccelConfig ++ new WithTestConfig

    val verilogString = getVerilogString(new LoopBlockNode(NumIns = List(1, 1, 1, 1, 1, 1), NumOuts = List(1), NumCarry = List(1, 1, 1), NumExits = 1, ID = 0))
    val filePath = "RTL/LoopNode.v"
    val writer = new PrintWriter(filePath)
    try {
        writer.write(verilogString)
    } finally {
        writer.close()
    }
}