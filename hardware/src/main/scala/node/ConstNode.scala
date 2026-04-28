package heteacc.node

import chisel3._
import chisel3.Module
import chipsalliance.rocketchip.config._
import heteacc.config._
import heteacc.interfaces._
import util._
import utility.UniformPrintfs


class ConstFastNode(value: BigInt, ID: Int)
                    (implicit val p: Parameters,
                     name: sourcecode.Name,
                     file: sourcecode.File)
  extends Module with HasAccelParams with UniformPrintfs {

  val io = IO(new Bundle {
    val enable = Flipped(Decoupled(new ControlBundle))
    val Out = Decoupled(new DataBundle)
  })

  /*===========================================*
   *            Registers                      *
   *===========================================*/
  val enable_R = RegInit(ControlBundle.default)
  val enable_valid_R = RegInit(false.B)

  val output_enable = Mux(enable_valid_R, enable_R, io.enable.bits)
  val taskID = output_enable.taskID
  val predicate = output_enable.control

  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/


  val output_value = value.asSInt(xlen.W).asUInt
  io.enable.ready := ~enable_valid_R
  io.Out.valid := enable_valid_R || io.enable.valid

  io.Out.bits := DataBundle(output_value, taskID, predicate)

  /*============================================*
   *            ACTIONS (possibly dangerous)    *
   *============================================*/
  when(!enable_valid_R && io.enable.fire && !io.Out.ready) {
    enable_valid_R := true.B
    enable_R := io.enable.bits
  }

  when(enable_valid_R && io.Out.fire) {
    enable_R := ControlBundle.default
    enable_valid_R := false.B
  }

}


// class ConstFastNode(value: BigInt, ID: Int)
//                     (implicit val p: Parameters,
//                      name: sourcecode.Name,
//                      file: sourcecode.File)
//   extends Module with HasAccelParams with UniformPrintfs {

//   val io = IO(new Bundle {
//     val enable = Flipped(Decoupled(new ControlBundle))
//     val Out = Decoupled(new DataBundle)
//   })

//   io.enable <> io.Out
//   val output_value = value.asSInt(xlen.W).asUInt
//   io.Out.bits := DataBundle(output_value, io.enable.bits.taskID, io.enable.bits.control)
// }

class ConstFastNodeWithVectorizationIO(NumLanes: Int)
                                      (implicit p: Parameters) extends AccelBundle {
  val enable = Flipped(Decoupled(new ControlBundle))
  val Out = Vec(NumLanes, Decoupled(new DataBundle))
}


class ConstFastNodeWithVectorization(value: BigInt, NumLanes: Int, ID: Int)
                                    (implicit val p: Parameters,
                                     name: sourcecode.Name,
                                     file: sourcecode.File)
  extends Module with HasAccelParams with UniformPrintfs {

  val io = IO(new ConstFastNodeWithVectorizationIO(NumLanes))

  val enable_R = RegInit(ControlBundle.default)
  val enable_valid_R = RegInit(false.B)

  val taskID = Mux(enable_valid_R, enable_R.taskID, io.enable.bits.taskID)
  val predicate = Mux(enable_valid_R, enable_R.control, io.enable.bits.control)

  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/


  val output_value = value.asSInt(xlen.W).asUInt
  val all_outputs_ready =
    if (NumLanes == 0) {
      true.B
    } else {
      io.Out.map(_.ready).reduce(_ && _)
    }

  io.enable.ready := ~enable_valid_R
  io.Out.foreach(_.valid := enable_valid_R || io.enable.valid)
  io.Out.foreach(_.bits := DataBundle(output_value, taskID, predicate))

  /*============================================*
   *            ACTIONS (possibly dangerous)    *
   *============================================*/
  when(!enable_valid_R && io.enable.fire && !all_outputs_ready) {
    enable_valid_R := true.B
    enable_R := io.enable.bits
  }

  when(enable_valid_R && all_outputs_ready) {
    enable_R := ControlBundle.default
    enable_valid_R := false.B
  }

}

import java.io.PrintWriter
object ConstNodeGen extends App {
  implicit val p = new WithAccelConfig ++ new WithTestConfig

  val constants = Seq(
    (1, 0), (-1, 1), (99, 2), (100, 3), (0, 4)
  )
  // sbt "test:runMain heteacc.node.ConstNodeGen"
  constants.foreach { case (value, id) =>
    val verilogString = getVerilogString(new ConstFastNode(value, id))
    val filePath = s"RTL/ConstNode/ConstFastNode_$value.v"
    val writer = new PrintWriter(filePath)
    try {
      writer.write(verilogString)
      println(s"Generated Verilog for ConstFastNode with value $value and saved to $filePath")
    } finally {
      writer.close()
    }
  }
}
