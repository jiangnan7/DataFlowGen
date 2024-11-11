package heteacc.interfaces


import chisel3._
import chisel3.util.Decoupled
import utility.Constants._
import heteacc.config._
import chipsalliance.rocketchip.config._

import scala.collection.immutable.ListMap

import chisel3._
import chisel3.util._
import chisel3.util.experimental.loadMemoryFromFileInline

trait InitSyncMem {
  def mem: SyncReadMem[UInt]

  def initMem(memoryFile: String) = loadMemoryFromFileInline(mem, memoryFile)
}
/**
 * Memory read request interface
 * @param p
 */
class ReadReq(implicit p: Parameters)
  extends RouteID {
  val address = UInt(xlen.W)
  val taskID = UInt(tlen.W)
  val Typ = UInt(8.W)

}

object ReadReq {
  def default(implicit p: Parameters): ReadReq = {
    val wire = Wire(new ReadReq)
    wire.address := 0.U
    wire.taskID := 0.U
    wire.RouteID := 0.U
    wire.Typ := MT_D
    wire
  }
}


//  data : data returned from scratchpad
// class ReadResp(implicit p: Parameters) extends ValidT
//     with RouteID {
//   val data = UInt(xlen.W)

//   override def toPrintable: Printable = {
//     p"ReadResp {\n" +
//       p"  RouteID: ${RouteID}\n" +
//       p"  data   : 0x${Hexadecimal(data)} }"
//   }
// }

//  data : data returned from scratchpad
class ReadResp(implicit p: Parameters) extends ValidT with RouteID {
  val data = UInt(xlen.W)

  override def toPrintable: Printable = {
    p"ReadResp {\n" +
      p"  RouteID: ${RouteID}\n" +
      p"  data   : 0x${Hexadecimal(data)} }"
  }
}
object ReadResp {
  def default(implicit p: Parameters): ReadResp = {
    val wire = Wire(new ReadResp)
    wire.data := 0.U
    wire.RouteID := 0.U
    wire.valid := false.B
    wire
  }
}

/**
  * Write request to memory
  *
  * @param p [description]
  * @return [description]
  */
//
// Word aligned to write to
// Node performing the write
// Mask indicates which bytes to update.
class WriteReq(implicit p: Parameters)
  extends RouteID {
  val address = UInt((xlen).W)// - 10
  val data = UInt(xlen.W)
  val mask = UInt((xlen / 8).W)
  val taskID = UInt(tlen.W)
  val Typ = UInt(8.W)
}

object WriteReq {
  def default(implicit p: Parameters): WriteReq = {
    val wire = Wire(new WriteReq)
    wire.address := 0.U
    wire.data := 0.U
    wire.mask := 0.U
    wire.taskID := 0.U
    wire.RouteID := 0.U
    wire.Typ := MT_D
    wire
  }
}

// Explicitly indicate done flag
class WriteResp(implicit p: Parameters) extends RouteID {
  val done = Bool()
}

//  data : data returned from scratchpad
class FUResp(implicit p: Parameters) extends AccelBundle {
  val data = UInt(xlen.W)

  override def toPrintable: Printable = {
    p"FUResp {\n" +
      p"  data   : 0x${Hexadecimal(data)} }"
  }
}

object FUResp {
  def default(implicit p: Parameters): FUResp = {
    val wire = Wire(new FUResp)
    wire.data := 0.U
    wire
  }
}

// MemReq and MemResp, 
// data structures used to represent 
// memory requests and memory responses.
class MemReq(implicit p: Parameters) extends AccelBundle()(p) {
  val addr = UInt(xlen.W)
  val data = UInt(xlen.W)
  val mask = UInt((xlen / 8).W)
  val tag = UInt((List(1, mshrLen).max).W)
  val taskID = UInt(tlen.W)
  val iswrite = Bool()

}

object MemReq {
  def apply()(implicit p: Parameters): MemReq = {
    val wire = Wire(new MemReq())
    wire.addr := 0.U
    wire.data := 0.U
    wire.mask := 0.U
    wire.tag := 0.U
    wire.taskID := 0.U
    wire.iswrite := false.B
    wire
  }

  def default(implicit p: Parameters): MemReq = {
    val wire = Wire(new MemReq())
    wire.addr := 0.U
    wire.data := 0.U
    wire.mask := 0.U
    wire.tag := 0.U
    wire.taskID := 0.U
    wire.iswrite := false.B
    wire
  }
}

trait ValidT extends AccelBundle {
  val valid = Bool()
}

// class MemResp(implicit p: Parameters) extends AccelBundle()(p)  with ValidT {
//   val data = UInt(xlen.W)
//   val tag = UInt((List(1, mshrLen).max).W)
//   val iswrite = Bool()
//   // def clone_and_set_tile_id( ): MemResp = {
//   //   val wire = Wire(new MemResp())
//   //   wire.data := this.data
//   //   wire.tag := this.tag
//   //   wire.iswrite := this.iswrite
//   //    wire
//   // }
// }
class MemResp(implicit p: Parameters) extends AccelBundle()(p) {
  val data = UInt(xlen.W)
  val tag = UInt((List(1, mshrLen).max).W)
  val iswrite = Bool()

}
object MemResp {
  def default(implicit p: Parameters): MemResp = {
    val wire = Wire(new MemResp())
    wire.data := 0.U
    wire.tag := 0.U
    wire.iswrite := false.B
    wire
  }
}
// object MemResp {
//   def default(implicit p: Parameters): MemResp = {
//     val wire = Wire(new MemResp())
//     wire.valid := false.B
//     wire.data := 0.U
//     wire.tag := 0.U
//     wire.iswrite := false.B
//     wire
//   }
// }
