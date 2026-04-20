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
import heteacc.mul._
import utility.UniformPrintfs


class ReductionNodeWithVectorizationIO(NumOuts: Int, NumLanes: Int)
            (implicit p: Parameters) 
            extends HandShakingDynIO(NumOuts)(new DataBundle) {

  val Input = Vec(NumLanes, Flipped(Decoupled(new DataBundle())))
}

class ReductionNodeWithVectorization(NumOuts: Int, NumLanes: Int, ID: Int, opCode: String, BranchSupport: Boolean)
                              (sign: Boolean)
                              (implicit p: Parameters,
                    name: sourcecode.Name,
                    file: sourcecode.File) extends HandShakingDyn(NumOuts, NumLanes)(new DataBundle())(p)
    with HasAccelShellParams{
  override lazy val io = IO(new ReductionNodeWithVectorizationIO(NumOuts, NumLanes))

  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  /*===========================================*
   *            Registers                      *
   *===========================================*/

  private val join = Module(new Join(NumLanes)) 
  private val oehb = Module(new OEHB())

  for (lane <- 0 until NumLanes) {
    join.pValid(lane) := io.Input(lane).valid
    io.Input(lane).ready := join.ready(lane)
  }

  join.nReady := oehb.dataIn.ready
  oehb.dataIn.valid := join.valid
  oehb.dataIn.bits := DontCare 

  val inputData = Wire(Vec(NumLanes, UInt(xlen.W)))
  for (i <- 0 until NumLanes) {
    inputData(i) := io.Input(i).bits.data
  }

  def reduceTree(data: Seq[UInt]): UInt = {
    if (data.size == 1) {
      data.head
    } else {
      val pairs = data.grouped(2).toSeq
      val reduced = pairs.map {
        case Seq(a, b) => 
          val alu = Module(new UALU(xlen, opCode))
          alu.io.in1 := a
          alu.io.in2 := b
          alu.io.out
      }
      reduceTree(reduced)
    }
  }

  
  val accumulator = RegInit(0.U(xlen.W))

  oehb.dataOut.ready := io.Out.map(_.ready).reduce(_ && _)

  for (i <- 0 until NumOuts) {
    io.Out(i).valid := oehb.dataOut.valid
  }
  val reducedResult = reduceTree(inputData)

  val nextAccum  = accumulator + reducedResult
  val finalResult = Mux(BranchSupport.B, nextAccum, reducedResult)

  io.Out.foreach(_.bits := DataBundle(finalResult, true.B, true.B))


  when(IsOutReady()) {
    if (log) {
      for (lane <- 0 until NumLanes) {
        printf(p"[LOG] [${module_name}] [TID: ${0}] [Reduction] [Name: ${node_name}] " +
        p"[ID: ${ID}] " +
        p"[BranchSupport: ${BranchSupport}] " +
        p"[In: ${Hexadecimal(io.Input(lane).bits.data)}] " +
        p"[Acc: ${Hexadecimal(accumulator)}] " +
        p"[Out: ${Hexadecimal(reducedResult)}] " +
        p"[Cycle: ${cycleCount}]\n")
      }
    }
    when(BranchSupport.B) {
      accumulator := accumulator + reducedResult
    }
    Reset()
  }

}