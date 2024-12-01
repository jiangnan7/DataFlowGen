package heteacc.node

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester, OrderedDecoupledHWIOTester}
import chisel3.Module
import chisel3.testers._
import chisel3.util._
import org.scalatest.{Matchers, FlatSpec}

import chipsalliance.rocketchip.config._
import heteacc.config._
import heteacc.interfaces._
import muxes._
import util._
import heteacc.mul._
import heteacc.fpu._
class FusedComputeNodeIO(NumIns: Int, NumOuts: Int)
                   (implicit p: Parameters)
extends HandShakingFusedIO (NumIns, NumOuts)(new DataBundle) {
}

class FusedComputeNode(NumIns: Int, NumOuts: Int, ID: Int, opCode: String)
                 (sign: Boolean)
                 (implicit  p: Parameters)
  extends HandShakingFused(NumIns, NumOuts, ID)(new DataBundle)(p) {
  override lazy val io = IO(new FusedComputeNodeIO(NumIns, NumOuts))
  // Printf debugging

def PrintOut(): Unit = {
       for(i <- 0 until NumIns) yield {
        		printf("\"O_%x(D,P)\" : \"%x,%x\",",i.U,InRegs(i).data,InRegs(i).predicate)
        	}
     
}




  val node_name = "fusion"
  val module_name = "FUSION"
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  val s_idle  :: s_COMPUTE :: Nil = Enum(2)
  val state = RegInit(s_idle)

  /*==========================================*
   *           Predicate Evaluation           *
   *==========================================*/

  val predicate = IsEnable() //IsInPredicate() & 
  val start = IsInValid() & IsEnableValid()

  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/


  when(start & state =/= s_COMPUTE) {
  	state := s_COMPUTE

    if (log) {  
      printf(p"[LOG] [${module_name}]" +
              p" [FUSION] [${node_name}] [Cycle: ${cycleCount}]\n")
    }
  }

  /*==========================================*
   *            Output Handshaking and Reset  *
   *==========================================*/
  InvalidOut()
  when(IsOutReady() & (state === s_COMPUTE)) {
    //if (Debug) getData(state)
    // Reset data
    state := s_idle

    // Valid out is a wire
    ValidOut()

    //Reset output
    Reset()
  }
  var signed = if (sign == true) "S" else "U"
  override val printfSigil = opCode + xlen +  "_" + signed + "_" + ID + ":"

  if (log == true && (comp contains "OP")) {
    val x = RegInit(0.U(xlen.W))
    x     := x + 1.U
  
    verb match {
      case "high"  => { }
      case "med"   => { }
      case "low"   => {
        printfInfo("\nCycle %d : \n{ \"Inputs\": {",x)
        printInValid()
        printf("},\n")
        printf("\"State\": {\"State\": \"%x\",",state)
        PrintOut()
        printf("},\n")
        printf("\"Outputs\": {\"Out\": %x}",io.Out(0).fire)
        printf("}\n")
       }
      case everythingElse => {}
    }
  }
  //def isDebug(): Boolean = {
   // Debug
  //}
}

class Chain(ID: Int, NumOps: Int, OpCodes: Array[String])(sign: Boolean)(implicit p: Parameters)
  extends FusedComputeNode(NumIns = NumOps + 1, NumOuts = NumOps + 1, ID = ID,OpCodes.mkString("_"))(sign)(p)
{

 // Declare chain of FUs 
  val FUs = for (i <- 0 until OpCodes.length) yield {
    val FU = Module(new UALU(xlen, OpCodes(i)))
    FU
  }

//  val FUs = for (i <- 0 until OpCodes.length) yield {
//   val FU = OpCodes(i) match {
//     case "Mul"=>() => new multiplier()
//     case _ => () => new  UALU(xlen, OpCodes(i))
//   }
//   FU
//   }


  FUs(0).io.in1 := InRegs(0).data
  FUs(0).io.in2 := InRegs(1).data
  val out_data_R = FUs(0).io.out
  io.Out(0).bits.data := out_data_R

  io.Out(0).bits.predicate := InRegs(0).predicate & InRegs(1).predicate
  io.Out(0).bits.taskID := InRegs(0).taskID

  for (i <- 1 until OpCodes.length)  {
      val out = RegNext(FUs(i-1).io.out)//need
      FUs(i).io.in1 := out
      FUs(i).io.in2 := InRegs(i+1).data
      val out1 = RegNext(FUs(i).io.out)
      io.Out(i).bits.data := FUs(i).io.out
      io.Out(i).bits.taskID := InRegs(i+1).taskID
      io.Out(i).bits.predicate := io.Out(i-1).bits.predicate & InRegs(i+1).predicate
    }

    io.Out((OpCodes.length)).bits.data := RegNext(FUs((OpCodes.length)-1).io.out)
    io.Out((OpCodes.length)).bits.predicate := io.Out((OpCodes.length)-2).bits.predicate & InRegs(OpCodes.length).predicate
    io.Out((OpCodes.length)).bits.taskID := InRegs(OpCodes.length).taskID


}




class FloatChain(ID: Int, NumOps: Int, OpCodes: Array[String])(sign: Boolean)(implicit p: Parameters)
  extends FusedComputeNode(NumIns = NumOps + 1, NumOuts = NumOps + 1, ID = ID,OpCodes.mkString("_"))(sign)(p)
{

  // Declare chain of FUs 
  val FUs = for (i <- 0 until OpCodes.length) yield {
    val FU = Module(new FPUALU(64, OpCodes(i), FType.D))
    FU
  }

  FUs(0).io.in1 := InRegs(0).data
  FUs(0).io.in2 := InRegs(1).data
  val out_data_R = FUs(0).io.out
  io.Out(0).bits.data := out_data_R

  io.Out(0).bits.predicate := InRegs(0).predicate & InRegs(1).predicate
  io.Out(0).bits.taskID := InRegs(0).taskID

  for (i <- 1 until OpCodes.length)  {
      val out = RegNext(FUs(i-1).io.out)//need
      FUs(i).io.in1 := out
      FUs(i).io.in2 := InRegs(i+1).data
      val out1 = RegNext(FUs(i).io.out)
      io.Out(i).bits.data := FUs(i).io.out
      io.Out(i).bits.taskID := InRegs(i+1).taskID
      io.Out(i).bits.predicate := io.Out(i-1).bits.predicate & InRegs(i+1).predicate
    }

    io.Out((OpCodes.length)).bits.data := RegNext(FUs((OpCodes.length)-1).io.out)
    io.Out((OpCodes.length)).bits.predicate := io.Out((OpCodes.length)-2).bits.predicate & InRegs(OpCodes.length).predicate
    io.Out((OpCodes.length)).bits.taskID := InRegs(OpCodes.length).taskID


}
