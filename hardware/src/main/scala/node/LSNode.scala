package heteacc.node

import chisel3._
import chisel3.util._
import chipsalliance.rocketchip.config._
import chisel3.util.experimental.BoringUtils
import heteacc.interfaces._
import utility.Constants._


import heteacc.config._
import chisel3.Module
import util._
import chipsalliance.rocketchip.config._
import chisel3.util.experimental.loadMemoryFromFileInline


trait InitMem {
  def mem: SyncReadMem[UInt]

  def initMem(memoryFile: String) = loadMemoryFromFileInline(mem, memoryFile)
}

class ReadWriteMem(size: Int, width: Int = 32, portNum: Int = 1) extends MultiIOModule with InitMem {

  val mem = SyncReadMem(size, UInt(width.W))
}

class insideMemory(size: Int, width: Int = 32, portNum: Int = 1) extends MultiIOModule with InitMem {
  val mem = SyncReadMem(size, UInt(width.W))
  private val addrWidth = log2Ceil(size)
  val r_en = IO(Input(Bool()))
  val w_en = IO(Input(Bool()))
  val addr = IO(Input(UInt(addrWidth.W)))
  val w_data = IO(Input(UInt(width.W)))
  val r_data = IO(Output(UInt(width.W)))
  r_data := DontCare

  when(r_en || w_en) {
    val rwPort = mem(addr)
    when(w_en) {
      rwPort := w_data
    }.otherwise {
      r_data := rwPort
    }
  }
}

class TEHB(size: Int = 32)(implicit val p: Parameters) extends MultiIOModule {
  val dataIn = IO(Flipped(DecoupledIO(new DataBundle)))
  val dataOut = IO(DecoupledIO(new DataBundle))

  private val full_reg = RegInit(Bool(), false.B)
  private val reg_en = Wire(Bool())
  private val mux_sel = Wire(Bool())
  private val data_reg = RegInit(UInt(size.W), 0.U)

  dataOut.valid := dataIn.valid | full_reg
  dataIn.ready := !full_reg
  reg_en := dataIn.ready & dataIn.valid & (!dataOut.ready)
  mux_sel := full_reg
  dataOut.bits.predicate <> DontCare
  dataOut.bits.taskID <> DontCare
  full_reg := dataOut.valid & (!dataOut.ready)

  when(reg_en) {
    data_reg := dataIn.bits.data
  }

  when(mux_sel) {
    dataOut.bits.data := data_reg
  }.otherwise {
    dataOut.bits := dataIn.bits
  }
  //  def apply(size : Int = 32)(in : DecoupledIO[UInt]) = {
  //    val tehb = Module(new TEHB(size))
  //    tehb.dataIn := in
  //    tehb.dataOut
  //  }
}


class MemoryEngine(Size: Int, ID: Int, NumRead: Int, NumWrite: Int)(implicit val p: Parameters) extends MultiIOModule  {
    
    val io = IO(new Bundle {

    val load_address  = Vec(NumRead, Flipped(DecoupledIO(new DataBundle)))
    val load_data = Vec(NumRead, DecoupledIO(new DataBundle))


    val store_address  = Vec(NumWrite, Flipped(DecoupledIO(new DataBundle)))
    val store_data = Vec(NumWrite, Flipped(DecoupledIO(new DataBundle)))
    
    // val finish = IO(Input(Bool()))
    // val rd = new CMEClientVector(NumRead)
    // val wr = new CMEClientVector(NumWrite)
  })
  for (i <- 0 until NumRead) {
    io.load_data(i).valid := false.B
    io.load_data(i).bits := DontCare
  }
  val mem = Module(new insideMemory(Size))

  def initMem(memoryFile: String) = mem.initMem(memoryFile)


  if (NumWrite == 0) {
    mem.addr := DontCare
    mem.r_en := false.B
    mem.w_data := DontCare
    mem.w_en := false.B
    val buffer = for (i <- 0 until NumRead) yield {
      val tehb = Module(new TEHB())
      tehb.dataIn.bits := DontCare
      tehb.dataIn.valid := false.B
      tehb.dataOut <> io.load_data(i)
      tehb
    }
    //    val load_valid = (load_address zip load_data).map(x => x._1.valid & x._2.ready)
    //    val load_valid = (load_address zip buffer).map(x => x._1.valid & x._2.dataOut.ready)
    val arb = Module(new Arbiter(new DataBundle, NumRead))
    arb.io.out.ready := true.B
    for (i <- 0 until NumRead) {
      //      arb.io.in(i).valid := load_valid(i)
      arb.io.in(i).valid := io.load_address(i).valid & buffer(i).dataIn.ready
      arb.io.in(i).bits := io.load_address(i).bits
      io.load_address(i).ready := arb.io.in(i).ready & buffer(i).dataIn.ready
      //      load_data(i).bits := DontCare
      //      load_data(i).valid := false.B
      //      buffer(i).dataOut <> load_data(i)
    }
    val select = Reg(UInt(log2Ceil(NumRead).W))
    val valid = RegInit(false.B)
    val data = RegInit(0.U(32.W))
    when(arb.io.out.valid) {
      mem.r_en := true.B
      mem.addr := io.load_address(arb.io.chosen).bits.data
      select := arb.io.chosen
    }
    valid := arb.io.out.valid
    //    load_data(select).valid := valid
    //    buffer(select).dataIn.valid := valid
    for (i <- 0 until NumRead) {
      when(i.U === select) {
        buffer(i).dataIn.valid := valid
      }
    }

    when(valid) {
      //      load_data(select).bits := mem.r_data
      for (i <- 0 until NumRead) {
        when(i.U === select) {
          buffer(i).dataIn.bits.data := mem.r_data
        }
      }
      data := mem.r_data
    }.otherwise {
      //      load_data(select).bits := data
      for (i <- 0 until NumRead) {
        when(i.U === select) {
          buffer(i).dataIn.bits.data := data
        }
      }
    }
  } else if (NumRead == 0 && NumWrite == 1){
    val join = Module(new Join())

    join.pValid(0) := io.store_address(0).valid
    join.pValid(1) := io.store_data(0).valid
    io.store_address(0).ready := join.ready(0)
    io.store_data(0).ready := join.ready(1)
    join.nReady := true.B
    val finish = false.B
    mem.r_en := !finish

    when(finish) {
      mem.addr := 32.U 
      mem.w_en := false.B
      mem.w_data := DontCare
    }.otherwise {
      mem.w_en := join.valid
      mem.addr := io.store_address(0).bits.data
      mem.w_data := io.store_data(0).bits.data
    }
  } else if (NumRead == 1 && NumWrite == 1){
    mem.addr := DontCare
    mem.r_en := false.B
    mem.w_data := DontCare
    mem.w_en := false.B
    io.load_address(0).ready := false.B
    io.store_data(0).ready := false.B
    io.store_address(0).ready := false.B
    when(io.store_address(0).valid) {
      val join = Module(new Join())

      join.pValid(0) := io.store_address(0).valid
      join.pValid(1) := io.store_data(0).valid
      io.store_address(0).ready := join.ready(0)
      io.store_data(0).ready := join.ready(1)
      join.nReady := true.B

      mem.w_en := join.valid
      mem.addr := io.store_address(0).bits.data
      mem.w_data := io.store_data(0).bits.data
    }.otherwise {
      val buffer = Module(new TEHB(32))
      buffer.dataIn.bits := DontCare
      buffer.dataIn.valid := false.B
      buffer.dataOut <> io.load_data(0)
      val arb = Module(new Arbiter(UInt(1.W), 1))
      arb.io.out.ready := true.B
      arb.io.in(0).valid := io.load_address(0).valid & buffer.dataIn.ready
      arb.io.in(0).bits := io.load_address(0).bits.data
      io.load_address(0).ready := arb.io.in(0).ready & buffer.dataIn.ready
      val valid = RegInit(false.B)
      val data = RegInit(0.U(32.W))
      when(arb.io.out.valid) {
        mem.r_en := true.B
        mem.addr := io.load_address(arb.io.chosen).bits.data
      }
      valid := arb.io.out.valid
      buffer.dataIn.valid := valid

      when(valid) {
        buffer.dataIn.bits.data := mem.r_data
        data := mem.r_data
      }.otherwise {
        buffer.dataIn.bits.data := data
      }
    }
  } else {

  }
}


class Load(NumOuts: Int, ID: Int, RouteID: Int)
          (implicit p: Parameters,
                     name: sourcecode.Name,
                     file: sourcecode.File)
                extends HandShakingNPS(NumOuts, ID)(new DataBundle)(p)
    // with HasAccelShellParams  with HasDebugCodes {
{
  // override lazy val io = IO(new LoadCCCIO(NumOuts))
  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  val (cycleCount, _) = Counter(true.B, 32 * 1024)
  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "


  val GepAddr = IO(Flipped(DecoupledIO(new DataBundle)))

  val address_out = IO(DecoupledIO(new DataBundle))
  val data_in = IO(Flipped(DecoupledIO(new DataBundle)))


  GepAddr <> address_out

  // io.data_in <> io.Out
   for (i <- 0 until NumOuts) {
    io.Out(i) <> data_in
    // io.Out(i).valid := io.data_in.valid
  }
  ValidOut()

  when(IsOutReady()){
    Reset()
    if (log) {
      printf("[LOG] " + "[" + module_name + "] [TID->%d] [LOAD] " + node_name + ": Output fired @ %d, Address:%d, Value: %d\n",
        enable_R.taskID, cycleCount, GepAddr.bits.data, data_in.bits.data)
    }
  }


}



class Store(NumOuts: Int, ID: Int, RouteID: Int)
          (implicit p: Parameters,
                      name: sourcecode.Name,
                      file: sourcecode.File)
                extends HandShakingNPS(NumOuts, ID)(new DataBundle)(p)
    with HasAccelShellParams  with HasDebugCodes {

  val node_name = name.value
  val module_name = file.value.split("/").tail.last.split("\\.").head.capitalize
  override val printfSigil = "[" + module_name + "] " + node_name + ": " + ID + " "
  val (cycleCount, _) = Counter(true.B, 32 * 1024)

  val GepAddr = IO(Flipped(DecoupledIO(new DataBundle)))
  val address_out = IO(DecoupledIO(new DataBundle))
  val inData = IO(Flipped(DecoupledIO(new DataBundle)))

  private val addrWidth = log2Ceil(32)

  private val join = Module(new Join(2))
  join.pValid(0) := GepAddr.valid
  join.pValid(1) := inData.valid

  join.nReady := address_out.ready & io.Out(0).ready

  GepAddr.ready := join.ready(0)
  inData.ready := join.ready(1)

  address_out.valid := join.valid
  io.Out(0).valid := join.valid

  address_out.bits := GepAddr.bits
  // data_out.bits := data_in.bits
  // val addr = Module(new ElasticBuffer(addrWidth))
  // addr.dataIn <> GepAddr
  // addr.dataOut <> address_out
  // val data = Module(new ElasticBuffer(32.U))
  // data.dataIn <> data_in
  // data.dataOut <> data_out
  
  for (i <- 0 until NumOuts) {
    io.Out(i) <> inData
    // io.Out(i).bits.data := data_in.bits.data
    // io.Out(i).bits.taskID := DontCare
    // io.Out(i).bits.predicate := DontCare
  }


  ValidOut()

  when(IsOutReady()){
    Reset()
    if (log) {
      printf(p"[LOG] [${module_name}] [TID: ${enable_R.taskID}] [STORE] " +
            p"[${node_name}] "+ 
            p"[Addr: ${Decimal(GepAddr.bits.data)}] " +
            p"[Data: ${Decimal(inData.bits.data)}] " +
            p"[Cycle: ${cycleCount}]\n")
    }
  }
       

}