package heteacc.mul
import chisel3._
import chisel3.util._
// import chisel3.Driver

class cla_8bit() extends Module {
    val io = IO(new Bundle{
    	//inputs
    	val A = Input(UInt(8.W))
    	val B = Input(UInt(8.W))
    	val Cin = Input(UInt(1.W))
    	//outputs
    	val Sout = Output(UInt(8.W))
    	val Cout = Output(UInt(1.W))
    })
    
    val A = io.A
    val B = io.B
    val Cin = io.Cin
    
    val P = A^B
    val G = A&B
    
    //val C = Wire(UInt(8.W))
    
    //carry look ahead
    val C_1 = G(0) | (P(0) & Cin)
    val C_2 = G(1) | (P(1) & G(0)) | (P(1,0).andR & Cin)
    val C_3 = G(2) | (P(2) & G(1)) | (P(2,1).andR & G(0)) | (P(2,0).andR & Cin)
    val C_4 = G(3) | (P(3) & G(2)) | (P(3,2).andR & G(1)) | (P(3,1).andR & G(0)) | (P(3,0).andR & Cin)
    val C_5 = G(4) | (P(4) & G(3)) | (P(4,3).andR & G(2)) | (P(4,2).andR & G(1)) | (P(4,1).andR & G(0)) | (P(4,0).andR & Cin)
    val C_6 = G(5) | (P(5) & G(4)) | (P(5,4).andR & G(3)) | (P(5,3).andR & G(2)) | (P(5,2).andR & G(1)) | (P(5,1).andR & G(0)) | (P(5,0).andR & Cin)
    val C_7 = G(6) | (P(6) & G(5)) | (P(6,5).andR & G(4)) | (P(6,4).andR & G(3)) | (P(6,3).andR & G(2)) | (P(6,2).andR & G(1)) | (P(6,1).andR & G(0)) | (P(6,0).andR & Cin)
    val C_8 = G(7) | (P(7) & G(6)) | (P(7,6).andR & G(5)) | (P(7,5).andR & G(4)) | (P(7,4).andR & G(3)) | (P(7,3).andR & G(2)) | (P(7,2).andR & G(1)) | (P(7,1).andR & G(0)) | (P(7,0).andR & Cin)
    
    val C = Cat(C_7, C_6, C_5, C_4, C_3, C_2, C_1, Cin)
    
	io.Sout := (~C)&(A^B) | C&(~(A^B))
    io.Cout := C_8
}

// object cla_8bit extends App{
//     (new chisel3.stage.ChiselStage).emitVerilog(new cla_8bit())
// }
