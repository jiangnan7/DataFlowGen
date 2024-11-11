package heteacc.mul
import chisel3._
import chisel3.util._
// import chisel3.Driver

class csa_3_input_32_bit extends Module {
    val io = IO(new Bundle{
    	//inputs
    	val A = Input(UInt(32.W))
    	val B = Input(UInt(32.W))
    	val Cin = Input(UInt(32.W))
    	//outputs
    	val Sout = Output(UInt(32.W))
    	val Cout = Output(UInt(32.W))
    })
    
    val A = io.A
    val B = io.B
    val Cin = io.Cin
    
	io.Sout := (~io.Cin)&(io.A^io.B) | io.Cin&(~(io.A^io.B))
    io.Cout := (~io.Cin)&(io.A&io.B) | io.Cin&(io.A|io.B)
}

// object csa_3_input_32_bit extends App{
//     (new chisel3.stage.ChiselStage).emitVerilog(new csa_3_input_32_bit())
// }
