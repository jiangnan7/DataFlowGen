package heteacc.mul
import chisel3._
import chisel3.util._

class csa_3_input_n_bit(width: Int) extends Module {
    val io = IO(new Bundle{
    	//inputs
    	val A = Input(UInt(width.W))
    	val B = Input(UInt(width.W))
    	val Cin = Input(UInt(width.W))
    	//outputs
    	val Sout = Output(UInt(width.W))
    	val Cout = Output(UInt(width.W))
    })
    
    val A = io.A
    val B = io.B
    val Cin = io.Cin
    
	io.Sout := (~io.Cin)&(io.A^io.B) | io.Cin&(~(io.A^io.B))
    io.Cout := (~io.Cin)&(io.A&io.B) | io.Cin&(io.A|io.B)
}

