package heteacc.mul
import chisel3._
import chisel3.util._

class full_adder extends Module {
    val io = IO(new Bundle{
    	//inputs
    	val A = Input(UInt(1.W))
    	val B = Input(UInt(1.W))
    	val Cin = Input(UInt(1.W))
    	//outputs
    	val Sout = Output(UInt(1.W))
    	val Cout = Output(UInt(1.W))
    })
    io.Sout := (~io.Cin)&(io.A^io.B) | io.Cin&(~(io.A^io.B))
    io.Cout := (~io.Cin)&(io.A&io.B) | io.Cin&(io.A|io.B)
}

