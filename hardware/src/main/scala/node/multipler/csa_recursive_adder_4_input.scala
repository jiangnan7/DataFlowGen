package heteacc.mul
import chisel3._
import chisel3.util._
// import chisel3.Driver

class csa_recursive_adder_4_input(width: Int) extends Module {
    val io = IO(new Bundle{
    	//inputs
    	val P_0 = Input(UInt(width.W))
    	val P_1 = Input(UInt(width.W))
    	val P_2 = Input(UInt(width.W))
    	val P_3 = Input(UInt(width.W))
    	//outputs
    	val Sout = Output(UInt((width+3).W))
    	val Cout = Output(UInt((width+3).W))
    })
    
    val P_0 = io.P_0
    val P_1 = io.P_1
    val P_2 = io.P_2
    val P_3 = io.P_3
    
    val csa_adder_1 = Module(new csa_3_input_n_bit(width))
    val csa_adder_2 = Module(new csa_3_input_n_bit(width))
    
    csa_adder_1.io.A := Cat(0.U(1.W), P_0(width-1, 1))
    csa_adder_1.io.B := P_1
    csa_adder_1.io.Cin := Cat(P_2(width-2, 0), 0.U(1.W))
    
    val csa_adder_1_Sout = csa_adder_1.io.Sout
    val csa_adder_1_Cout = csa_adder_1.io.Cout
    
    csa_adder_2.io.A := Cat(P_2(width-1), csa_adder_1_Sout(width-1, 1))
    csa_adder_2.io.B := csa_adder_1_Cout
    csa_adder_2.io.Cin := Cat(P_3(width-2, 0), 0.U(1.W))
    
    val csa_adder_2_Sout = csa_adder_2.io.Sout
    val csa_adder_2_Cout = csa_adder_2.io.Cout
    
	io.Sout := Cat(P_3(width-1), Cat(csa_adder_2_Sout, Cat(csa_adder_1_Sout(0), P_0(0))))
    io.Cout := Cat(csa_adder_2_Cout, 0.U(3.W))
}

// object csa_recursive_adder_4_input extends App{
//     (new chisel3.stage.ChiselStage).emitVerilog(new csa_recursive_adder_4_input(32))
// }
