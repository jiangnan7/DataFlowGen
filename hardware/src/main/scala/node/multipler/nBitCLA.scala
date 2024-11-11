package heteacc.mul
import chisel3._
import chisel3.util._
// import chisel3.Driver
//import org.scalatest.Assertions._

class CLA_adder(width: Int) extends Module {
    val io = IO(new Bundle{
        val A           = Input(UInt(width.W))
        val B           = Input(UInt(width.W))
        val Cin    = Input(UInt(1.W))
        val sum         = Output(UInt(width.W))
        val overflow    = Output(Bool())    
    })
    
    assert(width >= 2)

    val A = io.A
    val B = io.B
    val Cin = io.Cin

    val G = Wire(UInt(width.W))
    
    val P = A ^ B  
    G := A & B

    val partialProducts = new Array[chisel3.UInt]((width+2)*(width+1)/2 - 1)

    for(i <- 0 until (width+2)*(width+1)/2 - 1){
        partialProducts(i) = Wire(UInt(1.W))
    }

    partialProducts(0) := G(0)
    partialProducts(1) := P(0) & io.Cin
    partialProducts(2) := G(1)
    partialProducts(3) := P(1) & G(0)
    partialProducts(4) := P(1,0).andR & io.Cin

    var i: Int = 5

    for(carry_index <- 3 to width){
        partialProducts(i) := G(carry_index - 1)
        partialProducts(i+1) := P(carry_index - 1) & G(carry_index - 2)
        partialProducts(i+2) := P(carry_index - 1, 0).andR & io.Cin
        
        i = i + 3

        for(loop_index <- 0 to carry_index - 3){
            partialProducts(i) := P(carry_index - 1, loop_index+1).andR & G(loop_index)
            i = i + 1
        }
    }

    val generated_carry = new Array[chisel3.UInt](width)

    var index_begin:Int = 0
    var index_increment:Int = 2

    for(carry_index <- 0 until width){
        generated_carry(carry_index) = Wire(UInt(1.W))

        generated_carry(carry_index) := Cat(partialProducts.slice(index_begin, index_begin + index_increment)).orR
        index_begin = index_begin + index_increment
        index_increment = index_increment + 1
    }

    val C = Cat(Cat(generated_carry.slice(0, width-1).reverse), Cin)

    io.sum := (~C)&(A^B) | C&(~(A^B))
    io.overflow := generated_carry(width - 1).asBool
}

// object CLA_adder extends App{
//     (new chisel3.stage.ChiselStage).emitVerilog(new CLA_adder(32))
// }