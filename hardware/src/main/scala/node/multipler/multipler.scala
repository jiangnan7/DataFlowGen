package heteacc.mul

import chisel3._
import chisel3.util._

class multiplier extends Module {
    val io = IO(new Bundle{
    	//inputs
    	val in1 = Input(UInt(32.W))
    	val in2 = Input(UInt(32.W))
    	//outputs
    	val answer_high = Output(UInt(32.W))
    	val out = Output(UInt(32.W))
    })
    
    val partial_products = Wire(Vec(32, UInt(32.W)))

	//creating partial products
	for(i <- 0 to 31){
		partial_products(i) := Mux(io.in1(i) === 1.U, io.in2, 0.U)
	}
    
    //adders for 1st stage
    var stage1_adders_set1 = new Array[csa_3_input_n_bit](8)
    var stage1_adders_set2 = new Array[csa_3_input_n_bit](8)
    
    //output of stage1_adders_set1
    val stage1_adders_set1_Sout = Wire(Vec(8, UInt(32.W)))
    val stage1_adders_set1_Cout = Wire (Vec(8, UInt(32.W)))
    
    //output of stage1_adders_set2
    val stage1_adders_set2_Sout = Wire(Vec(8, UInt(32.W)))
    val stage1_adders_set2_Cout = Wire (Vec(8, UInt(32.W)))
    
    //output of stage1_adders
    val stage1_adders_Sout = Wire(Vec(8, UInt(35.W)))
    val stage1_adders_Cout = Wire (Vec(8, UInt(32.W)))
    
    for(i <- 0 to 7){
    
    	//println(i)
    
    	stage1_adders_set1(i) = Module(new csa_3_input_n_bit(32))
    	
    	stage1_adders_set1(i).io.A := Cat(0.U(1.W), partial_products(4*i + 0)(31, 1))
    	stage1_adders_set1(i).io.B := partial_products(4*i + 1)
    	stage1_adders_set1(i).io.Cin := Cat(partial_products(4*i + 2)(30, 0), 0.U(1.W))
    	
    	stage1_adders_set1_Sout(i) := stage1_adders_set1(i).io.Sout
    	stage1_adders_set1_Cout(i) := stage1_adders_set1(i).io.Cout
    	
    	stage1_adders_set2(i) = Module(new csa_3_input_n_bit(32))
    	
    	stage1_adders_set2(i).io.A := Cat(partial_products(4*i + 2)(31), stage1_adders_set1_Sout(i)(31, 1))
    	stage1_adders_set2(i).io.B := stage1_adders_set1_Cout(i)
    	stage1_adders_set2(i).io.Cin := Cat(partial_products(4*i + 3)(30, 0), 0.U(1.W))
    	
    	stage1_adders_set2_Sout(i) := stage1_adders_set2(i).io.Sout
    	stage1_adders_set2_Cout(i) := stage1_adders_set2(i).io.Cout
    	
    	stage1_adders_Sout(i) := Cat(Cat(partial_products(4*i + 3)(31), stage1_adders_set2_Sout(i)), Cat(stage1_adders_set1_Sout(i)(0), partial_products(4*i + 0)(0)))
    	stage1_adders_Cout(i) := stage1_adders_set2_Cout(i)
    }
    
    //adders for 2nd stage
    var stage2_adders_set1 = new Array[csa_3_input_n_bit](4)
    var stage2_adders_set2 = new Array[csa_3_input_n_bit](4)
    
    //ouput of stage2_adders_set1
    val stage2_adders_set1_Sout = Wire(Vec(4, UInt(32.W)))
    val stage2_adders_set1_Cout = Wire(Vec(4, UInt(32.W)))
    
    //output of stage2_adders_set2
    val stage2_adders_set2_Sout = Wire(Vec(4, UInt(32.W)))
    val stage2_adders_set2_Cout = Wire (Vec(4, UInt(32.W)))
    
    //output of stage2_adders
    val stage2_adders_Sout = Wire(Vec(4, UInt(39.W)))
    val stage2_adders_Cout = Wire (Vec(4, UInt(36.W)))
    
    for(i <- 0 to 3){
    	
    	stage2_adders_set1(i) = Module(new csa_3_input_n_bit(32))
    	
    	stage2_adders_set1(i).io.A := stage1_adders_Sout(2*i)(34, 3)
    	stage2_adders_set1(i).io.B := stage1_adders_Cout(2*i)
    	stage2_adders_set1(i).io.Cin := Cat(stage1_adders_Sout(2*i + 1)(30, 0), 0.U(1.W))
    	
    	stage2_adders_set1_Sout(i) := stage2_adders_set1(i).io.Sout
    	stage2_adders_set1_Cout(i) := stage2_adders_set1(i).io.Cout
    	
    	stage2_adders_set2(i) = Module(new csa_3_input_n_bit(32))
    	
    	stage2_adders_set2(i).io.A := Cat(stage1_adders_Sout(2*i + 1)(34, 31), stage2_adders_set1_Sout(i)(31, 4))
    	stage2_adders_set2(i).io.B := Cat(0.U(3.W), stage2_adders_set1_Cout(i)(31, 3))
    	stage2_adders_set2(i).io.Cin := stage1_adders_Cout(2*i + 1)
    	
    	stage2_adders_set2_Sout(i) := stage2_adders_set2(i).io.Sout
    	stage2_adders_set2_Cout(i) := stage2_adders_set2(i).io.Cout
    	
    	stage2_adders_Sout(i) := Cat(stage2_adders_set2_Sout(i), Cat(stage2_adders_set1_Sout(i)(3, 0), stage1_adders_Sout(2*i)(2, 0)))
    	stage2_adders_Cout(i) := Cat(stage2_adders_set2_Cout(i), Cat(0.U(1.W), stage2_adders_set1_Cout(i)(2, 0)))
    }
    
    //adders for 3rd stage
    var stage3_adders_set1 = new Array[csa_3_input_n_bit](2)
    var stage3_adders_set2 = new Array[csa_3_input_n_bit](2)
    
    //ouput of stage3_adders_set1
    val stage3_adders_set1_Sout = Wire(Vec(2, UInt(32.W)))
    val stage3_adders_set1_Cout = Wire(Vec(2, UInt(32.W)))
    
    //output of stage3_adders_set2
    val stage3_adders_set2_Sout = Wire(Vec(2, UInt(35.W)))
    val stage3_adders_set2_Cout = Wire (Vec(2, UInt(35.W)))
    
    //output of stage3_adders
    val stage3_adders_Sout = Wire(Vec(2, UInt(48.W)))
    val stage3_adders_Cout = Wire (Vec(2, UInt(44.W)))
    
    for(i <- 0 to 1){
    	stage3_adders_set1(i) = Module(new csa_3_input_n_bit(32))
    	
    	stage3_adders_set1(i).io.A := Cat(0.U(1.W), stage2_adders_Sout(2*i)(38, 8))
    	stage3_adders_set1(i).io.B := stage2_adders_Cout(2*i)(35, 4)
    	stage3_adders_set1(i).io.Cin := stage2_adders_Sout(2*i + 1)(31, 0)
    	
    	stage3_adders_set1_Sout(i) := stage3_adders_set1(i).io.Sout
    	stage3_adders_set1_Cout(i) := stage3_adders_set1(i).io.Cout
    	
    	stage3_adders_set2(i) = Module(new csa_3_input_n_bit(35))
    	
    	stage3_adders_set2(i).io.A := Cat(stage2_adders_Sout(2*i + 1)(38, 32), stage3_adders_set1_Sout(i)(31, 4))
    	stage3_adders_set2(i).io.B := Cat(0.U(6.W), stage3_adders_set1_Cout(i)(31, 3))
    	stage3_adders_set2(i).io.Cin := stage2_adders_Cout(2*i + 1)(34, 0)
    	
    	stage3_adders_set2_Sout(i) := stage3_adders_set2(i).io.Sout
    	stage3_adders_set2_Cout(i) := stage3_adders_set2(i).io.Cout
    	
    	stage3_adders_Sout(i) := Cat(Cat(stage2_adders_Cout(2*i + 1)(35), stage3_adders_set2_Sout(i)) , Cat(stage3_adders_set1_Sout(i)(3, 0), stage2_adders_Sout(2*i)(7, 0)))
    	stage3_adders_Cout(i) := Cat(Cat(Cat(stage3_adders_set2_Cout(i), 0.U(1.W)), Cat(stage3_adders_set1_Cout(i)(2, 0), 0.U(1.W))), stage2_adders_Cout(2*i)(3, 0))
    }
    
    val stage4_adders_set1 = Module(new csa_3_input_n_bit(32))
    
    stage4_adders_set1.io.A := stage3_adders_Sout(0)(47, 16)
    stage4_adders_set1.io.B := stage3_adders_Cout(0)(43, 12)
    stage4_adders_set1.io.Cin := stage3_adders_Sout(1)(31, 0)
    
    val stage4_adders_set1_Sout = stage4_adders_set1.io.Sout
    val stage4_adders_set1_Cout = stage4_adders_set1.io.Cout
    
    val stage4_adders_set2 = Module(new csa_3_input_n_bit(44))
    
    stage4_adders_set2.io.A := Cat(stage3_adders_Sout(1)(47, 32), stage4_adders_set1_Sout(31, 4))
    stage4_adders_set2.io.B := Cat(0.U(15.W), stage4_adders_set1_Cout(31, 3))
    stage4_adders_set2.io.Cin := stage3_adders_Cout(1)(43, 0)
    
    val stage4_adders_set2_Sout = stage4_adders_set2.io.Sout
    val stage4_adders_set2_Cout = stage4_adders_set2.io.Cout
    
    val stage4_adders_Sout = Cat(stage4_adders_set2_Sout, Cat(stage4_adders_set1_Sout(3, 0), stage3_adders_Sout(0)(15, 0)))
    val stage4_adders_Cout = Cat(Cat(Cat(stage4_adders_set2_Cout(42, 0), 0.U(1.W)), Cat(stage4_adders_set1_Cout(2, 0), 0.U(1.W))), stage3_adders_Cout(0)(11, 0))
    
    var full_adders = new Array[cla_8bit](15)
    
    full_adders(0) = Module(new cla_8bit())
    
    full_adders(0).io.Cin := 0.U
    full_adders(0).io.A := Cat(stage4_adders_Cout(3, 0), 0.U(4.W))
    full_adders(0).io.B := stage4_adders_Sout(7, 0)
    
    for(i <- 1 to 7){
    	full_adders(2*i - 1) = Module(new cla_8bit())
    	full_adders(2*i) = Module(new cla_8bit())
    	
    	full_adders(2*i - 1).io.Cin := 0.U
    	full_adders(2*i - 1).io.A := stage4_adders_Cout(i*8 + 3, i*8 - 4)
    	full_adders(2*i - 1).io.B := stage4_adders_Sout(i*8 + 7, i*8)
    	
    	full_adders(2*i).io.Cin := 1.U
    	full_adders(2*i).io.A := stage4_adders_Cout(i*8 + 3, i*8 - 4)
    	full_adders(2*i).io.B := stage4_adders_Sout(i*8 + 7, i*8)
    }
    
    //all cout of adders with 0 carry in
    val cin_0 = Cat(full_adders(13).io.Cout, full_adders(11).io.Cout, full_adders(9).io.Cout, full_adders(7).io.Cout, full_adders(5).io.Cout, full_adders(3).io.Cout, full_adders(1).io.Cout, full_adders(0).io.Cout)
    //all cout of adders with 1 carry in
    val cin_1 = Cat(full_adders(14).io.Cout, full_adders(12).io.Cout, full_adders(10).io.Cout, full_adders(8).io.Cout, full_adders(6).io.Cout, full_adders(4).io.Cout, full_adders(2).io.Cout, full_adders(0).io.Cout)
    
    val cin = Array(cin_0, cin_1)
    
    val answer_7_0 = full_adders(0).io.Sout
    val answer_15_8 = Mux(cin_1(0).asBool, full_adders(2).io.Sout, full_adders(1).io.Sout)
    val check_23_16 = (cin_1(0)&cin_1(1)) | ((~cin_1(0))&cin_0(1))
    val answer_23_16 = Mux(check_23_16.asBool, full_adders(4).io.Sout, full_adders(3).io.Sout)
    val check_31_24 = (cin_1(0)&cin_1(1)&cin_1(2)) | ((~cin_1(0))&cin_0(1)&cin_1(2)) | (cin_1(0)&(~cin_1(1))&cin_0(2)) | ((~cin_1(0))&(~cin_0(1))&cin_0(2))
    val answer_31_24 = Mux(check_31_24.asBool, full_adders(6).io.Sout, full_adders(5).io.Sout)
    
    
    val conditions_39_32 = new Array[chisel3.UInt](32)//Wire(Vec(32, UInt(1.W)))
    val product_results_39_32 = Wire(Vec(8, UInt(1.W)))
    
    var stringIteral:String = ""
    /*
    var a = new Array[chisel3.UInt](2)
    a(0) = Wire(UInt(3.W))
    a(0) := 0.U
    a(1) = Wire(UInt(8.W))
    a(1) := answer_7_0
    */
    //val b = Cat(a)
    
    for(i <- 0 to 7){
    
    	stringIteral = i.toBinaryString.reverse.padTo(3, '0').reverse
    	
    	conditions_39_32(4*i) = Wire(UInt(1.W))
    	conditions_39_32(4*i + 1) = Wire(UInt(1.W))
    	conditions_39_32(4*i + 2) = Wire(UInt(1.W))
    	conditions_39_32(4*i + 3) = Wire(UInt(1.W))
    	
    	//making the product
    	if(stringIteral(0) == '1'){
    		conditions_39_32(4*i) := cin_1(0)
    	}else{
    		conditions_39_32(4*i) := ~cin_1(0)
    	}
    	
    	for(j <- 1 to 2){
    		if(stringIteral(j) == '1'){
    			conditions_39_32(4*i + j) := cin(stringIteral(j-1).toString.toInt)(j)
    		}else{
    			conditions_39_32(4*i + j) := ~cin(stringIteral(j-1).toString.toInt)(j)
    		}
    		//println(conditions_39_32(4*i + j))
    	}
    	/*
    	if(stringIteral(1) == '1'){
    		conditions_39_32(4*i + 1) := cin(stringIteral(0).toString.toInt)(1)
    	}else{
    		conditions_39_32(4*i + 1) := ~cin(stringIteral(0).toString.toInt)(1)
    	}
    	
    	if(stringIteral(2) == '1'){
    		conditions_39_32(4*i + 2) := cin(stringIteral(1).toString.toInt)(2)
    	}else{
    		conditions_39_32(4*i + 2) := ~cin(stringIteral(1).toString.toInt)(2)
    	}
    	*/
    	conditions_39_32(4*i + 3) := cin(stringIteral(2).toString.toInt)(3)
    	//println(stringIteral(2).toString.toInt.toString)
    	
    	product_results_39_32(i) := Cat(conditions_39_32.slice(4*i, 4*i + 4)).andR.asUInt
    	
    	//product_results_39_32(i) := Cat(conditions_39_32(4*i), conditions_39_32(4*i + 1), conditions_39_32(4*i + 2), conditions_39_32(4*i + 3)).andR.asUInt
    }
    
    val check_39_32 = Cat(product_results_39_32).orR
    val answer_39_32 = Mux(check_39_32, full_adders(8).io.Sout, full_adders(7).io.Sout)
    
    val conditions_47_40 = new Array[chisel3.UInt](80)//Wire(Vec(32, UInt(1.W)))
    val product_results_47_40 = Wire(Vec(16, UInt(1.W)))
    
    for(i <- 0 to 15){
    
    	stringIteral = i.toBinaryString.reverse.padTo(4, '0').reverse
    	
    	for(j <- 0 to 4){
    		conditions_47_40(5*i + j) = Wire(UInt(1.W))
    	}
    	
    	//making the product
    	if(stringIteral(0) == '1'){
    		conditions_47_40(5*i) := cin_1(0)
    	}else{
    		conditions_47_40(5*i) := ~cin_1(0)
    	}
    	
    	for(j <- 1 to 3){
    		if(stringIteral(j) == '1'){
    			conditions_47_40(5*i + j) := cin(stringIteral(j-1).toString.toInt)(j)
    		}else{
    			conditions_47_40(5*i + j) := ~cin(stringIteral(j-1).toString.toInt)(j)
    		}
    	}
    	conditions_47_40(5*i + 4) := cin(stringIteral(3).toString.toInt)(4)
    	
    	product_results_47_40(i) := Cat(conditions_47_40.slice(5*i, 5*i + 5)).andR.asUInt
    }
    
    val check_47_40 = Cat(product_results_47_40).orR
    val answer_47_40 = Mux(check_47_40, full_adders(10).io.Sout, full_adders(9).io.Sout)
    
    val conditions_55_48 = new Array[chisel3.UInt](32*6)//Wire(Vec(32, UInt(1.W)))
    val product_results_55_48 = Wire(Vec(32, UInt(1.W)))
    
    for(i <- 0 to 31){
    
    	stringIteral = i.toBinaryString.reverse.padTo(5, '0').reverse
    	
    	for(j <- 0 to 5){
    		conditions_55_48(6*i + j) = Wire(UInt(1.W))
    	}
    	
    	//making the product
    	if(stringIteral(0) == '1'){
    		conditions_55_48(6*i) := cin_1(0)
    	}else{
    		conditions_55_48(6*i) := ~cin_1(0)
    	}
    	
    	for(j <- 1 to 4){
    		if(stringIteral(j) == '1'){
    			conditions_55_48(6*i + j) := cin(stringIteral(j-1).toString.toInt)(j)
    		}else{
    			conditions_55_48(6*i + j) := ~cin(stringIteral(j-1).toString.toInt)(j)
    		}
    	}
    	conditions_55_48(6*i + 5) := cin(stringIteral(4).toString.toInt)(5)
    	
    	product_results_55_48(i) := Cat(conditions_55_48.slice(6*i, 6*i + 6)).andR.asUInt
    }
    
    val check_55_48 = Cat(product_results_55_48).orR
    val answer_55_48 = Mux(check_55_48, full_adders(12).io.Sout, full_adders(11).io.Sout)
    
    val conditions_63_56 = new Array[chisel3.UInt](64*7)//Wire(Vec(32, UInt(1.W)))
    val product_results_63_56 = Wire(Vec(64, UInt(1.W)))
    
    for(i <- 0 to 63){
    
    	stringIteral = i.toBinaryString.reverse.padTo(6, '0').reverse
    	
    	for(j <- 0 to 6){
    		conditions_63_56(7*i + j) = Wire(UInt(1.W))
    	}
    	
    	//making the product
    	if(stringIteral(0) == '1'){
    		conditions_63_56(7*i) := cin_1(0)
    	}else{
    		conditions_63_56(7*i) := ~cin_1(0)
    	}
    	
    	for(j <- 1 to 5){
    		if(stringIteral(j) == '1'){
    			conditions_63_56(7*i + j) := cin(stringIteral(j-1).toString.toInt)(j)
    		}else{
    			conditions_63_56(7*i + j) := ~cin(stringIteral(j-1).toString.toInt)(j)
    		}
    	}
    	conditions_63_56(7*i + 6) := cin(stringIteral(5).toString.toInt)(6)
    	
    	product_results_63_56(i) := Cat(conditions_63_56.slice(7*i, 7*i + 7)).andR.asUInt
    }
    
    val check_63_56 = Cat(product_results_63_56).orR
    val answer_63_56 = Mux(check_63_56, full_adders(14).io.Sout, full_adders(13).io.Sout)
    
    val answer = stage4_adders_Sout + Cat(stage4_adders_Cout, 0.U(4.W))
    
	io.answer_high := Cat(answer_63_56, answer_55_48, answer_47_40, answer_39_32)
    io.out := Cat(answer_31_24, answer_23_16, answer_15_8, answer_7_0)
}

// object multiplier extends App{
//     (new chisel3.stage.ChiselStage).emitVerilog(new multiplier())
// }
