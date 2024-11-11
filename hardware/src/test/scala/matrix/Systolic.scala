package matrix

import chisel3._
import chisel3.util._
import chisel3.iotesters.{ChiselFlatSpec, Driver, OrderedDecoupledHWIOTester, PeekPokeTester}
import org.scalatest.{FlatSpec, Matchers}
import muxes._
import heteacc.config._
import utility._
import chipsalliance.rocketchip.config._

// class SystolicBaseTests(df: SystolicSquareBuffered[UInt])(implicit p: Parameters) extends PeekPokeTester(df) {
//   poke(df.io.activate, false.B)
//   // left * right
//   df.io.left.zipWithIndex.foreach { case (io, i) => poke(io, (i + 1).U) }
//   df.io.right.zipWithIndex.foreach { case (io, i) => poke(io, (i + 1).U) }
//   poke(df.io.activate, true.B)
//   step(1)
//   poke(df.io.activate, false.B)
//   step(1)
//   for (i <- 0 until 16) {
//     if (peek(df.io.output.valid) == 1) {
//       for (i <- 0 until df.N * df.N) {
//         print(peek(df.io.output.bits(i)) + ",")
//       }
//       print("\n")
//     }
//     step(1)
//   }
//   print("\n")
// }
class SystolicBaseTests(df: SystolicSquareBuffered[UInt])(implicit p: Parameters) extends PeekPokeTester(df) {
  poke(df.io.activate, false.B)
  
  // 初始化输入数据
  df.io.left.zipWithIndex.foreach { case (io, i) => poke(io, (i + 1).U) }
  df.io.right.zipWithIndex.foreach { case (io, i) => poke(io, (i + 1).U) }
  
  poke(df.io.activate, true.B) // 激活阵列
  step(1)                      // 等待一个时钟周期
  poke(df.io.activate, false.B) // 停止激活
  
  var totalCycles = 0 // 记录总时钟周期数
  var outputReady = false // 标记结果是否准备好

  // 监测输出结果
  while (!outputReady && totalCycles < 1000) { // 设置上限防止无限循环
    if (peek(df.io.output.valid) == 1) {
      // 如果输出有效，打印结果并标记完成
      for (i <- 0 until df.N * df.N) {
        print(peek(df.io.output.bits(i)) + ",")
      }
      print("\n")
      outputReady = true
    } else {
      step(1) // 等待一个时钟周期
      totalCycles += 1
    }
  }

  print(s"Total cycles taken: $totalCycles\n")
}


class Systolic_Tester extends FlatSpec with Matchers {
//   implicit val p = config.Parameters.root((new Mat_VecCoWithTestConfignfig).toInstance)
    implicit val p = new WithAccelConfig(HeteaccAccelParams())
    // implicit val p = config.Parameters.root((new Mat_VecConfig).toInstance
    // implicit val p = new WithAccelConfig ++ new WithTestConfig
  it should "Typ Compute Tester" in {
    chisel3.iotesters.Driver.execute(Array(
      
      // "-ll", "Info",
        "-tn", "matrix",
        "-tbn", "verilator",
        "-td", s"test_run_dir/matrix",
        "-tts", "0001",
        "--generate-vcd-output", "on"),

      () => new SystolicSquareBuffered(UInt(32.W), 16)) {
      c => new SystolicBaseTests(c)
    } should be(true)
  }
}
