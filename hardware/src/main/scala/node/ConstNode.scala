package heteacc.node

import chisel3._
import chisel3.Module
import chipsalliance.rocketchip.config._
import heteacc.config._
import heteacc.interfaces._
import util._
import utility.UniformPrintfs

class Constant(value: BigInt, ID: Int) (implicit val p: Parameters,
                     name: sourcecode.Name,
                     file: sourcecode.File)
  extends Module with HasAccelParams with UniformPrintfs {
  val io = IO(new Bundle {
    val enable = Flipped(Decoupled(new ControlBundle))
    val Out = Decoupled(new DataBundle)
  })
  
  val enable_R = RegInit(ControlBundle.default)
  val enable_valid_R = RegInit(false.B)

  val output_value = value.asSInt(xlen.W).asUInt
  val taskID = io.enable.bits.taskID
  val predicate = io.enable.bits.control
  


  io.enable.ready := ~enable_valid_R
  io.Out.valid := enable_valid_R
  io.Out.bits := DataBundle(output_value, taskID, predicate)

}

class ConstFastNode(value: BigInt, ID: Int)
                    (implicit val p: Parameters,
                     name: sourcecode.Name,
                     file: sourcecode.File)
  extends Module with HasAccelParams with UniformPrintfs {

  val io = IO(new Bundle {
    val enable = Flipped(Decoupled(new ControlBundle))
    val Out = Decoupled(new DataBundle)
  })

  /*===========================================*
   *            Registers                      *
   *===========================================*/
  val enable_R = RegInit(ControlBundle.default)
  val enable_valid_R = RegInit(false.B)

  val taskID = Mux(enable_valid_R, enable_R.taskID, io.enable.bits.taskID)
  val predicate = Mux(enable_valid_R, enable_R.control, io.enable.bits.control)

  /*===============================================*
   *            Latch inputs. Wire up output       *
   *===============================================*/


  val output_value = value.asSInt(xlen.W).asUInt
  // val output_value = value.asSInt(16.W).asUInt
  io.enable.ready := ~enable_valid_R
  io.Out.valid := enable_valid_R

  io.Out.bits := DataBundle(output_value, taskID, predicate)

  /*============================================*
   *            ACTIONS (possibly dangerous)    *
   *============================================*/
  val s_idle :: s_fire :: Nil = Enum(2)
  val state = RegInit(s_idle)

  switch(state) {
    is(s_idle) {
      when(io.enable.fire) {
        io.Out.valid := true.B
        when(io.Out.fire) {
          state := s_idle
        }.otherwise {
          state := s_fire
          enable_valid_R := true.B
          enable_R <> io.enable.bits
        }
      }
    }
    is(s_fire) {
      when(io.Out.fire) {
        //Restart the registers
        enable_R := ControlBundle.default
        enable_valid_R := false.B
        state := s_idle
      }
    }
  }

}

// class ConstFastFloatNode(value: Double, ID: Int)
//                    (implicit val p: Parameters,
//                     name: sourcecode.Name,
//                     file: sourcecode.File)
//   extends Module with HasAccelParams with UniformPrintfs {

//   val io = IO(new Bundle {
//     val enable = Flipped(Decoupled(new ControlBundle))
//     val Out = Decoupled(new DataBundle)
//   })

//   /*===========================================*
//    *            Registers                      *
//    *===========================================*/
//   val enable_R = RegInit(ControlBundle.default)
//   val enable_valid_R = RegInit(false.B)

//   val taskID = Mux(enable_valid_R, enable_R.taskID, io.enable.bits.taskID)
//   val predicate = Mux(enable_valid_R, enable_R.control, io.enable.bits.control)

//   /*===============================================*
//    *            Latch inputs. Wire up output       *
//    *===============================================*/
//   def ieee(value: Long): UInt = {
//     val bits = Wire(UInt(xlen.W))
//     bits := value.U(xlen.W)
//     bits
//   }
//   // 将浮点数转换为 IEEE 754 二进制表示的 UInt
//   val output_value = Wire(UInt(xlen.W))
//   output_value := ieee(java.lang.Double.doubleToLongBits(value))



//   io.enable.ready := ~enable_valid_R
//   io.Out.valid := enable_valid_R

//   io.Out.bits := DataBundle(output_value, taskID, predicate)

//   /*============================================*
//    *            ACTIONS (possibly dangerous)    *
//    *============================================*/
//   val s_idle :: s_fire :: Nil = Enum(2)
//   val state = RegInit(s_idle)

//   switch(state) {
//     is(s_idle) {
//       when(io.enable.fire) {
//         io.Out.valid := true.B
//         when(io.Out.fire) {
//           state := s_idle
//         } .otherwise {
//           state := s_fire
//           enable_valid_R := true.B
//           enable_R <> io.enable.bits
//         }
//       }
//     }
//     is(s_fire) {
//       when(io.Out.fire) {
//         // 重置寄存器
//         enable_R := ControlBundle.default
//         enable_valid_R := false.B
//         state := s_idle
//       }
//     }
//   }

//   /* 工具函数：将 Double 转换为 UInt */
//   def ieee(value: Long): UInt = {
//     val bits = Wire(UInt(xlen.W))
//     bits := value.U(xlen.W)
//     bits
//   }
// }
