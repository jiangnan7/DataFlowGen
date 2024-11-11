package matrix


import chisel3._
import chisel3.util._
import heteacc.interfaces.axi.AXIParams
import utility._
//import examples._
import heteacc.junctions._
import heteacc.node._
import chisel3._
import chisel3.Module
import chisel3.util._
import org.scalatest.{FlatSpec, Matchers}
import heteacc.config._
import heteacc.node._
import heteacc.interfaces._
import muxes._
import utility._
import heteacc.junctions._
import chipsalliance.rocketchip.config._
import heteacc.config._
import heteacc.interfaces._
import NastiConstants._
import chipsalliance.rocketchip.config._
import heteacc.config._
import utility._
import heteacc.memory.cache.HasCacheAccelParams
import heteacc.interfaces.axi._


object MAC {

  trait OperatorMAC[T] {
    def mac(l: T, r: T, c: T)(implicit p: Parameters): T
  }

  object OperatorMAC {

    implicit object UIntMAC extends OperatorMAC[UInt] {
      def mac(l: UInt, r: UInt, c: UInt)(implicit p: Parameters): UInt = {
        val x     = Wire(l.cloneType)
        val FXALU = Module(new UALU(32, "Mac"))
        FXALU.io.in1 := l
        FXALU.io.in2 := r
        FXALU.io.in3.get := c
        x := FXALU.io.out.asTypeOf(l)
        x
      }
    }
  }

  def mac[T](l: T, r: T, c: T)(implicit op: OperatorMAC[T], p: Parameters): T = op.mac(l, r, c)
}