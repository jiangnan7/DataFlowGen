package heteacc.memory

import chisel3._
import chisel3.util._
import chipsalliance.rocketchip.config._
import heteacc.config._
import heteacc.interfaces._
import utility._

import chisel3.experimental._
import chisel3.util.experimental.loadMemoryFromFileInline

trait InitSyncMem {
  def mem: SyncReadMem[UInt]

  def initMem(memoryFile: String) = loadMemoryFromFileInline(mem, memoryFile)
}

class InitMem(size: Int) extends MultiIOModule with InitSyncMem {

  val mem = SyncReadMem(size, UInt(32.W))

}