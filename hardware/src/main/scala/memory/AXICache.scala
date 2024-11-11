package heteacc.memory.cache


import chisel3._
import chisel3.util._
import chipsalliance.rocketchip.config._
import heteacc.config._
import utility._
import heteacc.interfaces._
import heteacc.interfaces.axi._
import heteacc.junctions._

trait HasCacheAccelParams extends HasAccelParams with HasAccelShellParams {

  val nWays = accelParams.nways
  val nSets = accelParams.nsets
  val bBytes = accelParams.cacheBlockBytes
  val bBits = bBytes << 3
  val blen = log2Ceil(bBytes)
  val slen = log2Ceil(nSets)
  val taglen = xlen - (slen + blen)
  val nWords = bBits / xlen
  val wBytes = xlen / 8
  val byteOffsetBits = log2Ceil(wBytes)
  val dataBeats = (bBits + memParams.dataBits - 1) / memParams.dataBits
}