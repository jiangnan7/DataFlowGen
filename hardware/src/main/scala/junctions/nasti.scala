package heteacc.junctions
import Chisel._
import scala.math.max
import scala.collection.mutable.ArraySeq
import chipsalliance.rocketchip.config._
import utility._


// The code defines a parameter, module, and data structure
// for the Nasti (Native Interface) bus, a high-performance, 
// low-latency bus protocol for connecting processors to peripherals.
case object NastiKey extends Field[NastiParameters]

case class NastiParameters(dataBits: Int, addrBits: Int, idBits: Int)

trait HasNastiParameters {
  implicit val p: Parameters
  val nastiExternal = p(NastiKey)
  val nastiXDataBits = nastiExternal.dataBits
  val nastiWStrobeBits = nastiXDataBits / 8
  val nastiXAddrBits = nastiExternal.addrBits
  val nastiWIdBits = nastiExternal.idBits
  val nastiRIdBits = nastiExternal.idBits
  val nastiXIdBits = max(nastiWIdBits, nastiRIdBits)
  val nastiXUserBits = 1
  val nastiAWUserBits = nastiXUserBits
  val nastiWUserBits = nastiXUserBits
  val nastiBUserBits = nastiXUserBits
  val nastiARUserBits = nastiXUserBits
  val nastiRUserBits = nastiXUserBits
  val nastiXLenBits = 8
  val nastiXSizeBits = 3
  val nastiXBurstBits = 2
  val nastiXCacheBits = 4
  val nastiXProtBits = 3
  val nastiXQosBits = 4
  val nastiXRegionBits = 4
  val nastiXRespBits = 2

  def bytesToXSize(bytes: UInt) = MuxLookup(bytes, UInt("b111"), Array(
    UInt(1) -> UInt(0),
    UInt(2) -> UInt(1),
    UInt(4) -> UInt(2),
    UInt(8) -> UInt(3),
    UInt(16) -> UInt(4),
    UInt(32) -> UInt(5),
    UInt(64) -> UInt(6),
    UInt(128) -> UInt(7)))
}

abstract class NastiModule(implicit val p: Parameters) extends Module
  with HasNastiParameters
abstract class NastiBundle(implicit val p: Parameters) extends Bundle
  with HasNastiParameters

  
abstract class NastiChannel(implicit p: Parameters) extends NastiBundle()(p)
abstract class NastiMasterToSlaveChannel(implicit p: Parameters) extends NastiChannel()(p)
abstract class NastiSlaveToMasterChannel(implicit p: Parameters) extends NastiChannel()(p)

trait HasNastiMetadata extends HasNastiParameters {
  val addr   = UInt(width = nastiXAddrBits)
  val len    = UInt(width = nastiXLenBits)
  val size   = UInt(width = nastiXSizeBits)
  val burst  = UInt(width = nastiXBurstBits)
  val lock   = Bool()
  val cache  = UInt(width = nastiXCacheBits)
  val prot   = UInt(width = nastiXProtBits)
  val qos    = UInt(width = nastiXQosBits)
  val region = UInt(width = nastiXRegionBits)
}

trait HasNastiData extends HasNastiParameters {
  val data = UInt(width = nastiXDataBits)
  val last = Bool()
}

class NastiReadIO(implicit val p: Parameters) extends Bundle {
  val ar = Decoupled(new NastiReadAddressChannel)
  val r  = Decoupled(new NastiReadDataChannel).flip
}

class NastiWriteIO(implicit val p: Parameters) extends Bundle {
  val aw = Decoupled(new NastiWriteAddressChannel)
  val w  = Decoupled(new NastiWriteDataChannel)
  val b  = Decoupled(new NastiWriteResponseChannel).flip
}

class NastiIO(implicit val p: Parameters) extends Bundle {
  val aw = Decoupled(new NastiWriteAddressChannel)
  val w  = Decoupled(new NastiWriteDataChannel)
  val b  = Decoupled(new NastiWriteResponseChannel).flip
  val ar = Decoupled(new NastiReadAddressChannel)
  val r  = Decoupled(new NastiReadDataChannel).flip
}

class NastiAddressChannel(implicit p: Parameters) extends NastiMasterToSlaveChannel()(p)
    with HasNastiMetadata

class NastiResponseChannel(implicit p: Parameters) extends NastiSlaveToMasterChannel()(p) {
  val resp = UInt(width = nastiXRespBits)
}

class NastiWriteAddressChannel(implicit p: Parameters) extends NastiAddressChannel()(p) {
  val id   = UInt(width = nastiWIdBits)
  val user = UInt(width = nastiAWUserBits)
}

class NastiWriteDataChannel(implicit p: Parameters) extends NastiMasterToSlaveChannel()(p)
    with HasNastiData {
  val id   = UInt(width = nastiWIdBits)
  val strb = UInt(width = nastiWStrobeBits)
  val user = UInt(width = nastiWUserBits)
}

class NastiWriteResponseChannel(implicit p: Parameters) extends NastiResponseChannel()(p) {
  val id   = UInt(width = nastiWIdBits)
  val user = UInt(width = nastiBUserBits)
}

class NastiReadAddressChannel(implicit p: Parameters) extends NastiAddressChannel()(p) {
  val id   = UInt(width = nastiRIdBits)
  val user = UInt(width = nastiARUserBits)
}

class NastiReadDataChannel(implicit p: Parameters) extends NastiResponseChannel()(p)
    with HasNastiData {
  val id   = UInt(width = nastiRIdBits)
  val user = UInt(width = nastiRUserBits)
}


object NastiConstants {
  def BURST_FIXED = UInt("b00")
  def BURST_INCR  = UInt("b01")
  def BURST_WRAP  = UInt("b10")

  def RESP_OKAY = UInt("b00")
  def RESP_EXOKAY = UInt("b01")
  def RESP_SLVERR = UInt("b10")
  def RESP_DECERR = UInt("b11")

  def CACHE_DEVICE_NOBUF = UInt("b0000")
  def CACHE_DEVICE_BUF   = UInt("b0001")
  def CACHE_NORMAL_NOCACHE_NOBUF = UInt("b0010")
  def CACHE_NORMAL_NOCACHE_BUF   = UInt("b0011")

  def AXPROT(instruction: Bool, nonsecure: Bool, privileged: Bool): UInt =
    Cat(instruction, nonsecure, privileged)

  def AXPROT(instruction: Boolean, nonsecure: Boolean, privileged: Boolean): UInt =
    AXPROT(Bool(instruction), Bool(nonsecure), Bool(privileged))
}


import NastiConstants._

object NastiWriteAddressChannel {
  def apply(id: UInt, addr: UInt, size: UInt,
      len: UInt = UInt(0), burst: UInt = BURST_INCR)
      (implicit p: Parameters) = {
    val aw = Wire(new NastiWriteAddressChannel)
    aw.id := id
    aw.addr := addr
    aw.len := len
    aw.size := size
    aw.burst := burst
    aw.lock := Bool(false)
    aw.cache := CACHE_DEVICE_NOBUF
    aw.prot := AXPROT(false, false, false)
    aw.qos := UInt("b0000")
    aw.region := UInt("b0000")
    aw.user := UInt(0)
    aw
  }
}

object NastiReadAddressChannel {
  def apply(id: UInt, addr: UInt, size: UInt,
      len: UInt = UInt(0), burst: UInt = BURST_INCR)
      (implicit p: Parameters) = {
    val ar = Wire(new NastiReadAddressChannel)
    ar.id := id
    ar.addr := addr
    ar.len := len
    ar.size := size
    ar.burst := burst
    ar.lock := Bool(false)
    ar.cache := CACHE_DEVICE_NOBUF
    ar.prot := AXPROT(false, false, false)
    ar.qos := UInt(0)
    ar.region := UInt(0)
    ar.user := UInt(0)
    ar
  }
}

object NastiWriteDataChannel {
  def apply(data: UInt, strb: Option[UInt] = None,
            last: Bool = Bool(true), id: UInt = UInt(0))
           (implicit p: Parameters): NastiWriteDataChannel = {
    val w = Wire(new NastiWriteDataChannel)
    w.strb := strb.getOrElse(Fill(w.nastiWStrobeBits, UInt(1, 1)))
    w.data := data
    w.last := last
    w.id   := id
    w.user := UInt(0)
    w
  }
}

object NastiReadDataChannel {
  def apply(id: UInt, data: UInt, last: Bool = Bool(true), resp: UInt = UInt(0))(
      implicit p: Parameters) = {
    val r = Wire(new NastiReadDataChannel)
    r.id := id
    r.data := data
    r.last := last
    r.resp := resp
    r.user := UInt(0)
    r
  }
}

object NastiWriteResponseChannel {
  def apply(id: UInt, resp: UInt = UInt(0))(implicit p: Parameters) = {
    val b = Wire(new NastiWriteResponseChannel)
    b.id := id
    b.resp := resp
    b.user := UInt(0)
    b
  }
}
