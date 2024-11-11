package heteacc.config

import chisel3._
import chisel3.util._
import chipsalliance.rocketchip.config._
import heteacc.fpu.FType
import utility.{HeteaccParameterizedBundle}
import heteacc.interfaces.axi.AXIParams
import heteacc.junctions.{NastiKey, NastiParameters}
import utility._
trait AccelParams {

  var xlen: Int
  var ylen: Int
  val tlen: Int
  val glen: Int
  val typeSize: Int
  val beats: Int
  val mshrLen: Int
  val fType: FType

  //Cache
  val nways: Int
  val nsets: Int

  def cacheBlockBytes: Int

  // Debugging dumps
  val log: Boolean
  val clog: Boolean
  val verb: String
  val comp: String
}

/**
  * VCR parameters.
  * These parameters are used on VCR interfaces and modules.
  */
trait DCRParams {
  val nCtrl: Int
  val nECnt: Int
  val nVals: Int
  val nPtrs: Int
  val regBits: Int
}

/**
  * DME parameters.
  * These parameters are used on DME interfaces and modules.
  */
trait DMEParams {
  val nReadClients: Int
  val nWriteClients: Int
}

//It defines non-variable data structures that are used for configuration parameters.
case class HeteaccAccelParams(
                                 dataLen: Int = 32,
                                 addrLen: Int = 32,
                                 taskLen: Int = 5,
                                 groupLen: Int = 16,
                                 mshrLen: Int = 8,
                                 tSize: Int = 64,
                                 verbosity: String = "low",
                                 components: String = "",
                                 printLog: Boolean = true,
                                 printMemLog: Boolean = false,
                                 printCLog: Boolean = false,
                                 cacheNWays: Int = 1,
                                 cacheNSets: Int = 256
                               ) extends AccelParams {

  var xlen: Int = dataLen
  var ylen: Int = addrLen
  val tlen: Int = taskLen
  val glen: Int = groupLen
  val typeSize: Int = tSize
  val beats: Int = 0
  val mshrlen: Int = mshrLen
  val fType = dataLen match {
    case 64 => FType.D
    case 32 => FType.S
    case 16 => FType.H
  }

  //Cache
  val nways = cacheNWays // TODO: set-associative
  val nsets = cacheNSets

  def cacheBlockBytes: Int = 8 * (xlen >> 3) // 4 x 64 bits = 32B

  // Debugging dumps
  val log: Boolean = printLog
  val memLog: Boolean = printMemLog
  val clog: Boolean = printCLog
  val verb: String = verbosity
  val comp: String = components

}

/**
 * DCR parameters.
 * These parameters are used on DCR interfaces and modules.
 */
case class HeteaccDCRParams(numCtrl: Int = 1,
                              numEvent: Int = 1,
                              numVals: Int = 2,
                              numPtrs: Int = 4,
                              numRets: Int = 0) {
  val nCtrl = numCtrl
  val nECnt = numEvent + numRets
  val nVals = numVals
  val nPtrs = numPtrs
  val regBits = 32
}


/**
  * DME parameters.
  * These parameters are used on DME interfaces and modules.
  */
case class HeteaccDMEParams(numRead: Int = 1,
                              numWrite: Int = 1) extends DMEParams {
  val nReadClients: Int = numRead
  val nWriteClients: Int = numWrite
  require(nReadClients > 0,
    s"\n\n[Heteacc] [DMEParams] nReadClients must be larger than 0\n\n")
  require(
    nWriteClients > 0,
    s"\n\n[Heteacc] [DMEParams] nWriteClients must be larger than 0\n\n")
}


/**
 * Debug Parameters
 * These parameters are used on Debug nodes.
 */
case class DebugParams(len_data: Int = 64,
                       len_id: Int = 8,
                       len_code: Int = 5,
                       iteration_len: Int = 10,
                       len_guard: Int = 2) {
  val gLen = len_guard
  val idLen = len_id
  val codeLen = len_code
  val iterLen = iteration_len
  val dataLen = len_data - (gLen + idLen + codeLen + + iterLen)
  val packetLen = len_data
}


/*We define a configuration key
HeteaccConfigKey
to store the Accel parameter configurations of type HeteaccAccelParams.*/
case object HeteaccConfigKey extends Field[HeteaccAccelParams]

case object DCRKey extends Field[HeteaccDCRParams]

case object DMEKey extends Field[HeteaccDMEParams]

case object HostParamKey extends Field[AXIParams]

case object MemParamKey extends Field[AXIParams]

case object DebugParamKey extends Field[DebugParams]


// This class extends the Config class. 
// It is used to create instances of specific configurations
// containing accel and debugging parameters.
class WithAccelConfig(inParams: HeteaccAccelParams = HeteaccAccelParams(), inDebugs: DebugParams =  DebugParams())//
  extends Config((site, here, up) => {
    // Core
    case HeteaccConfigKey => inParams
    // case DebugParamKey =>  inDebugs

       case NastiKey => new NastiParameters(
         idBits = 12,
         dataBits = 32,
         addrBits = 32)

       case MemParamKey=> AXIParams(
        addrBits = 32, dataBits = 32, userBits = 5,
                        lenBits = 16, // limit to 16 beats, instead of 256 beats in AXI4
                        coherent = true
       )
        case DebugParamKey =>  inDebugs
  }
  )

/**
  * Please note that the dLen from WithSimShellConfig should be the same value as
  * AXI -- memParams:dataBits
  *
  * @param vcrParams
  * @param dmeParams
  * @param hostParams
  * @param memParams
  */
class WithTestConfig(vcrParams: HeteaccDCRParams = HeteaccDCRParams(),
                      dmeParams: HeteaccDMEParams = HeteaccDMEParams(),
                      hostParams: AXIParams = AXIParams(
                        addrBits = 16, dataBits = 32, idBits = 13, lenBits = 4),
                      memParams: AXIParams = AXIParams(
                        addrBits = 32, dataBits = 32, userBits = 5,
                        lenBits = 4, // limit to 16 beats, instead of 256 beats in AXI4
                        coherent = true),
                      nastiParams: NastiParameters = NastiParameters(dataBits = 32, addrBits = 32, idBits = 13))
  extends Config((site, here, up) => {
    // Core
    case DCRKey => vcrParams
    case DMEKey => dmeParams
    case HostParamKey => hostParams
    case MemParamKey => memParams
    case NastiKey => nastiParams
  }
  )
// It is used to provide access to the Accel parameters.
trait HasAccelParams {
  implicit val p: Parameters

  def accelParams: HeteaccAccelParams = p(HeteaccConfigKey)

  val xlen = accelParams.xlen
  val ylen = accelParams.ylen
  val tlen = accelParams.tlen
  val glen = accelParams.glen
  val mshrLen = accelParams.mshrLen
  val typeSize = accelParams.typeSize
  val beats = typeSize / xlen
  val fType = accelParams.fType
  val log = accelParams.log
  val memLog = accelParams.memLog
  val clog = accelParams.clog
  val verb = accelParams.verb
  val comp = accelParams.comp

}

trait HasAccelShellParams {
  implicit val p: Parameters

  def dcrParams: HeteaccDCRParams = p(DCRKey)

  def dmeParams: HeteaccDMEParams = p(DMEKey)

  def hostParams: AXIParams = p(HostParamKey)

  def memParams: AXIParams = p(MemParamKey)

  def nastiParams: NastiParameters = p(NastiKey)
  
  def dbgParams: DebugParams = p(DebugParamKey)
}

trait HasDebugCodes {
  implicit val p: Parameters

  def debugParams: DebugParams = p(DebugParamKey)

  val DbgLoadAddress = "b00001".U(debugParams.codeLen.W)
  val DbgLoadData = "b00010".U(debugParams.codeLen.W)
  val DbgStoreAddress = "b0011".U(debugParams.codeLen.W)
  val DbgStoreData = "b0100".U(debugParams.codeLen.W)
  val DbgComputeData = "b0101".U(debugParams.codeLen.W)
  val DbgPhiData = "b0110".U(debugParams.codeLen.W)
}

abstract class DebugBase(params: DebugParams) extends HeteaccGenericParameterizedBundle(params)

class DebugPacket(params: DebugParams) extends DebugBase(params) {
  val gFlag = UInt(params.gLen.W)
  val bID = UInt(params.idLen.W)
  val debugCode = UInt(params.codeLen.W)
  val iterationID = UInt(params.iterLen.W)
  val data = UInt(params.dataLen.W)

  def packet(): UInt = {
    val packet = WireInit(0.U(params.packetLen.W))
    packet := Cat(gFlag, bID, debugCode, iterationID, data)
    packet
  }

}

object DebugPacket{
  def apply(gflag: UInt, id :UInt, code :UInt, iteration: UInt, data: UInt)
           (params : DebugParams): DebugPacket =
  {
    val packet = Wire(new DebugPacket(params))
    packet.gFlag := gflag
    packet.bID := id
    packet.debugCode := code
    packet.iterationID := iteration
    packet.data := data
    packet
  }
}


abstract class AccelBundle(implicit val p: Parameters) extends HeteaccParameterizedBundle()(p)
  with HasAccelParams