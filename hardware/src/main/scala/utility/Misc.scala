package utility

import Chisel._
import scala.math._
import chipsalliance.rocketchip.config._
import chipsalliance.rocketchip.config._

class HeteaccParameterizedBundle(implicit p: Parameters) extends Bundle

abstract class HeteaccGenericParameterizedBundle[+T <: Object](val params: T) extends Bundle
{
  override def cloneType = {
    try {
      this.getClass.getConstructors.head.newInstance(params).asInstanceOf[this.type]
    } catch {
      case e: java.lang.IllegalArgumentException =>
        throw new Exception("Unable to use GenericParameterizedBundle.cloneType on " +
                       this.getClass + ", probably because " + this.getClass +
                       "() takes more than one argument.  Consider overriding " +
                       "cloneType() on " + this.getClass, e)
    }
  }
}