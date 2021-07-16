package de.tu_berlin.dos.arm.spark_utils.adjustments

import org.slf4j.LoggerFactory
import py4j.GatewayServer


object EllisApplicationRunner {
  def main(args: Array[String]): Unit = {

    val app: EllisApplication = new EllisApplication()
    val server: GatewayServer = new GatewayServer(app)
    LoggerFactory.getLogger(classOf[EllisApplication]).info("Gateway Server started...")
    server.start()
  }
}
