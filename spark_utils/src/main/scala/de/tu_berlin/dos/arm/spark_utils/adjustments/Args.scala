package de.tu_berlin.dos.arm.spark_utils.adjustments

import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.rogach.scallop.exceptions.ScallopException

class Args(a: Seq[String]) extends ScallopConf(a) {
  override def onError(e: Throwable): Unit = e match {
    case ScallopException(message) =>
      println(message)
//      println(s"Usage: allocation-assistant -c <config> -r <max runtime> -m <memory> -s <slots> " +
//        s"-i <fallback containers> -N <max containers> " +
//        s"[more args ...] <Jar> [Jar args ...]")
      println()
      printHelp()
      System.exit(1)
    case other => super.onError(e)
  }

  val dbPath: ScallopOption[String] = opt[String](name = "db", noshort = true, default = Some("./target/bell"),
    descr = "Path to the H2 database")
//  val config: ScallopOption[String] = opt[String](required = true, descr = "Path to the .conf file")

  val maxRuntime: ScallopOption[Double] = opt[Double](required = true, short = 'r',
    descr = "Maximum runtime in seconds")

  val minContainers: ScallopOption[Int] = opt[Int](short = 'n', default = Option(1),
    descr = "Minimum number of containers to assign")
  val maxContainers: ScallopOption[Int] = opt[Int](required = true, short = 'N',
    descr = "Maximum number of containers to assign")

  val adaptive: ScallopOption[Boolean] = opt[Boolean](default = Option(false), noshort = true,
    descr = "Enables runtime adjustments by scaling between jobs")

  verify()
}
