/*
 * KMeans workload for BigDataBench
 */
package de.tu_berlin.dos.arm.spark_utils.jobs

import Utils.{isEllisEnabled, isEnelEnabled}
import de.tu_berlin.dos.arm.spark_utils.adjustments.{EllisScaleOutListener, EnelScaleOutListener}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.rogach.scallop.exceptions.ScallopException
import org.rogach.scallop.{ScallopConf, ScallopOption}

object KMeans {
  val MLLibKMeans = org.apache.spark.mllib.clustering.KMeans

  def main(args: Array[String]): Unit = {

    val conf = new KMeansArgs(args)
    val appSignature = "KMeans"

    val splits = 2
    val sparkConf = new SparkConf()
      .setAppName(appSignature)

    val sparkContext = new SparkContext(sparkConf)

    var listener: EnelScaleOutListener = null
    if (isEnelEnabled(sparkConf)){
      listener = new EnelScaleOutListener(sparkContext, sparkConf)
      sparkContext.addSparkListener(listener)
    }
    if (isEllisEnabled(sparkConf)) {
      sparkContext.addSparkListener(new EllisScaleOutListener(sparkContext, sparkConf))
    }

    println("Start KMeans training...")
    // Load and parse the data
    val data = sparkContext.textFile(conf.input(), splits)
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))

    val clusters = new org.apache.spark.mllib.clustering.KMeans()
      .setEpsilon(0)
      .setK(conf.k())
      .setMaxIterations(conf.iterations())
      .run(parsedData)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    clusters.clusterCenters.foreach(v => {
      println(v)
    })

    while(listener != null  && listener.hasOpenFutures){
      Thread.sleep(5000)
    }
    sparkContext.stop()
  }
}

class KMeansArgs(a: Seq[String]) extends ScallopConf(a) {
  val input: ScallopOption[String] = trailArg[String](required = true, name = "<input>",
    descr = "Input file").map(_.toLowerCase)

  val k: ScallopOption[Int] = opt[Int](required = true, descr = "Amount of clusters")
  val iterations: ScallopOption[Int] = opt[Int](noshort = true, default = Option(100),
    descr = "Amount of KMeans iterations")

  override def onError(e: Throwable): Unit = e match {
    case ScallopException(message) =>
      println(message)
      println()
      printHelp()
      System.exit(1)
    case other => super.onError(e)
  }

  verify()
}

