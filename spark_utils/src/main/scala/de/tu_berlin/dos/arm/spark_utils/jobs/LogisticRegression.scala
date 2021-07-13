package de.tu_berlin.dos.arm.spark_utils.jobs

import Utils.{isEllisEnabled, isEnelEnabled}
import de.tu_berlin.dos.arm.spark_utils.adjustments.{EllisScaleOutListener, EnelScaleOutListener}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.rogach.scallop.exceptions.ScallopException
import org.rogach.scallop.{ScallopConf, ScallopOption}


object LogisticRegression {
  def main(args: Array[String]): Unit = {

    val conf = new LRArgs(args)

    val appSignature = "LogisticRegression"

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

    var data = sparkContext.textFile(conf.input(), sparkContext.defaultMinPartitions).map(s => {
      val parts = s.split(',')
      val (labelStr, featuresArr) = parts.splitAt(1)
      val label = java.lang.Double.parseDouble(labelStr(0))
      val features = Vectors.dense(featuresArr.map(java.lang.Double.parseDouble))
      LabeledPoint(label, features)
    })

    if (conf.cache()) {
      data = data.cache()
    }
    // Split data into training (60%) and test (40%).
    val Array(training, test) = data.randomSplit(Array(0.8, 0.2))

    // Run training algorithm to build the model
    val lr = new LogisticRegressionWithLBFGS()
      .setNumClasses(3)
    lr.optimizer
      .setNumIterations(conf.iterations())
      .setConvergenceTol(Double.MinPositiveValue)
    val model = lr.run(training)

    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy
    println(s"Accuracy = $accuracy")

    while(listener != null && listener.hasOpenFutures){
      Thread.sleep(5000)
    }
    sparkContext.stop()
  }
}

class LRArgs(a: Seq[String]) extends ScallopConf(a) {
  val input: ScallopOption[String] = trailArg[String](required = true, name = "<input>",
    descr = "Input file").map(_.toLowerCase)
  val iterations: ScallopOption[Int] = opt[Int](noshort = true, default = Option(100),
    descr = "Amount of Logistic Regression iterations")
  val cache: ScallopOption[Boolean] = opt[Boolean](noshort = true, default = Option(false),
    descr = "Caches the input data")

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

  verify()
}
