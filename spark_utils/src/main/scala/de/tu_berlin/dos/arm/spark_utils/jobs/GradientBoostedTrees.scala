package de.tu_berlin.dos.arm.spark_utils.jobs

import Utils.{isEllisEnabled, isEnelEnabled}
import de.tu_berlin.dos.arm.spark_utils.adjustments.{EllisScaleOutListener, EnelScaleOutListener}
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.rogach.scallop.exceptions.ScallopException
import org.rogach.scallop.{ScallopConf, ScallopOption}

object GradientBoostedTrees {
  def main(args: Array[String]): Unit = {

    val conf = new GradientBoostedTreesArgs(args)
    val appSignature = "GradientBoostedTrees"

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
    sparkContext.textFile(conf.input())

    // Load and parse the data file.
    val data = MLUtils.loadLabeledPoints(sparkContext, conf.input())
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a GradientBoostedTrees model.
    // The defaultParams for Classification use LogLoss by default.
    val boostingStrategy = BoostingStrategy
      .defaultParams("Regression")
    boostingStrategy.setNumIterations(conf.iterations()) // Note: Use more iterations in practice.
    //    boostingStrategy.treeStrategy.setNumClasses(2)
    //    boostingStrategy.treeStrategy.setMaxDepth(5)
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    //    boostingStrategy.treeStrategy.setCategoricalFeaturesInfo(Map[Int, Int]())

    val model = org.apache.spark.mllib.tree.GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification GBT model:\n" + model.toDebugString)

    while(listener != null && listener.hasOpenFutures){
      Thread.sleep(5000)
    }
    sparkContext.stop()
  }
}

class GradientBoostedTreesArgs(a: Seq[String]) extends ScallopConf(a) {
  val input: ScallopOption[String] = trailArg[String](required = true, name = "<input>",
    descr = "Input file").map(_.toLowerCase)

  val iterations: ScallopOption[Int] = opt[Int](noshort = true, default = Option(100),
    descr = "Amount of iterations")

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

