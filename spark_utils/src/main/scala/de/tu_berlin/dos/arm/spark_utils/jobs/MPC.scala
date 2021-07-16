package de.tu_berlin.dos.arm.spark_utils.jobs

import Utils.{isEllisEnabled, isEnelEnabled}
import de.tu_berlin.dos.arm.spark_utils.adjustments.{EllisScaleOutListener, EnelScaleOutListener}
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint => NewLabeledPoint}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession
import org.rogach.scallop.exceptions.ScallopException
import org.rogach.scallop.{ScallopConf, ScallopOption}

object MPC {
  def main(args: Array[String]): Unit = {


    val conf = new MPCArgs(args)

    val appSignature = "MPC"

    val sparkConf = new SparkConf()
      .setAppName(appSignature)

    val spark = SparkSession
      .builder
      .appName(appSignature)
      .config(sparkConf)
      .getOrCreate()
    import spark.implicits._

    var listener: EnelScaleOutListener = null
    if (isEnelEnabled(sparkConf)){
      listener = new EnelScaleOutListener(spark.sparkContext, sparkConf)
      spark.sparkContext.addSparkListener(listener)
    }
    if (isEllisEnabled(sparkConf)) {
      spark.sparkContext.addSparkListener(new EllisScaleOutListener(spark.sparkContext, sparkConf))
    }

    val data = spark.sparkContext.textFile(conf.input(), spark.sparkContext.defaultMinPartitions).map(s => {
      val parts = s.split(',')
      val (labelStr, featuresArr) = parts.splitAt(1)
      val label = java.lang.Double.parseDouble(labelStr(0))
      val features = Vectors.dense(featuresArr.map(java.lang.Double.parseDouble))
      LabeledPoint(label, features)
    })
      .map(lp => {
        NewLabeledPoint(lp.label, lp.features.asML)
      })
      .toDF()


    // Split the data into train and test
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val input_dimension = conf.input_dimension()
    val output_dimension = conf.output_size()
    val layers = Array[Int](input_dimension, 100, 50, output_dimension)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setSolver("gd")
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setTol(Double.MinPositiveValue)
      .setMaxIter(conf.iterations())

    // train the model
    val model = trainer.fit(train)

    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))

    while(listener != null && listener.hasOpenFutures){
      Thread.sleep(5000)
    }
    spark.stop()
  }
}

class MPCArgs(a: Seq[String]) extends ScallopConf(a) {
  //  val config = opt[String](required = true,
  //    descr = "Path to the .conf file")
  //
  val input: ScallopOption[String] = trailArg[String](required = true, name = "<input>",
    descr = "Input file").map(_.toLowerCase)
  val iterations: ScallopOption[Int] = opt[Int](noshort = true, default = Option(100),
    descr = "Amount of SGD iterations")
  val input_dimension: ScallopOption[Int] = opt[Int](noshort = true, default = Option(200),
    descr = "Dimension of input")
  val output_size: ScallopOption[Int] = opt[Int](noshort = true, default = Option(3),
    descr = "Size of output")

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
