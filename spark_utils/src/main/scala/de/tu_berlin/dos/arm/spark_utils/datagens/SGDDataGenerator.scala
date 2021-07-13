package de.tu_berlin.dos.arm.spark_utils.datagens

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

import java.util.concurrent.ThreadLocalRandom
import scala.math.pow

object SGDDataGenerator {
  def main(args: Array[String]): Unit = {
    if (args.length != 4) {
      Console.err.println("Usage: SGDDataGenerator <samples> <dimension> <output> <defaultHdfsFs>")
      System.exit(-1)
    }

    val m = args(0).toInt
    val n = args(1).toInt
    val outputPath = args(2)
    val defaultFs = args(3)


    System.setProperty("HADOOP_USER_NAME", "drms")
    val path = new Path(outputPath)
    val conf = new Configuration()
    conf.set("fs.defaultFS", defaultFs)
    conf.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName);
    conf.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName);
    val fs = FileSystem.get(conf)
    val os = fs.create(path)

    for (_ <- 0 until m) {
      val x = ThreadLocalRandom.current().nextDouble() * 10
      val noise = ThreadLocalRandom.current().nextGaussian() * 3

      // generate the function value with added gaussian noise
      val label = function(x) + noise

      // generate a vandermatrix from x
      val vector = polyvander(x, n - 1)

      // write to file in MLUtils format (label, feature1 feature2 ... featureN)
      os.writeBytes(label + "," + vector.mkString(" ") + "\n")
    }

    fs.close()
  }

  def polyvander(x: Double, order: Int): Array[Double] = {
    (0 to order).map(pow(x, _)).toArray
  }

  def function(x: Double): Double = {
    2 * x + 10
  }
}
