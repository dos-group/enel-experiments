package de.tu_berlin.dos.arm.spark_utils.datagens

import breeze.linalg.DenseVector
import breeze.stats.distributions.Rand
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
case class MeanConf(mean: DenseVector[Double], stdDev: Double)

object KMeansDataGenerator {

  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      println("KMeansDataGenerator <samples> <cluster> <output> <defaultHdfsFs>")
      System.exit(1)
    }

    val n = args(0).toInt
    val k = args(1).toInt
    val outputPath = args(2)
    val defaultFs = args(3)

    val dim = 2
    val stdDev = .012

    Rand.generator.setSeed(0)

    val centers = uniformRandomCenters(dim, k, stdDev)
    val centerDistribution = Rand.choose(centers)

    System.setProperty("HADOOP_USER_NAME", "drms")
    val path = new Path(outputPath)
    val conf = new Configuration()
    conf.set("fs.defaultFS", defaultFs)
    conf.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName);
    conf.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName);

    val fs = FileSystem.get(conf)
    val os = fs.create(path)
    (1 to n).foreach(_ => {
      val MeanConf(mean, stdDev) = centerDistribution.draw()
      val p = mean + DenseVector.rand[Double](mean.length, Rand.gaussian(0, stdDev))
      os.writeBytes(p.toArray.mkString(" ") + "\n")
    })
    fs.close()

  }

  def uniformRandomCenters(dim: Int, k: Int, stdDev: Double): Seq[MeanConf] = {
    (1 to k).map(_ => {
      val mean = DenseVector.rand[Double](dim)
      MeanConf(mean, stdDev)
    })
  }
}
