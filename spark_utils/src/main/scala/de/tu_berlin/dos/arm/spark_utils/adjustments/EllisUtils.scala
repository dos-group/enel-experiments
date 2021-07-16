package de.tu_berlin.dos.arm.spark_utils.adjustments

import breeze.linalg._
import de.tu_berlin.dos.arm.spark_utils.prediction.{Bell, Ernest, UnivariatePredictor}
import org.slf4j.{Logger, LoggerFactory}
import scalikejdbc._

import scala.language.postfixOps


object EllisUtils {

  def computeInitialScaleOut(appEventId: Long, appSignature: String, minExecutors: Int, maxExecutors: Int, targetRuntimeMs: Int): Int = {
    val (scaleOuts, runtimes) = EllisUtils.getNonAdaptiveRuns(appEventId, appSignature)

    val halfExecutors = (minExecutors + maxExecutors) / 2

    scaleOuts.length match {
      case 0 => maxExecutors
      case 1 => halfExecutors
      case 2 =>

        if (runtimes.max < targetRuntimeMs) {
          (minExecutors + halfExecutors) / 2
        } else {
          (halfExecutors + maxExecutors) / 2
        }

      case _ =>

        val predictedScaleOuts = (minExecutors to maxExecutors).toArray
        val predictedRuntimes = EllisUtils.computePredictionsFromStageRuntimes(appEventId, appSignature, predictedScaleOuts)

        val candidateScaleOuts = (predictedScaleOuts zip predictedRuntimes)
          .filter(_._2 < targetRuntimeMs)
          .map(_._1)

        if (candidateScaleOuts.isEmpty) {
          predictedScaleOuts(argmin(predictedRuntimes))
        } else {
          candidateScaleOuts.min
        }
    }
  }

    def getNonAdaptiveRuns(appEventId: Long, appSignature: String): (Array[Int], Array[Int]) = {
      val result = DB readOnly { implicit session =>
        sql"""
      SELECT APP_EVENT.STARTED_AT, SCALE_OUT, DURATION_MS
      FROM APP_EVENT JOIN JOB_EVENT ON APP_EVENT.ID = JOB_EVENT.APP_EVENT_ID
      WHERE APP_ID = ${appSignature}
      AND ID < ${appEventId};
      """.map({ rs =>
          val startedAt = rs.timestamp("started_at")
          val scaleOut = rs.int("scale_out")
          val durationMs = rs.int("duration_ms")
          (startedAt, scaleOut, durationMs)
        }).list().apply()
      }

      val (scaleOuts, runtimes) = result
        .groupBy(_._1)
        .toArray
        .flatMap(t => {
          val jobStages = t._2
          val scaleOuts = jobStages.map(_._2)
          val scaleOut = scaleOuts.head
          val nonAdaptive = scaleOuts.forall(scaleOut == _)
          if (nonAdaptive) {
            val runtime = jobStages.map(_._3).sum
            List((scaleOut, runtime))
          } else {
            List()
          }
        })
        .unzip

      (scaleOuts, runtimes)
    }

  def computePredictionsFromStageRuntimes(appEventId: Long, appSignature: String, predictedScaleOuts: Array[Int]): Array[Int] = {
    val result = DB readOnly { implicit session =>
      sql"""
      SELECT JOB_ID, SCALE_OUT, DURATION_MS
      FROM APP_EVENT JOIN JOB_EVENT ON APP_EVENT.ID = JOB_EVENT.APP_EVENT_ID
      WHERE APP_ID = ${appSignature}
      AND ID < ${appEventId}
      ORDER BY JOB_ID;
      """.map({ rs =>
        val jobId = rs.int("job_id")
        val scaleOut = rs.int("scale_out")
        val durationMs = rs.int("duration_ms")
        (jobId, scaleOut, durationMs)
      }).list().apply()
    }
    val jobRuntimeData: Map[Int, List[(Int, Int, Int)]] = result.groupBy(_._1)

    val predictedRuntimes: Array[DenseVector[Int]] = jobRuntimeData.keys
      .toArray
      .sorted
      .map(jobId => {
        val (x, y) = jobRuntimeData(jobId).map(t => (t._2, t._3)).toArray.unzip
        val predictedRuntimes: Array[Int] = EllisUtils.computePredictions(x, y, predictedScaleOuts)
        DenseVector(predictedRuntimes)
      })

    predictedRuntimes.fold(DenseVector.zeros[Int](predictedScaleOuts.length))(_ + _).toArray
  }

  def computePredictions(scaleOuts: Array[Int], runtimes: Array[Int], predictedScaleOuts: Array[Int]): Array[Int] = {
    val x = convert(DenseVector(scaleOuts), Double)
    val y = convert(DenseVector(runtimes), Double)

    // calculate the range over which the runtimes must be predicted
    val xPredict = DenseVector(predictedScaleOuts)

    // subdivide the scaleout range into interpolation and extrapolation
    //    val interpolationMask: BitVector = (xPredict :>= min(scaleOuts)) :& (xPredict :<= max(scaleOuts))
    val interpolationMask: BitVector = (xPredict >:= min(scaleOuts)) &:& (xPredict <:= max(scaleOuts))
    val xPredictInterpolation = xPredict(interpolationMask).toDenseVector
    val xPredictExtrapolation = xPredict(!interpolationMask).toDenseVector

    // predict with respective model
    val yPredict = DenseVector.zeros[Double](xPredict.length)

    // fit ernest
    val ernest: UnivariatePredictor = new Ernest()
    ernest.fit(x, y)

    val uniqueScaleOuts = unique(x).length
    if (uniqueScaleOuts <= 2) {
      // for very few data, just take the mean
      yPredict := sum(y) / y.length
    } else if (uniqueScaleOuts <= 5) {
      // if too few data use ernest model
      yPredict := ernest.predict(convert(xPredict, Double))
    } else {
      // fit data using bell (for interpolation)
      val bell: UnivariatePredictor = new Bell()
      bell.fit(x, y)
      yPredict(interpolationMask) := bell.predict(convert(xPredictInterpolation, Double))
      yPredict(!interpolationMask) := ernest.predict(convert(xPredictExtrapolation, Double))
    }

    yPredict.map(_.toInt).toArray
  }
}

class EllisApplication() {

  private val logger: Logger = LoggerFactory.getLogger(classOf[EllisApplication])
  private var dbPathApp: String = _

  def init(dbPath: String): Unit ={
    dbPathApp = dbPath
    Class.forName("org.h2.Driver")
    ConnectionPool.singleton(s"jdbc:h2:$dbPath", "sa", "")
  }

  def createTables(): Unit = {
    DB localTx { implicit session =>
      sql"""
      CREATE TABLE IF NOT EXISTS app_event (
      id INT AUTO_INCREMENT PRIMARY KEY NOT NULL,
      app_id VARCHAR(64) NOT NULL,
      started_at TIMESTAMP NOT NULL,
      finished_at TIMESTAMP,
      UNIQUE (app_id, started_at));
      """.execute.apply()

      sql"""
      CREATE TABLE IF NOT EXISTS job_event (
      app_event_id INT NOT NULL,
      job_id INT NOT NULL,
      started_at TIMESTAMP NOT NULL,
      finished_at TIMESTAMP NOT NULL,
      duration_ms INT NOT NULL,
      scale_out INT NOT NULL,
      PRIMARY KEY (app_event_id, job_id),
      FOREIGN KEY (app_event_id) REFERENCES app_event (id));
      """.execute.apply()
    }
  }

  def computeInitialScaleOut(dbPath: String, appEventId: Long,
                             appSignature: String, minExecutors: Int, maxExecutors: Int, targetRuntimeMs: Int): Int = {
    logger.info(s"New request: ${dbPath}, ${appEventId}, ${appSignature}, ${minExecutors}, ${maxExecutors}, ${targetRuntimeMs}")
    if(dbPathApp != dbPath){
      init(dbPath)
    }
    createTables()

    logger.info(s"Consider all database entries with App-Event-ID < ${appEventId}...")
    val scaleOut: Int = EllisUtils.computeInitialScaleOut(appEventId, appSignature, minExecutors, maxExecutors, targetRuntimeMs)
    logger.info(s"[App-Event-ID: ${appEventId}] Initial scale-out recommendation: ${scaleOut}")

    scaleOut
  }
}
