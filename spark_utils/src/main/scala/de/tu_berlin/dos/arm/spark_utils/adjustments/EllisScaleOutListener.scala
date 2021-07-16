package de.tu_berlin.dos.arm.spark_utils.adjustments

import java.util.Date
import breeze.linalg._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.scheduler._
import scalikejdbc._
import org.slf4j.{Logger, LoggerFactory}

import scala.language.postfixOps

class EllisScaleOutListener(sparkContext: SparkContext, sparkConf: SparkConf) extends SparkListener {
  private val logger: Logger = LoggerFactory.getLogger(classOf[EllisScaleOutListener])
  logger.info("Initializing Ellis listener")
  private val appSignature: String = sparkConf.get("spark.app.name")
  checkConfigurations()
  private val dbPath: String = sparkConf.get("spark.customExtraListener.dbPath")
  private val minExecutors: Int = sparkConf.get("spark.customExtraListener.minExecutors").toInt
  private val maxExecutors: Int = sparkConf.get("spark.customExtraListener.maxExecutors").toInt
  private val targetRuntimeMs: Int = sparkConf.get("spark.customExtraListener.targetRuntimems").toInt
  private val isAdaptive: Boolean = sparkConf.getBoolean("spark.customExtraListener.isAdaptive",defaultValue = true)
  private val method: String = sparkConf.get("spark.customExtraListener.method")

  private val active: Boolean = !isAdaptive || (isAdaptive && method.equals("ellis"))

  private var appEventId: Long = _
  private var appStartTime: Long = _
  private var jobStartTime: Long = _
  private var jobEndTime: Long = _

  private var scaleOut: Int = _
  private var nextScaleOut: Int = _


  Class.forName("org.h2.Driver")
  ConnectionPool.singleton(s"jdbc:h2:$dbPath", "sa", "")

  private val initialExecutors: Int = sparkConf.get("spark.customExtraListener.initialExecutors").toInt
  scaleOut = initialExecutors
  nextScaleOut = scaleOut
  logger.info(s"Using initial scale-out of $scaleOut.")

  def checkConfigurations(){
    /**
     * check parameters are set in environment
     */
    val parametersList = List("spark.customExtraListener.dbPath", "spark.customExtraListener.minExecutors",
      "spark.customExtraListener.initialExecutors", "spark.customExtraListener.maxExecutors",
      "spark.customExtraListener.targetRuntimems", "spark.customExtraListener.method")
    logger.info("Current spark conf" + sparkConf.toDebugString)
    for (param <- parametersList) {
      if (!sparkConf.contains(param)) {
        throw new IllegalArgumentException("parameter " + param + " is not shown in the Environment!")
      }
    }
  }

  override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd): Unit = {

    if(!active){
      return
    }

    DB localTx { implicit session =>
      sql"""
    UPDATE app_event
    SET finished_at = ${new Date(applicationEnd.time)}
    WHERE id = ${appEventId};
    """.update().apply()
    }
  }

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {

    if(!active){
      return
    }

    logger.info(s"Job ${jobStart.jobId} started.")
    jobStartTime = jobStart.time

    // https://stackoverflow.com/questions/29169981/why-is-sparklistenerapplicationstart-never-fired
    // onApplicationStart workaround
    if (appStartTime == 0) {
      appStartTime = jobStartTime

      DB localTx { implicit session =>
        appEventId =
          sql"""
      INSERT INTO app_event (
        app_id,
        started_at
      )
      VALUES (
        ${appSignature},
        ${new Date(appStartTime)}
      );
      """.updateAndReturnGeneratedKey("id").apply()
      }
    }
  }

  override def onJobEnd(jobEnd: SparkListenerJobEnd): Unit = {

    if(!active){
      return
    }

    jobEndTime = jobEnd.time
    val jobDuration = jobEndTime - jobStartTime
    logger.info(s"Job ${jobEnd.jobId} finished in $jobDuration ms with $scaleOut nodes.")

    DB localTx { implicit session =>
      sql"""
    INSERT INTO job_event (
      app_event_id,
      job_id,
      started_at,
      finished_at,
      duration_ms,
      scale_out
    )
    VALUES (
      ${appEventId},
      ${jobEnd.jobId},
      ${new Date(jobStartTime)},
      ${new Date(jobEndTime)},
      ${jobDuration},
      ${scaleOut}
    );
    """.update().apply()
    }

    if (nextScaleOut != scaleOut) {
      scaleOut = nextScaleOut
    }

    if (isAdaptive && method.equals("ellis")) {
      val (scaleOuts, runtimes) = EllisUtils.getNonAdaptiveRuns(appEventId, appSignature)
      if (scaleOuts.length > 3) { // do not scale adaptively for the bootstrap runs
        updateScaleOut(jobEnd.jobId)
      }
    }
  }

  def updateScaleOut(jobId: Int): Unit = {
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

    // calculate the prediction for the remaining runtime depending on scale-out
    val predictedScaleOuts = (minExecutors to maxExecutors).toArray
    val remainingRuntimes: Array[DenseVector[Int]] = jobRuntimeData.keys
      .filter(_ > jobId)
      .toArray
      .sorted
      .map(jobId => {
        val (x, y) = jobRuntimeData(jobId).map(t => (t._2, t._3)).toArray.unzip
        val predictedRuntimes: Array[Int] = EllisUtils.computePredictions(x, y, predictedScaleOuts)
        DenseVector(predictedRuntimes)
      })

    if (remainingRuntimes.length <= 1) {
      return
    }

    val nextJobId = jobRuntimeData.keys.filter(_ > jobId).min

    // predicted runtimes of the next job
    val nextJobRuntimes = remainingRuntimes.head
    // predicted runtimes sum of the jobs *after* the next job
    val futureJobsRuntimes = remainingRuntimes.drop(1).fold(DenseVector.zeros[Int](predictedScaleOuts.length))(_ + _)

    val currentRuntime = jobEndTime - appStartTime
    logger.info(s"Current runtime: $currentRuntime")
    val nextJobRuntime = nextJobRuntimes(scaleOut - minExecutors)
    logger.info(s"Next job runtime prediction: $nextJobRuntime")
    val remainingTargetRuntime = targetRuntimeMs - currentRuntime - nextJobRuntime
    logger.info(s"Remaining runtime: $remainingTargetRuntime")
    val remainingRuntimePrediction = futureJobsRuntimes(scaleOut - minExecutors)
    logger.info(s"Remaining runtime prediction: $remainingRuntimePrediction")

    // check if current scale-out can fulfill the target runtime constraint
    val relativeSlackUp = 1.05
    val absoluteSlackUp = 0

    val relativeSlackDown = .85
    val absoluteSlackDown = 0

    if (remainingRuntimePrediction > remainingTargetRuntime * relativeSlackUp + absoluteSlackUp) {

      val nextScaleOutIndex = futureJobsRuntimes.findAll(_ < remainingTargetRuntime * .9)
        .sorted
        .headOption
        .getOrElse(argmin(futureJobsRuntimes))
      val nextScaleOut = predictedScaleOuts(nextScaleOutIndex)

      if (nextScaleOut != scaleOut) {
        logger.info(s"Adjusting scale-out to $nextScaleOut after job $nextJobId.")
        val requestResult = sparkContext.requestTotalExecutors(scaleOut, 0, Map[String, Int]())
        logger.info("Change scaling result: " + requestResult.toString)
        this.nextScaleOut = nextScaleOut
      }

    } else if (remainingRuntimePrediction < remainingTargetRuntime * relativeSlackDown - absoluteSlackDown) {

      val nextScaleOutIndex = futureJobsRuntimes.findAll(_ < remainingTargetRuntime * .9)
        .sorted
        .headOption
        .getOrElse(argmin(futureJobsRuntimes))
      val nextScaleOut = predictedScaleOuts(nextScaleOutIndex)

      if (nextScaleOut < scaleOut) {
        logger.info(s"Adjusting scale-out to $nextScaleOut after job $nextJobId.")
        val requestResult = sparkContext.requestTotalExecutors(scaleOut, 0, Map[String, Int]())
        logger.info("Change scaling result: " + requestResult.toString)
        this.nextScaleOut = nextScaleOut
      }

    } else {
      logger.debug(s"Scale-out is not changed.")
    }

  }
}
