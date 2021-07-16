package de.tu_berlin.dos.arm.spark_utils.adjustments

import org.apache.spark.executor.TaskMetrics
import org.apache.spark.scheduler._
import org.apache.spark.storage.RDDInfo
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s.native.{Json, Serialization}
import org.json4s.{DefaultFormats, FieldSerializer, Formats}
import org.slf4j.{Logger, LoggerFactory}
import sttp.client3._
import sttp.client3.asynchttpclient.future.AsyncHttpClientFutureBackend
import sttp.client3.json4s._

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}
import scala.collection.mutable.ListBuffer
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.concurrent.duration.{Duration, SECONDS}
import scala.util.Try


case class UpdateRequestPayload(application_execution_id: String,
                                application_id: Option[String],
                                job_id: Option[Int],
                                update_event: String,
                                updates: String)

case class PredictionResponsePayload(best_predicted_scale_out_per_job: List[List[Int]],
                                     best_predicted_runtime_per_job: List[List[Float]])

case class PredictionRequestPayload(application_execution_id: String,
                                    application_id: String,
                                    job_id: Int,
                                    update_event: String,
                                    updates: String,
                                    predict: Boolean)


class EnelScaleOutListener(sparkContext: SparkContext, sparkConf: SparkConf) extends SparkListener {

  private val logger: Logger = LoggerFactory.getLogger(classOf[EnelScaleOutListener])
  logger.info("Initializing Enel listener")
  private val applicationId: String = sparkConf.getAppId
  private val applicationSignature: String = sparkConf.get("spark.app.name")
  checkConfigurations()
  private val restTimeout: Int = sparkConf.get("spark.customExtraListener.restTimeout").toInt
  private val service: String = sparkConf.get("spark.customExtraListener.service")
  private val port: Int = sparkConf.get("spark.customExtraListener.port").toInt
  private val onlineScaleOutPredictionEndpoint: String = sparkConf.get("spark.customExtraListener.onlineScaleOutPredictionEndpoint")
  private val updateInformationEndpoint: String = sparkConf.get("spark.customExtraListener.updateInformationEndpoint")
  private val applicationExecutionId: String = sparkConf.get("spark.customExtraListener.applicationExecutionId")
  private val isAdaptive: Boolean = sparkConf.getBoolean("spark.customExtraListener.isAdaptive", defaultValue = true)
  private val method: String = sparkConf.get("spark.customExtraListener.method")

  private val active: Boolean = !isAdaptive || (isAdaptive && method.equals("enel"))

  private val infoMap: ConcurrentHashMap[String, scala.collection.mutable.Map[String, Any]] =
    new ConcurrentHashMap[String, scala.collection.mutable.Map[String, Any]]()

  // scale-out, time of measurement, total time
  private val scaleOutBuffer: ListBuffer[(Int, Long)] = ListBuffer()

  private val initialExecutors: Int = sparkConf.get("spark.customExtraListener.initialExecutors").toInt
  logger.info(s"Using initial scale-out of $initialExecutors.")

  // for json4s
  implicit val serialization: Serialization.type = org.json4s.native.Serialization
  implicit val CustomFormats: Formats = DefaultFormats +
    FieldSerializer[scala.collection.mutable.Map[String, Any]]() +
    FieldSerializer[Map[String, Any]]()
  private val futureBuffer: ListBuffer[Future[Any]] = ListBuffer[Future[Any]]()

  // keep track of concurrent prediction requests
  private val concurrentPredictionRequest = new AtomicInteger(0)
  // only set predicted scale-out of next job if not already in jobEnd phase of predecessor job
  private val allowUpdate = new AtomicBoolean(false)
  // keep track of current job-id
  private val currentJobId = new AtomicInteger(0)
  // keep track of actual scale-out (measured)
  private val currentScaleOut = new AtomicInteger(0)
  // keep track of last response length (in order to determine most recent information)
  private val lastResponseLength = new AtomicInteger(-1)
  // cache predictions and update them from time to time
  private val predictedScaleOutMap: ConcurrentHashMap[Int, Int] = new ConcurrentHashMap[Int, Int]()
  predictedScaleOutMap.put(0, initialExecutors)

  def getInitialScaleOutCount(executorHost: String): Int = {
    val allExecutors = sparkContext.getExecutorMemoryStatus.toSeq.map(_._1)
    val driverHost: String = sparkContext.getConf.get("spark.driver.host")
    allExecutors
      .filter(! _.split(":")(0).equals(driverHost))
      .filter(! _.split(":")(0).equals(executorHost.split(":")(0)))
      .toList
      .length
  }

  def getExecutorCount: Int = {
    currentScaleOut.get()
  }

  def saveDivision(a: Int, b: Int): Double = {
    saveDivision(a.toDouble, b.toDouble)
  }

  def saveDivision(a: Long, b: Long): Double = {
    saveDivision(a.toDouble, b.toDouble)
  }

  def saveDivision(a: Double, b: Double): Double = {
    if(a == 0.0){
      0.0
    }
    else if(b == 0.0){
      1.0
    }
    else{
      Try(a / b).getOrElse(0.0)
    }
  }

  def checkConfigurations(){
    /**
     * check parameters are set in environment
     */
    val parametersList = List("spark.customExtraListener.restTimeout", "spark.customExtraListener.service",
      "spark.customExtraListener.initialExecutors", "spark.customExtraListener.port", "spark.customExtraListener.onlineScaleOutPredictionEndpoint",
      "spark.customExtraListener.updateInformationEndpoint", "spark.customExtraListener.applicationExecutionId", "spark.customExtraListener.method")
    logger.info("Current spark conf" + sparkConf.toDebugString)
    for (param <- parametersList) {
      if (!sparkConf.contains(param)) {
        throw new IllegalArgumentException("parameter " + param + " is not shown in the Environment!")
      }
    }
  }

  def updateInformation(applicationId: Option[String], updateMap: Map[String, Any], updateEvent: String): Unit = {

    val backend = HttpURLConnectionBackend()

    try {
      val payload: UpdateRequestPayload = UpdateRequestPayload(
        applicationExecutionId,
        applicationId,
        None,
        updateEvent,
        Json(CustomFormats).write(updateMap))

      basicRequest
        .post(uri"http://$service:$port/$updateInformationEndpoint")
        .body(payload)
        .response(ignore)
        .send(backend)
    } finally backend.close()
  }

  def computeRescalingTimeRatio(startTime: Long, endTime: Long): Double = {

    val scaleOutList: List[(Int, Long)] = scaleOutBuffer.toList

    val dividend: Long = scaleOutList
      .sortBy(_._2)
      .zipWithIndex.map { case (tup, idx) => (tup._1, tup._2, Try(scaleOutList(idx + 1)._2 - tup._2).getOrElse(0L))}
      .filter(e => e._2 + e._3 >= startTime && e._2 <= endTime)
      .drop(1) // drop first element => is respective start scale-out
      .dropRight(1) // drop last element => is respective end scale-out
      .map(e => {
        val startTimeScaleOut: Long = e._2
        var endTimeScaleOut: Long = e._2 + e._3
        if (e._3 == 0L)
          endTimeScaleOut = endTime

        val intervalStartTime: Long = Math.min(Math.max(startTime, startTimeScaleOut), endTime)
        val intervalEndTime: Long = Math.max(startTime, Math.min(endTime, endTimeScaleOut))

        intervalEndTime - intervalStartTime
      })
      .sum

    saveDivision(dividend, endTime - startTime)
  }

  def extractFromRDD(seq: Seq[RDDInfo]): (Int, Int, Long, Long) = {
    var numPartitions: Int = 0
    var numCachedPartitions: Int = 0
    var memSize: Long = 0L
    var diskSize: Long = 0L
    seq.foreach(rdd => {
      numPartitions += rdd.numPartitions
      numCachedPartitions += rdd.numCachedPartitions
      memSize += rdd.memSize
      diskSize += rdd.diskSize
    })
    Tuple4(numPartitions, numCachedPartitions, memSize, diskSize)
  }

  def extractFromTaskMetrics(taskMetrics: TaskMetrics): (Double, Double, Double, Double, Double) =  {
    // cpu time is nanoseconds, run time is milliseconds
    val cpuUtilization: Double = saveDivision(taskMetrics.executorCpuTime, (taskMetrics.executorRunTime * 1000000))
    val gcTimeRatio: Double = saveDivision(taskMetrics.jvmGCTime, taskMetrics.executorRunTime)
    val shuffleReadWriteRatio: Double = saveDivision(taskMetrics.shuffleReadMetrics.totalBytesRead,
      taskMetrics.shuffleWriteMetrics.bytesWritten)
    val inputOutputRatio: Double = saveDivision(taskMetrics.inputMetrics.bytesRead, taskMetrics.outputMetrics.bytesWritten)
    val memorySpillRatio: Double = saveDivision(taskMetrics.diskBytesSpilled, taskMetrics.peakExecutionMemory)
    Tuple5(cpuUtilization, gcTimeRatio, shuffleReadWriteRatio, inputOutputRatio, memorySpillRatio)
  }

  override def onExecutorAdded(executorAdded: SparkListenerExecutorAdded): Unit = {
    handleScaleOutMonitoring(Option(executorAdded.time), Option(executorAdded.executorInfo.executorHost))
  }

  override def onExecutorRemoved(executorRemoved: SparkListenerExecutorRemoved): Unit = {
    handleScaleOutMonitoring(Option(executorRemoved.time), Option("NO_HOST"))
  }

  def handleScaleOutMonitoring(executorActionTime: Option[Long], executorHost: Option[String]): Unit = {

    synchronized {
      if(!active){
        return
      }
       // no scale-out yet? get the number of currently running executors
      if(currentScaleOut.get() == 0){
        currentScaleOut.set(getInitialScaleOutCount(executorHost.getOrElse("NO_HOST")))
      }
      // an executor was removed? Else: an executor was added
      if(executorHost.isDefined && executorHost.get.equals("NO_HOST")){
        currentScaleOut.decrementAndGet()
      }
      else if(executorHost.isDefined){
        currentScaleOut.incrementAndGet()
      }

      logger.info(s"Current number of executors: ${currentScaleOut.get()}.")
      if(executorActionTime.isDefined){
        scaleOutBuffer.append((currentScaleOut.get(), executorActionTime.get))
      }
    }

  }

  override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd): Unit = {

    if(!active){
      return
    }

    logger.info(s"Application ${applicationId} finished.")
  }

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {

    if(!active){
      return
    }

    currentJobId.set(jobStart.jobId)
    allowUpdate.compareAndSet(false, true)

    if(jobStart.jobId == 0){
      handleScaleOutMonitoring(None, None)
    }

    val startScaleOut = getExecutorCount

    // https://stackoverflow.com/questions/29169981/why-is-sparklistenerapplicationstart-never-fired
    // onApplicationStart workaround
    if(jobStart.jobId == 0){
      logger.info(s"Application ${applicationId} started.")

      val updateMap: Map[String, Any] = Map(
        "application_id" -> applicationId,
        "application_signature" -> applicationSignature,
        "attempt_id" -> sparkContext.applicationAttemptId.orNull,
        "start_time" -> jobStart.time,
        "start_scale_out" -> startScaleOut
      )
      updateInformation(Option(applicationId), updateMap, "APPLICATION_START")
    }

    logger.info(s"Job ${jobStart.jobId} started.")

    val mapKey: String = f"appId=${applicationId}-jobId=${jobStart.jobId}"

    infoMap.put(mapKey, scala.collection.mutable.Map[String, Any](
      "job_id" -> jobStart.jobId,
      "start_time" -> jobStart.time,
      "start_scale_out" -> startScaleOut,
      "stages" -> jobStart.stageInfos.map(si => f"${si.stageId}").mkString(",")
    ))
  }

  override def onJobEnd(jobEnd: SparkListenerJobEnd): Unit = {

    if(!active){
      return
    }

    allowUpdate.compareAndSet(true, false)

    logger.info(s"Job ${jobEnd.jobId} finished.")

    val mapKey: String = f"appId=${applicationId}-jobId=${jobEnd.jobId}"

    val rescalingTimeRatio: Double = computeRescalingTimeRatio(
      infoMap.get(mapKey)("start_time").toString.toLong,
      jobEnd.time
    )

    infoMap.put(mapKey, infoMap.get(mapKey).++(scala.collection.mutable.Map[String, Any](
      "end_time" -> jobEnd.time,
      "end_scale_out" -> getExecutorCount,
      "predicted_scale_out" -> predictedScaleOutMap.get(jobEnd.jobId),
      "rescaling_time_ratio" -> rescalingTimeRatio,
      "stages" -> infoMap.get(mapKey)("stages").toString.split(",")
        .map(si => f"${si}" ->  infoMap.get(f"appId=${applicationId}-jobId=${jobEnd.jobId}-stageId=${si}")).toMap
    )))

    predictedScaleOutMap.put(jobEnd.jobId + 1,
      predictedScaleOutMap.getOrDefault(jobEnd.jobId + 1, predictedScaleOutMap.get(jobEnd.jobId)))

    if (predictedScaleOutMap.get(jobEnd.jobId + 1) != predictedScaleOutMap.get(jobEnd.jobId)) {

      logger.info(s"Adjusting predicted scale-out from ${predictedScaleOutMap.get(jobEnd.jobId)} to ${predictedScaleOutMap.get(jobEnd.jobId + 1)}.")
      val requestResult = sparkContext.requestTotalExecutors(predictedScaleOutMap.get(jobEnd.jobId + 1), 0, Map[String, Int]())
      logger.info("Change scaling result: " + requestResult.toString)
    }
    else {
      logger.info(s"Scale-out is not changed.")
    }

    handleRequestScaleOut(jobEnd.jobId)
  }

  override def onStageSubmitted(stageSubmitted: SparkListenerStageSubmitted): Unit = {

    if(!active){
      return
    }

    logger.info(s"Stage ${stageSubmitted.stageInfo.stageId} submitted.")

    val stageInfo: StageInfo = stageSubmitted.stageInfo

    val mapKey: String = f"appId=${applicationId}-jobId=${currentJobId.get()}-stageId=${stageInfo.stageId}"

    infoMap.put(mapKey, scala.collection.mutable.Map[String, Any](
      "start_time" -> stageInfo.submissionTime,
      "start_scale_out" -> getExecutorCount,
      "stage_id" -> f"${stageInfo.stageId}",
      "stage_name" -> stageInfo.name,
      "parent_stage_ids" -> stageInfo.parentIds.map(psi => f"${psi}"),
      "num_tasks" -> stageInfo.numTasks
    ))
  }

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {

    if(!active){
      return
    }

    logger.info(s"Stage ${stageCompleted.stageInfo.stageId} completed.")

    val stageInfo: StageInfo = stageCompleted.stageInfo
    val metricsInfo: (Double, Double, Double, Double, Double) = extractFromTaskMetrics(stageInfo.taskMetrics)

    val mapKey: String = f"appId=${applicationId}-jobId=${currentJobId.get()}-stageId=${stageInfo.stageId}"

    val rescalingTimeRatio: Double = computeRescalingTimeRatio(
      stageInfo.submissionTime.getOrElse(0L),
      stageInfo.completionTime.getOrElse(0L)
    )

    val rddInfo: (Int, Int, Long, Long) = extractFromRDD(stageInfo.rddInfos)

    infoMap.put(mapKey, infoMap.get(mapKey).++(scala.collection.mutable.Map[String, Any](
      "attempt_id" -> stageInfo.attemptNumber(),
      "end_time" -> stageInfo.completionTime,
      "end_scale_out" -> getExecutorCount,
      "rescaling_time_ratio" -> rescalingTimeRatio,
      "failure_reason" -> stageInfo.failureReason.getOrElse(""),
      "rdd_num_partitions" -> rddInfo._1,
      "rdd_num_cached_partitions" -> rddInfo._2,
      "rdd_mem_size" -> rddInfo._3,
      "rdd_disk_size" -> rddInfo._4,
      "metrics" -> scala.collection.mutable.Map[String, Double](
        "cpu_utilization" -> metricsInfo._1,
        "gc_time_ratio" -> metricsInfo._2,
        "shuffle_rw_ratio" -> metricsInfo._3,
        "data_io_ratio" -> metricsInfo._4,
        "memory_spill_ratio" -> metricsInfo._5
      )
    )))
  }

  def getOpenFutures: List[Future[Any]] = {
    futureBuffer.toList.filter(!_.isCompleted)
  }

  def hasOpenFutures: Boolean = {
    getOpenFutures.nonEmpty
  }

  def handleRequestScaleOut(jobId: Int): Unit = {

    val mapKey: String = f"appId=${applicationId}-jobId=${jobId}"
    // check if we are allowed to request prediction and max. number of concurrent requests not yet reached
    val requestPrediction = isAdaptive && method.equals("enel") &&
      (
        concurrentPredictionRequest.compareAndSet(0, 1) ||
        concurrentPredictionRequest.compareAndSet(1, 2) ||
        concurrentPredictionRequest.compareAndSet(2, 3)
        )

    val backend = AsyncHttpClientFutureBackend(
      options = SttpBackendOptions.connectionTimeout(Duration(restTimeout, SECONDS))
    )

    val payload: PredictionRequestPayload = PredictionRequestPayload(
      applicationExecutionId,
      applicationId,
      jobId,
      "JOB_END",
      Json(CustomFormats).write(infoMap.get(mapKey)),
      requestPrediction)

    val response: Future[Response[Either[ResponseException[String, Exception], PredictionResponsePayload]]] = basicRequest
      .post(uri"http://$service:$port/$onlineScaleOutPredictionEndpoint")
      .body(payload)
      .readTimeout(Duration(restTimeout, SECONDS))
      .response(asJson[PredictionResponsePayload])
      .send(backend)

    futureBuffer.append(response)

    for {
      res <- response
    } {
      try{
        val bestScaleOutPerJob: List[List[Int]] = res.body.right.get.best_predicted_scale_out_per_job
        // only proceed if there are successor jobs / we got a prediction result
        val remainingJobs = bestScaleOutPerJob
          .filter(_.head > currentJobId.get())
          .sortBy(_.head)

        if(remainingJobs.nonEmpty) {
          // update tracking values
          if(lastResponseLength.get() == -1 || lastResponseLength.get() > bestScaleOutPerJob.length) {
            lastResponseLength.set(bestScaleOutPerJob.length)
            remainingJobs.foreach(sub_list => {
              if(sub_list.head == (currentJobId.get() + 1) && allowUpdate.get())
              predictedScaleOutMap.put(sub_list.head, sub_list.last)
            })
          }
        }
      }
      finally {
        backend.close()
        // if we requested a prediction: decrement counter of concurrent prediction requests
        if(requestPrediction){
          concurrentPredictionRequest.decrementAndGet()
        }
      }
    }
  }
}
