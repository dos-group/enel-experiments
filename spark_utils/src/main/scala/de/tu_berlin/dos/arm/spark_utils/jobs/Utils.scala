package de.tu_berlin.dos.arm.spark_utils.jobs

import org.apache.spark.SparkConf

object Utils {


  def isListenerEnabled(sparkConf: SparkConf, listenerName: String): Boolean ={
    /**
     * check whether a listener is enabled
     */
    val isAdaptive: Boolean = sparkConf.getBoolean("spark.customExtraListener.isAdaptive", defaultValue = true)
    val method: String = sparkConf.get("spark.customExtraListener.method")
    val enabled = !isAdaptive || method.equals(listenerName)
    enabled
  }

  def isEnelEnabled(sparkConf: SparkConf): Boolean ={
    isListenerEnabled(sparkConf, "enel")
  }

  def isEllisEnabled(sparkConf: SparkConf): Boolean ={
    isListenerEnabled(sparkConf, "ellis")
  }

}
