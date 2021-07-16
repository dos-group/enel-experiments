# Data Generation, Benchmark Jobs, and custom Spark Listeners

This folder and package is an adapted version of the repository offered with Ellis. It can be found [here](https://github.com/dos-group/runtime-adjustments-experiments). 

Ellis uses Spark listener to implement its tuning strategy on job level.
It is the implementation of
Thamsen, Lauritz, et al. "Ellis: Dynamically Scaling Distributed Dataflows to Meet Runtime Targets." 2017 IEEE International Conference on Cloud Computing Technology and Science (CloudCom). IEEE, 2017.

We compared our approach Enel to Ellis in terms of meeting runtime targets. To do so, we build upon the used benchmark jobs, performed upgrades if necessary, and adapted the original Ellis Spark listener so that it can work in our environment. 

We then implemented our own listener (`EnelScaleOutListener`). It utilizes an http client to communicate with our python service. Also, fine-grained information about statistics are collected. The scale-out is eventually updated based on the response of our python service.


* Environment
  
  - Spark: `v3.1.1`
  - Scala: `2.12.10`
  - H2:    `1.4.194`
    

* Setup H2 Database
  
  * Install and run H2 (Ellis uses this database)
    
    ```bash
    java -cp h2-1.4.194.jar org.h2.tools.Server -webAllowOthers -tcpAllowOthers
    ```
    
  * Initialize H2
    
    Connect to H2 console to create a new database (e.g., test), and execute the script in  `/src/main/resources/db/migration/V1__Initial_version.sql` to create tables.
  

* Usage
  
    The entry class is `de.tu_berlin.dos.arm.spark_utils.adjustments.EllisScaleOutListener`.
    
    You can either set it as an additional listener in Spark (which can be transparent to users), or you can explicitly create it in your Spark applications.
    In order to run the listener independently (without Spark Operator), you can pass the parameters like the example below.
  
    ```bash
    $SPARK_HOME/bin/spark-submit \
        --conf spark.master=local[3] 	\
        --conf spark.customExtraListener.dbPath=tcp://my-server:9092/~/test 	\
        --conf spark.customExtraListener.minExecutors=1 	\
        --conf spark.customExtraListener.maxExecutors=2 	\
        --conf spark.customExtraListener.initialExecutors=1 	\
        --conf spark.customExtraListener.targetRuntimems=600	\
        --k 3 \
        hdfs://my-server:9000/spark/kmeans/small.txt
    ```
  
  * How to choose a tuning mode
  
    Mode | Ellis enabled | Enel enabled
    ---| ---| ---
    Profiling (isAdaptive*=False)|✅|✅
    Ellis Tuning (isAdaptive=True, method**=ellis)|✅|❌ 
    Enel Tuning(isAdaptive=True, method=enel)|❌|✅
  
    * *set isAdaptive by pass value to `spark.customExtraListener.isAdaptive`.
    * **set method by pass value to `spark.customExtraListener.method`.
  
* Dataset Generator
  
  1. Vandermonde and Points 
    
      The generators are in `/src` folder under package `de.tu_berlin.dos.arm.spark_utils.datagens`.  
  
  2. Multiclass
  
      The generator is in `/python` folder.


* Workload
  
  Workloads are located in `de.tu_berlin.dos.arm.spark_utils.jobs`.