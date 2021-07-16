ARG SPARK_IMAGE=gcr.io/spark-operator/spark:v3.0.0
FROM ${SPARK_IMAGE}

USER root

# Information about memory usage for Spark version >=3.0.0
# Requires proc filesystem (procfs) and the command pgrep available in runtime
RUN yes | apt-get install procps

USER ${spark_uid}

ARG TARGET_DIR=/etc/metrics/conf
RUN mkdir -p ${TARGET_DIR}
# Configuration for PrometheusServlet: Native Metric Monitoring via Prometheus for Spark version >= 3.0
RUN echo "*.sink.prometheusServlet.class=org.apache.spark.metrics.sink.PrometheusServlet" > ${TARGET_DIR}/metrics.properties
RUN echo "*.sink.prometheusServlet.path=/metrics/prometheus" >> ${TARGET_DIR}/metrics.properties
RUN echo "master.sink.prometheusServlet.path=/metrics/master/prometheus" >> ${TARGET_DIR}/metrics.properties
RUN echo "applications.sink.prometheusServlet.path=/metrics/applications/prometheus" >> ${TARGET_DIR}/metrics.properties
# Enable JvmSource
RUN echo "driver.source.jvm.class=org.apache.spark.metrics.source.JvmSource" >> ${TARGET_DIR}/metrics.properties
RUN echo "executor.source.jvm.class=org.apache.spark.metrics.source.JvmSource" >> ${TARGET_DIR}/metrics.properties

ENTRYPOINT ["/opt/entrypoint.sh"]
