The Dockerfile provides a slight modification of the original spark image.
The following changes are made:

- Install `apt-get install procps` so that the runtime is enriched and we can actually use `"spark.executor.processTreeMetrics.enabled": "true"`.
- Create a slim `metrics.properties` file, such that we enable the PrometheusServlet shipped with Spark >=3.0.0. This will greatly simplify the monitoring of our spark applications.

Build the image, push it to your repository and use it in the spark job-template. Make sure to make spark aware of `metrics.properties`, i.e. set the Spark configuration property `"spark.metrics.conf": "$TARGET_DIR"` in `templates/sparkjobcluster.yaml`. Here, $TARGET_DIR denotes the directory specified in Dockerfile.