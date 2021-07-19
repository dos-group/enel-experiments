# Enel Service
FastAPI python web service that handles the training of models, submission and updating of applications, and predictions.

## Prerequisites

In order to use our service, it is required to have the following things already in place:

- A Kubernetes cluster
- The Spark-Operator (click [here](https://github.com/GoogleCloudPlatform/spark-on-k8s-operator) for more details)
- An HDFS cluster (here, we will store trained models)
- A MongoDB Database (replicaset, as we make use of transactions). Consider using [this](https://github.com/bitnami/charts/tree/master/bitnami/mongodb) helm chart.

## Technical Details

#### Important Packages
* Python `3.8.0+`
* [PyTorch](https://pytorch.org/) `1.8.0`
* [PyTorch Ignite](https://pytorch.org/ignite/) `0.4.2`
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/1.7.2/) `1.7.2`
* [Ray Tune](https://docs.ray.io/en/releases-1.4.0/tune/index.html) `1.4.0`
* [Optuna](https://optuna.org/) `2.8.0`

All necessary packages will be installed in an environment.

#### Install Environment
install dependencies with: 

    conda env [create / update] -f environment.yml

For activation of the environment, execute:

    [source / conda] activate enel_service

#### Start the server

Eventually, start the server:

    python main.py

Assuming you are using the default configuration, you can inspect the API here:

    http://127.0.0.1:5000/docs

## Environment Variables

When defined, the respective environment variables will be used and default values will be overridden.
Some of the most important are listed below. A full list can be inferred from `common/configuration.py`, 
where each class property refers to a case-insensitive environment variable (with default values if not specified). 

|Variable | Description|
|---|---|
|`HOST` | host to run the application, default: 127.0.0.1|
|`PORT` | port to run the application, default: 5000|
|`LOGGING_LEVEL` | logging level, default: INFO|
|`KUBERNETES_ENDPOINT` | ip:port of kubernetes master|
|`KUBERNETES_NAMESPACE` | namespace to deploy to, default: default|
|`KUBERNETES_API_KEY` | BearerToken to authenticate to k8s master|
|`KUBERNETES_API_KEY_PREFIX` | Prefix for API key. default: Bearer|
|`KUBECONFIG` | Path to kubernetes-config
|`HDFS_ENDPOINT` | ip:port of hdfs master|
|`HDFS_OUTPUT_DIR` | output directory for persisted metrics|
|`MONGODB_ENDPOINT`| The endpoint of the mongoDB database, default: 127.0.0.1|
|`MONGODB_PORT`| The port of the mongoDB database, default: 27017|
|`MONGODB_PASSWORD`| The password for connecting to the mongoDB database|

All other possible environment variables can be inferred from `common/configuration.py`.


## Hyperparameter Optimization
During Model-Pretraining, we use [Ray Tune](https://docs.ray.io/en/releases-1.4.0/tune/index.html) and [Optuna](https://optuna.org/) to carry out a search for optimized hyperparameters. As of now, we draw 40 trials using [Optuna](https://optuna.org/) from a manageable search space. All details are listed in the `onlinepredictor_config.py` and are in most cases directly passed to the respective functions / methods / classes in [Ray Tune](https://docs.ray.io/en/releases-1.4.0/tune/index.html). For instance, it can be inferred from the `onlinepredictor_config.py` file that we used the [ASHAScheduler](https://docs.ray.io/en/releases-1.4.0/tune/api_docs/schedulers.html?highlight=asha#asha-tune-schedulers-ashascheduler) implemented in Ray Tune to early terminate bad trials, pause trials, clone trials, and alter hyperparameters of a running trial. Also, we used Optuna's default sampler, namely [TPESampler](https://optuna.readthedocs.io/en/v2.8.0/reference/generated/optuna.samplers.TPESampler.html), to provide trial suggestions.


## Pretraining of Models
Firstly, we need to make sure that a pretrained model is available. In a production environment, this could be assured by a periodically executed job. For instance, given that you already have historical data in your mongoDB database, call the endpoint

     http://<my-server>:<my-port>/training/trigger_model_training
with the necessary arguments to make the service fetch data from mongoDB, split it, and conduct a training given the provided configuration. The best model is then saved to HDFS.

## Submission of Applications
An application can be submitted by calling the endpoint

     http://<my-server>:<my-port>/submission/submit
with the required arguments. This will determine a good initial scale-out, and submit the application to the internally used spark-operator. It will also pass certain properties to the application, which can then be used by our implemented spark listeners.

## Dynamic Adjustments
If configured accordingly, the listener will send updates to our service and also request scale-out recommendations. This happens via the

     http://<my-server>:<my-port>/prediction/online_scale_out_prediction
endpoint. The Enel-service then gets model artifacts, performs fine-tuning on the available job-information, and predicts the most suitable scale-out for the remaining job runtime. As of know, for this to work, the artifacts first need to be fetched, as this will lead to caching and faster access times later. This can be done using the endpoint

     http://<my-server>:<my-port>/prediction/preload/{base_name}
where `base_name` is one of `[gbt, mpc, logisticregression, kmeans]`, i.e. one of our used benchmark jobs.

# Questions? Something does not work or remains unclear?
Please get in touch, we are happy to help!
