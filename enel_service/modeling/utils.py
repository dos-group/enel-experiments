import copy
import logging
from functools import partial, lru_cache
from typing import Optional, Union, Any, List

from bson import ObjectId
from fastapi import HTTPException, status
from torch.jit import RecursiveScriptModule

from enel_service.common.apis.hdfs_api import HdfsApi
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.configuration import HdfsSettings, MongoSettings, GeneralSettings, PredictionSettings

# load hdfs settings
from enel_service.common.db_schemes import JobExecutionModel, ApplicationExecutionModel, MetricsModel
from enel_service.config.onlinepredictor_config import OnlinePredictorConfig
from enel_service.modeling.model_wrappers import OnlinePredictorModel
from enel_service.modeling.models import AutoEncoder
from enel_service.modeling.transforms import CustomCompose, TransformationHandler

general_settings: GeneralSettings = GeneralSettings.get_instance()
hdfs_settings: HdfsSettings = HdfsSettings.get_instance()
mongo_settings: MongoSettings = MongoSettings.get_instance()
prediction_settings: PredictionSettings = PredictionSettings.get_instance()


@lru_cache(maxsize=None)
def cache_artifact(hdfs_api: HdfsApi, file_name: str):
    if hdfs_api.exists_file(file_name, directory=hdfs_settings.hdfs_pretrained_models_subdir):
        artifact, _ = hdfs_api.load(file_name,
                                    target_dir=hdfs_settings.hdfs_pretrained_models_subdir)
        return artifact
    else:
        return None


def load_artifact(hdfs_api: HdfsApi, file_name: str):
    artifact: Optional[Any] = cache_artifact(hdfs_api, file_name)
    if artifact is not None:
        return copy.deepcopy(artifact) if "torchscript" not in file_name else artifact.__copy__()
    return artifact


def reset_artifact_cache():
    cache_artifact.cache_clear()


async def preload_artifacts(base_name: str, hdfs_api: HdfsApi):

    algo_map: dict = {
        "kmeans": "KMeans",
        "gbt": "GradientBoostedTrees",
        "logisticregression": "LogisticRegression",
        "mpc": "MPC"
    }

    algo = algo_map.get(base_name, "")

    logging.info(f"Prefetch artifacts for algorithm '{algo}'...")
    key_prefix = f"spark_{algo}_onlinepredictor"
    get_all_artifacts("onlinepredictor", key_prefix, hdfs_api)
    logging.info("Success!")


def save_artifact(hdfs_api: HdfsApi, obj: any, file_name: str):
    hdfs_api.save(obj,
                  file_name,
                  mode="w",
                  target_dir=hdfs_settings.hdfs_pretrained_models_subdir)


async def get_by_execution_id(execution_id: str, mongo_api: MongoApi, scope: str = "application"):
    target_collection: str = mongo_settings.mongodb_application_execution_collection
    if scope != "application":
        target_collection = mongo_settings.mongodb_job_execution_collection

    db_element: Optional[Any] = await mongo_api.find_one(target_collection,
                                                         {"_id": execution_id})
    if db_element is None:
        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED,
                            detail="No element found in DB with specified Id.")
    return JobExecutionModel(**db_element) if scope != "application" else ApplicationExecutionModel(**db_element)


def get_all_artifacts(model_name: str, db_element: Union[ApplicationExecutionModel, JobExecutionModel, str],
                      hdfs_api: HdfsApi):
    get_artifact = partial(load_artifact, hdfs_api)
    strict: bool = general_settings.evaluation_mode != "TEST"
    # extract request parameters
    key_prefix: str = db_element
    if not isinstance(db_element, str):
        key_prefix: str = f"{db_element.global_specs.system_name}_{db_element.global_specs.algorithm_name}_{model_name}"
    # get artifacts from cache or hdfs
    checkpoint: Optional[dict] = get_artifact(f"{key_prefix}_checkpoint.pt")
    data_transformer: Optional[CustomCompose] = get_artifact(f"{key_prefix}_data_transformer.pt")
    model_wrapper: Optional[AutoEncoder, OnlinePredictorModel] = get_artifact(f"{key_prefix}_model_wrapper.pt")
    model: Optional[RecursiveScriptModule] = get_artifact(f"{key_prefix}_torchscript_model.pt")

    if checkpoint is None:
        logging.warning("No training checkpoint yet available.")
        if strict:
            raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED,
                                detail="No training checkpoint yet available.")
        else:
            checkpoint = {}

    if data_transformer is None:
        logging.warning("No fitted data-transformer yet available.")
        if strict:
            raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED,
                                detail="No fitted data-transformer yet available.")
        else:
            config = OnlinePredictorConfig()
            data_transformer = CustomCompose(transformer_specs=config.transformer_specs)

    if model_wrapper is None:
        logging.warning("No fitted model wrapper yet available.")
        if strict:
            raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED,
                                detail="No fitted model wrapper yet available.")
        else:
            config_class = OnlinePredictorConfig
            wrapper_class = OnlinePredictorModel
            model_wrapper = wrapper_class(config_class())

    if model is None:
        logging.warning("No fitted model yet available.")
        if strict:
            raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED,
                                detail="No fitted model yet available.")
        else:
            model = model_wrapper.get_torchscript_model_instance(checkpoint,
                                                                 log_meta_information=True,
                                                                 disable_autoencoder=True)

    model = model_wrapper.check_device(model, checkpoint)

    return checkpoint, data_transformer, model_wrapper, model


def prepare_for_inference(db_element: Union[ApplicationExecutionModel, JobExecutionModel],
                          prediction_type: str, successor_jobs: Optional[List[JobExecutionModel]]):
    if prediction_type not in ["initial", "online", "offline"]:
        raise ValueError(f"'prediction_type' must be one of ['initial', 'online', 'offline']")
    if prediction_type == "online" and (successor_jobs is None or len(successor_jobs) == 0):
        raise ValueError("'successor jobs' must be provided when doing online predictions.")

    db_min_scale_out: int = db_element.global_specs.min_scale_out
    db_max_scale_out: int = db_element.global_specs.max_scale_out

    scale_out_range: List[int] = []
    db_data: List[Union[ApplicationExecutionModel, JobExecutionModel]] = []  # used for actual db update, if necessary
    eval_data: List[TransformationHandler] = []  # used for prediction pipeline

    if prediction_type == "offline":
        scale_out_range = [int(number) for number in range(db_min_scale_out, db_max_scale_out + 1)]
        for so in scale_out_range:
            temp_db_element: ApplicationExecutionModel = ApplicationExecutionModel(**copy.deepcopy(db_element.dict()))
            temp_db_element.start_scale_out = so
            temp_db_element.end_scale_out = so
            db_data.append(temp_db_element)
            eval_data.append(TransformationHandler(ApplicationExecutionModel(**copy.deepcopy(temp_db_element.dict()))))
    elif prediction_type in ["initial", "online"]:

        current_scale_out: Optional[int] = None
        ratio: float

        if prediction_type == "online":
            ratio = prediction_settings.prediction_rescaling_time_ratio
            current_scale_out = db_element.end_scale_out
            upper_bound: int = current_scale_out
            upper_bound += int(prediction_settings.prediction_step_size_scale_out * (db_max_scale_out - db_min_scale_out))
            scale_out_range = [int(number) for number in range(db_min_scale_out, min(db_max_scale_out, upper_bound) + 1)]
        else:
            ratio = 0
            scale_out_range = [int(number) for number in range(db_min_scale_out, db_max_scale_out + 1)]

        for so in scale_out_range:
            for succ_job in successor_jobs:
                temp_succ_job: JobExecutionModel = JobExecutionModel(**copy.deepcopy(succ_job.dict()))

                # check if direct successor, for this we assume rescaling overhead
                if temp_succ_job.job_id == db_element.job_id + 1:
                    temp_succ_job.start_scale_out = current_scale_out or so
                    temp_succ_job.end_scale_out = so
                    temp_succ_job.rescaling_time_ratio = ratio
                    for stage_id, stage in temp_succ_job.stages.items():
                        temp_succ_job.stages.get(stage_id).start_scale_out = current_scale_out or so
                        temp_succ_job.stages.get(stage_id).end_scale_out = so
                        temp_succ_job.stages.get(stage_id).rescaling_time_ratio = ratio
                else:
                    temp_succ_job.start_scale_out = so
                    temp_succ_job.end_scale_out = so
                    temp_succ_job.rescaling_time_ratio = 0
                    for stage_id, stage in temp_succ_job.stages.items():
                        temp_succ_job.stages.get(stage_id).start_scale_out = so
                        temp_succ_job.stages.get(stage_id).end_scale_out = so
                        temp_succ_job.stages.get(stage_id).rescaling_time_ratio = 0

                # needs to have a different idea
                temp_succ_job.id = str(ObjectId())
                # rest: clean up, or use information from real ref element
                temp_succ_job.application_id = db_element.application_id
                temp_succ_job.application_execution_id = db_element.application_execution_id
                temp_succ_job.attempt_id = ""
                temp_succ_job.start_time = None
                temp_succ_job.end_time = None
                for stage_id, stage in temp_succ_job.stages.items():
                    temp_succ_job.stages.get(stage_id).failure_reason = ""
                    temp_succ_job.stages.get(stage_id).attempt_id = 0
                    temp_succ_job.stages.get(stage_id).metrics = MetricsModel(**{
                        "cpu_utilization": 0,
                        "gc_time_ratio": 0,
                        "shuffle_rw_ratio": 0,
                        "data_io_ratio": 0,
                        "memory_spill_ratio": 0
                    })
                    temp_succ_job.stages.get(stage_id).start_time = None
                    temp_succ_job.stages.get(stage_id).end_time = None

                db_data.append(temp_succ_job)
                eval_data.append(TransformationHandler(JobExecutionModel(**copy.deepcopy(temp_succ_job.dict()))))

    return scale_out_range, eval_data, db_data
