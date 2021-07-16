import logging
from typing import List, Optional, Any, Tuple, Dict

import pymongo
import torch
from fastapi import HTTPException, status

from enel_service.common.apis.hdfs_api import HdfsApi
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.configuration import GeneralSettings, MongoSettings
from enel_service.common.db_schemes import JobExecutionModel
from enel_service.config.onlinepredictor_config import OnlinePredictorConfig
from enel_service.modeling.datasets import ExecutionDataset
from enel_service.modeling.schemes import OnlineRuntimePredictionResponse
from enel_service.modeling.transforms import CustomData
from enel_service.modeling.utils import get_by_execution_id, get_all_artifacts, prepare_for_inference

# load settings
general_settings: GeneralSettings = GeneralSettings.get_instance()
mongo_settings: MongoSettings = MongoSettings.get_instance()
onlinepredictor_config: OnlinePredictorConfig = OnlinePredictorConfig()


async def handle_online_runtime_prediction(job_execution_id: str,
                                           hdfs_api: HdfsApi,
                                           mongo_api: MongoApi):
    # PREPARATION
    # get entry from DB
    db_element: JobExecutionModel = await get_by_execution_id(job_execution_id, mongo_api, scope="job")
    # get artifacts
    checkpoint, data_transformer, model_wrapper, model = get_all_artifacts("onlinepredictor",
                                                                           db_element,
                                                                           hdfs_api)
    # get successor jobs
    related_successor_jobs: List[Any] = await mongo_api. \
        aggregate(mongo_settings.mongodb_job_execution_collection, [{
        "$match": {
            **{f"global_specs.{k}": v for k, v in db_element.global_specs.dict().items()},
            **{f"optional_specs.{k}": v for k, v in db_element.optional_specs.dict().items()},
            'application_signature': db_element.application_signature,
            'job_id': {'$gt': db_element.job_id},
            'start_time': {'$exists': True, '$ne': None},
            'end_time': {'$exists': True, '$ne': None}
        }}, {
        "$group": {
            "_id": "$job_id",
            "doc": {"$first": "$$ROOT"}
        }}, {
        "$replaceRoot": {
            "newRoot": "$doc"
        }}, {
        "$sort": {
            "application_id": pymongo.ASCENDING,
            "job_id": pymongo.ASCENDING
        }}
    ])

    successor_jobs: List[JobExecutionModel] = [JobExecutionModel(**elem_dict) for elem_dict in related_successor_jobs]
    logging.info(f"Number of successor jobs: {len(successor_jobs)}")
    if len(successor_jobs) <= 1:
        return OnlineRuntimePredictionResponse(scale_outs=[], predicted_job_dict={}, abort=True)

    # FINE-TUNING (if possible)
    mongo_search: dict = {
        "filter": {
            'application_execution_id': db_element.application_execution_id,
            'application_signature': db_element.application_signature,
            'job_id': {'$lte': db_element.job_id},
            'start_time': {'$exists': True, '$ne': None},
            'end_time': {'$exists': True, '$ne': None}
        },
        "sort": [
            ("application_id", pymongo.ASCENDING),
            ("job_id", pymongo.ASCENDING)
        ]
    }

    # get predecessor jobs
    predecessor_jobs_dataset: Optional[ExecutionDataset] = None
    try:
        predecessor_jobs_dataset = await ExecutionDataset.from_config("job",
                                                                      mongo_api,
                                                                      mongo_settings.mongodb_job_execution_collection,
                                                                      mongo_search,
                                                                      data_transformer,
                                                                      suffix="training")
    except BaseException as exc:
        logging.warning("Did not find data for fine-tuning", exc_info=exc)
    logging.info(f"Number of predecessor jobs: {len(predecessor_jobs_dataset.raw_file_names)}")
    if len(predecessor_jobs_dataset.raw_file_names) == 0:
        return OnlineRuntimePredictionResponse(scale_outs=[], predicted_job_dict={}, abort=True)

    fit_time: float = 0.0
    if predecessor_jobs_dataset is not None and len(predecessor_jobs_dataset):
        model = model_wrapper.fit(model, predecessor_jobs_dataset, checkpoint)
        fit_time = model_wrapper.fit_time

    # PREDICTION
    scale_out_range, eval_data, db_data = prepare_for_inference(db_element, "online", successor_jobs)
    data_transformer.fit(eval_data, suffix="training")

    # get predictions
    total_runtimes_list: List[List[float]] = []
    total_predict_time: float = 0
    for scale_out in scale_out_range:
        new_eval_data: List[CustomData] = [data_transformer(element) for i, element in enumerate(eval_data)
                                           if db_data[i].end_scale_out == scale_out]

        # batch jobs of each app together in correct order
        new_eval_data = CustomData.to_app_batch_list(new_eval_data)
        result: list = model_wrapper.predict(model, new_eval_data)
        runtimes: torch.Tensor = torch.cat([d["job_runtime_pred"] for d in result if "job_runtime_pred" in d], dim=0)
        total_predict_time += model_wrapper.predict_time
        total_runtimes_list.append(sum(runtimes.tolist(), []))

    if len(total_runtimes_list) != len(scale_out_range):
        logging.error("There are more predicted runtimes than scale-outs!")
        raise HTTPException(status_code=status.HTTP_417_EXPECTATION_FAILED,
                            detail="There are more predicted runtimes than scale-outs!")

    predicted_job_dict: Dict[str, List[Tuple[int, float, float]]] = {}
    for temp_j_idx, temp_job_id in enumerate(range(db_element.job_id + 1, db_element.job_id + 1 + len(successor_jobs))):
        value_list: List[Tuple[int, float, float]] = []
        for sc_idx, scale_out in enumerate(scale_out_range):
            runtimes: List[float] = total_runtimes_list[sc_idx]
            value_list.append((scale_out, sum(runtimes[:temp_j_idx]), sum(runtimes[temp_j_idx:])))
        predicted_job_dict[str(temp_job_id)] = value_list

    return OnlineRuntimePredictionResponse(**{
        "scale_outs": scale_out_range,
        "predicted_job_dict": predicted_job_dict,
        "abort": False,
        "fit_time": fit_time,
        "predict_time": total_predict_time
    })
