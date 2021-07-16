import datetime
import logging

from fastapi import HTTPException, status
from typing import List, Union, Any
import numpy as np
import pymongo
import torch

from enel_service.common.apis.hdfs_api import HdfsApi
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.configuration import MongoSettings, GeneralSettings
from enel_service.common.db_schemes import ApplicationExecutionModel, JobExecutionModel
from enel_service.config.onlinepredictor_config import OnlinePredictorConfig
from enel_service.modeling.bell_utils import AllocationAssistant
from enel_service.modeling.schemes import OfflineScaleOutPredictionResponse, OfflineRuntimePredictionResponse
from enel_service.modeling.transforms import CustomData
from enel_service.modeling.utils import get_all_artifacts, prepare_for_inference

# load settings
general_settings: GeneralSettings = GeneralSettings.get_instance()
mongo_settings: MongoSettings = MongoSettings.get_instance()
onlinepredictor_config: OnlinePredictorConfig = OnlinePredictorConfig()


async def handle_initial_runtime_prediction(app_db_element: ApplicationExecutionModel, hdfs_api, mongo_api):
    # get artifacts
    checkpoint, data_transformer, model_wrapper, model = get_all_artifacts("onlinepredictor",
                                                                           app_db_element,
                                                                           hdfs_api)

    # get successor jobs
    first_job_past_values: List[Any] = await mongo_api. \
        aggregate(mongo_settings.mongodb_job_execution_collection, [{
        "$match": {
            **{f"global_specs.{k}": v for k, v in app_db_element.global_specs.dict().items()},
            **{f"optional_specs.{k}": v for k, v in app_db_element.optional_specs.dict().items()},
            'application_signature': app_db_element.global_specs.algorithm_name,
            'job_id': {'$eq': 0},
            'start_time': {'$exists': True, '$ne': None},
            'end_time': {'$exists': True, '$ne': None}
        }}, {
        "$project": {
            "start_time": 1,
            "end_time": 1,
            "application_execution_id": 1
        }}, {
        "$lookup": {
            "from": mongo_settings.mongodb_application_execution_collection,
            "localField": "application_execution_id",
            "foreignField": "_id",
            "as": "main_app"
        }}, {
        "$project": {
            "duration": {"$divide": [{"$subtract": ["$end_time", "$start_time"]}, 1000]},
            "scale_out": {"$first": "$main_app.predicted_scale_out"}
        }}
    ])

    durations = [d["duration"] for d in first_job_past_values]
    scale_outs = [d["scale_out"] for d in first_job_past_values]

    print("Durations:", durations)
    print("Scale-Outs:", scale_outs)

    # get successor jobs
    related_successor_jobs: List[Any] = await mongo_api. \
        aggregate(mongo_settings.mongodb_job_execution_collection, [{
        "$match": {
            **{f"global_specs.{k}": v for k, v in app_db_element.global_specs.dict().items()},
            **{f"optional_specs.{k}": v for k, v in app_db_element.optional_specs.dict().items()},
            'application_signature': app_db_element.global_specs.algorithm_name,
            'job_id': {'$gt': 0},
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

    first_job_dummy: JobExecutionModel = JobExecutionModel(**{
        **successor_jobs[0].dict(),
        "job_id": 0,
        "application_id": "dummy_application_id",
        "application_execution_id": "dummy_application_execution_id"
    })

    # PREDICTION
    scale_out_range, eval_data, db_data = prepare_for_inference(first_job_dummy, "initial", successor_jobs)
    data_transformer.fit(eval_data, suffix="training")

    bell_pred_times = AllocationAssistant()\
        .fit((np.array(scale_outs), np.array(durations)))\
        .predict((np.array(scale_out_range), None))

    # get predictions
    total_runtimes_list: List[float] = []
    total_predict_time: float = 0
    for scale_out in scale_out_range:
        new_eval_data: List[CustomData] = [data_transformer(element) for i, element in enumerate(eval_data)
                                           if db_data[i].end_scale_out == scale_out]

        # batch jobs of each app together in correct order
        new_eval_data = CustomData.to_app_batch_list(new_eval_data)

        result: list = model_wrapper.predict(model, new_eval_data)
        runtimes: torch.Tensor = torch.cat([d["job_runtime_pred"] for d in result if "job_runtime_pred" in d], dim=0)
        total_predict_time += model_wrapper.predict_time
        total_runtimes_list.append(sum(sum(runtimes.tolist(), [])) + bell_pred_times[scale_out_range.index(scale_out)])

    if len(total_runtimes_list) != len(scale_out_range):
        logging.error("There are more predicted runtimes than scale-outs!")
        raise HTTPException(status_code=status.HTTP_417_EXPECTATION_FAILED,
                            detail="There are more predicted runtimes than scale-outs!")

    _, _, db_data = prepare_for_inference(app_db_element, "offline", None)

    indices = [scale_out_range.index(sc) for sc in list(range(min(scale_outs), max(scale_outs) + 1))]

    return OfflineRuntimePredictionResponse(**{
        "scale_outs": [scale_out_range[i] for i in indices],
        "predicted_runtimes": [total_runtimes_list[i] for i in indices],
        "prepared_db_entries": [db_data[i] for i in indices],
        "predict_time": total_predict_time
    })


async def handle_initial_scale_out_prediction(app_db_element: ApplicationExecutionModel,
                                              hdfs_api: HdfsApi,
                                              mongo_api: MongoApi):
    prep_start: datetime = datetime.datetime.now()

    response: OfflineRuntimePredictionResponse = await handle_initial_runtime_prediction(app_db_element,
                                                                                         hdfs_api,
                                                                                         mongo_api)

    preparation_time: float = (datetime.datetime.now().replace(tzinfo=None) -
                               prep_start.replace(tzinfo=None)).total_seconds()
    preparation_time -= (response.fit_time + response.predict_time)

    scale_outs: List[Union[int, float]] = response.scale_outs
    predicted_runtimes: List[Union[int, float]] = response.predicted_runtimes
    prepared_db_entries: List[ApplicationExecutionModel] = response.prepared_db_entries

    # find index of entry with shortest runtime
    min_element_index: int = next((i for i, pred_rt in enumerate(predicted_runtimes)
                                   if pred_rt < (prepared_db_entries[i].global_specs.max_runtime * 0.9)),
                                  predicted_runtimes.index(min(predicted_runtimes)))

    best_scale_out: Union[int, float] = scale_outs[min_element_index]
    best_predicted_runtime: Union[int, float] = predicted_runtimes[min_element_index]

    prepared_db_entry: ApplicationExecutionModel = prepared_db_entries[min_element_index]
    prepared_db_entry.fit_time = response.fit_time
    prepared_db_entry.predict_time = response.predict_time
    prepared_db_entry.preparation_time = preparation_time
    prepared_db_entry.start_scale_out = best_scale_out
    prepared_db_entry.end_scale_out = best_scale_out

    return OfflineScaleOutPredictionResponse(**{
        "prepared_db_entry": prepared_db_entry,
        "best_scale_out": best_scale_out,
        "best_predicted_runtime": best_predicted_runtime
    })
