import datetime
from typing import Union, List, Optional, Dict, Tuple
from fastapi import BackgroundTasks
import logging

from enel_service.common.apis.hdfs_api import HdfsApi
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.configuration import MongoSettings
from enel_service.common.db_schemes import JobExecutionModel, ApplicationExecutionModel
from enel_service.modeling.handlers_runtime import handle_online_runtime_prediction
from enel_service.modeling.handlers_updating import handle_update_information
from enel_service.modeling.schemes import \
    OnlineScaleOutPredictionRequest, OnlineScaleOutPredictionResponse, UpdateInformationRequest, \
    OnlineRuntimePredictionResponse

mongo_settings: MongoSettings = MongoSettings.get_instance()


async def handle_online_scale_out_prediction(request: OnlineScaleOutPredictionRequest,
                                             background_tasks: BackgroundTasks,
                                             hdfs_api: HdfsApi,
                                             mongo_api: MongoApi):
    logging_prefix: str = f"[Application-Execution-Id: {request.application_execution_id},  " \
                          f"Application-Id: {request.application_id}, " \
                          f"Job-Id: {request.job_id}]"

    prep_start: datetime = datetime.datetime.now()

    update_information_request: UpdateInformationRequest = UpdateInformationRequest(**request.dict())

    # if request was only to report new data and no prediction needed, we simply return
    if not request.predict:
        background_tasks.add_task(handle_update_information, update_information_request, mongo_api)
        return OnlineScaleOutPredictionResponse()
    else:
        job_db_element: Optional[JobExecutionModel] = await handle_update_information(update_information_request,
                                                                                      mongo_api)
    if job_db_element is not None:
        app_db_element: ApplicationExecutionModel = ApplicationExecutionModel(
            **(await mongo_api.find_one(
                mongo_settings.mongodb_application_execution_collection,
                {"_id": job_db_element.application_execution_id})))

        response: OnlineRuntimePredictionResponse = await handle_online_runtime_prediction(job_db_element.id,
                                                                                           hdfs_api,
                                                                                           mongo_api)

        # if not enough successor or predecessor jobs, we simply return
        if response.abort:
            return OnlineScaleOutPredictionResponse()

        preparation_time: float = (datetime.datetime.now().replace(tzinfo=None) -
                                   prep_start.replace(tzinfo=None)).total_seconds()
        preparation_time -= (response.fit_time + response.predict_time)

        current_runtime: float = (job_db_element.end_time.replace(tzinfo=None) -
                                  app_db_element.start_time.replace(tzinfo=None)).total_seconds()
        current_runtime += response.fit_time + response.predict_time + preparation_time
        logging.info(f"{logging_prefix} Current runtime: {current_runtime:.2f}s")
        remaining_runtime: float = (app_db_element.global_specs.max_runtime * 1.05) - current_runtime
        logging.info(f"{logging_prefix} Remaining runtime: {remaining_runtime:.2f}s")

        # extract from response
        scale_outs: List[float] = response.scale_outs
        predicted_job_dict: Dict[str, List[Tuple[int, float, float]]] = response.predicted_job_dict

        best_predicted_scale_out_per_job: List[Tuple[int, int]] = []
        best_predicted_runtime_per_job: List[Tuple[int, float]] = []

        for job_id, tup_list in sorted(predicted_job_dict.items(), key=lambda item: int(item[0])):
            # each tuple has following form: <Scale-Out, Predicted passed time, Predicted remaining time>
            pred_rts: List[float] = [el[-1] for el in tup_list]
            rem_rts: List[float] = [remaining_runtime - el[1] for el in tup_list]
            # find index of entry with shortest runtime
            min_element_index: int = next(
                (i for i in range(len(tup_list)) if pred_rts[i] < rem_rts[i]),
                pred_rts.index(min(pred_rts)))
            # find corresponding elements
            best_scale_out: int = int(scale_outs[min_element_index])
            best_predicted_runtime: float = pred_rts[min_element_index]
            # append to collector lists
            best_predicted_scale_out_per_job.append((int(job_id), best_scale_out))
            best_predicted_runtime_per_job.append((int(job_id), best_predicted_runtime))

        await mongo_api.update_one(mongo_settings.mongodb_job_execution_collection,
                                   {"_id": job_db_element.id},
                                   {"$set": {
                                       "fit_time": response.fit_time,
                                       "predict_time": response.predict_time,
                                       "preparation_time": preparation_time
                                   }})

        return OnlineScaleOutPredictionResponse(
            best_predicted_scale_out_per_job=best_predicted_scale_out_per_job,
            best_predicted_runtime_per_job=best_predicted_runtime_per_job
        )
