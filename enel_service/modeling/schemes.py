from typing import Union, List, Optional, Dict, Tuple
from pydantic import BaseModel, Json
from enel_service.common.db_schemes import *


class OfflineRuntimePredictionResponse(BaseModel):
    scale_outs: List[Union[int, float]]
    predicted_runtimes: List[Union[int, float]]
    prepared_db_entries: List[ApplicationExecutionModel]
    fit_time: Optional[float] = 0.0
    predict_time: Optional[float] = 0.0

    class Config:
        schema_extra = {
            "example": {
                "scale_outs": [2, 3, 4, 5, 6],
                "predicted_runtimes": [100, 110, 120, 130],
                "prepared_db_entries": [{}, {}, {}, {}],
                "fit_time": 12.0,
                "predict_time": 0.01
            }
        }


class OfflineScaleOutPredictionResponse(BaseModel):
    prepared_db_entry: ApplicationExecutionModel
    best_scale_out: Union[int, float]
    best_predicted_runtime: Union[int, float]

    class Config:
        schema_extra = {
            "example": {
                "prepared_db_entry": {},
                "best_scale_out": 2,
                "best_predicted_runtime": 100
            }
        }


class TriggerModelTrainingRequest(BaseModel):
    system_name: str
    algorithm_name: str
    model_name: str
    experiment_name: str

    class Config:
        schema_extra = {
            "example": {
                "system_name": "spark",
                "algorithm_name": "LogisticRegression",
                "model_name": "onlinepredictor",
                "experiment_name": "trivial"
            }
        }


class RootDataUpdateModel(ApplicationExecutionModel):
    start_scale_out: Optional[int]
    end_scale_out: Optional[int]

    global_specs: Optional[GlobalSpecsModel]
    optional_specs: Optional[OptionalSpecsModel]
    master_specs: Optional[MasterSpecsModel]
    worker_specs: Optional[WorkerSpecsModel]

    flink_template_values: Optional[dict]
    spark_template_values: Optional[dict]

    id: Optional[str]
    attempt_id: Optional[str]

    application_execution_id: Optional[str]
    job_id: Optional[int]

    predicted_scale_out: Optional[int]

    stages: Optional[Dict[str, StageDataModel]]
    rescaling_time_ratio: Optional[float]


class UpdateInformationRequest(BaseModel):
    application_execution_id: str
    application_id: str
    job_id: Optional[int]
    updates: Union[RootDataUpdateModel,
                   Json[RootDataUpdateModel]]
    update_event: str


class OnlineScaleOutPredictionRequest(UpdateInformationRequest):
    predict: bool


class OnlineRuntimePredictionResponse(BaseModel):
    scale_outs: List[float]
    predicted_job_dict: Dict[str, List[Tuple[int, float, float]]]
    abort: bool
    fit_time: Optional[float] = 0.0
    predict_time: Optional[float] = 0.0


class OnlineScaleOutPredictionResponse(BaseModel):
    best_predicted_scale_out_per_job: Optional[List[Tuple[int, int]]] = []
    best_predicted_runtime_per_job: Optional[List[Tuple[int, float]]] = []

    class Config:
        schema_extra = {
            "example": {
                "best_predicted_scale_out_per_job": [(4, 4)],
                "best_predicted_runtime_per_job": [(4, 78.9)]
            }
        }
