from datetime import datetime
from typing import List, Dict, Union, Optional

from pydantic import BaseModel, Field


class GlobalSpecsModel(BaseModel):
    solution_name: str = Field(...)  # e.g. "enel", "ellis"
    system_name: str = Field(...)  # e.g. "spark", "flink"
    template_version: str = Field(default="v1")

    experiment_name: str = Field(...)

    algorithm_name: str = Field(...)
    algorithm_args: list = Field(default=[])

    data_size_MB: int = Field(...)
    data_characteristics: str = Field(default="")

    machine_name: str = Field(...)
    environment_name: str = Field(...)

    min_scale_out: int = Field(..., gt=0)
    max_scale_out: int = Field(..., gt=0)
    max_runtime: int = Field(..., gt=0)


class OptionalSpecsModel(BaseModel):
    hadoop_version: str = Field(default=None)
    spark_version: str = Field(default=None)
    scala_version: str = Field(default=None)
    java_version: str = Field(default=None)


class MasterSpecsModel(BaseModel):
    scale_out: int = Field(..., gt=0)
    cores: int = Field(..., gt=0)
    memory: str = Field(...)
    memory_overhead: str = Field(...)


class WorkerSpecsModel(BaseModel):
    scale_out: int = Field(..., gt=0)
    cores: int = Field(..., gt=0)
    memory: str = Field(...)
    memory_overhead: str = Field(...)


class MetricsModel(BaseModel):
    cpu_utilization: float = Field(...)
    gc_time_ratio: float = Field(...)
    shuffle_rw_ratio: float = Field(...)
    data_io_ratio: float = Field(...)
    memory_spill_ratio: float = Field(...)


class ApplicationSubmissionModel(BaseModel):
    global_specs: GlobalSpecsModel = Field(...)
    optional_specs: OptionalSpecsModel = Field(...)
    master_specs: MasterSpecsModel = Field(...)
    worker_specs: WorkerSpecsModel = Field(...)

    flink_template_values: dict = Field(default={})
    spark_template_values: dict = Field(default={})


class CoreDataModel(BaseModel):
    # scale-out at start and end
    start_scale_out: int = Field(..., gt=0)
    end_scale_out: int = Field(..., gt=0)
    # time at start and end
    start_time: datetime = Field(default=None)
    end_time: datetime = Field(default=None)


class ApplicationExecutionModel(ApplicationSubmissionModel, CoreDataModel):
    id: str = Field(..., alias="_id")
    application_id: str = Field(default=None)
    application_signature: str = Field(default=None)

    attempt_id: str = Field(default="")

    # training, prediction, and preparation times
    fit_time: float = Field(default=None)
    predict_time: float = Field(default=None)
    preparation_time: float = Field(default=None)

    # what is the predicted scale-out?
    predicted_scale_out: int = Field(default=None)

    created_at: datetime = Field(default=None)
    updated_at: datetime = Field(default=None)

    class Config:
        allow_population_by_field_name = True


class StageDataModel(CoreDataModel):
    stage_id: int = Field(...)
    stage_name: str = Field(...)
    failure_reason: str = Field(default="")
    attempt_id: int = Field(default=0, ge=0)
    num_tasks: int = Field(...)
    parent_stage_ids: List[int] = Field(default=[])
    # some aggregated task metrics
    metrics: MetricsModel = Field(...)
    # rdd info
    rdd_num_partitions: int = Field(...)
    rdd_num_cached_partitions: int = Field(...)
    rdd_mem_size: int = Field(...)
    rdd_disk_size: int = Field(...)
    # time required for rescaling. In the default case, we dont have a rescaling
    rescaling_time_ratio: float = Field(default=0.0, ge=0.0)


class JobExecutionModel(ApplicationExecutionModel):
    application_execution_id: str
    job_id: int = Field(...)

    stages: Dict[str, StageDataModel] = Field(default={})
    # time required for rescaling. In the default case, we dont have a rescaling
    rescaling_time_ratio: float = Field(default=0.0, ge=0.0)



