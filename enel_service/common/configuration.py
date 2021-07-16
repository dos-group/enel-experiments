import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings

# https://fastapi.tiangolo.com/advanced/settings/?h=envir
root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GeneralSettings(BaseSettings):
    root_dir: Optional[str] = root_dir
    logging_level: Optional[str] = "INFO"
    num_workers: Optional[int] = 1
    host: Optional[str] = "127.0.0.1"
    port: Optional[int] = 5000
    # will be overridden
    config_dir: Optional[str] = "config"
    temp_dir: Optional[str] = "temp"
    evaluation_mode: Optional[str] = "PROD"

    @staticmethod
    @lru_cache()
    def get_instance():
        general_settings: GeneralSettings = GeneralSettings()
        general_settings.config_dir = os.path.join(general_settings.root_dir, general_settings.config_dir)
        general_settings.temp_dir = os.path.join(general_settings.root_dir, general_settings.temp_dir)
        return general_settings


class PredictionSettings(BaseSettings):
    prediction_rescaling_time_ratio: Optional[float] = 0.9
    prediction_step_size_scale_out: Optional[float] = 0.125
    prediction_num_neighbors: Optional[int] = 5

    @staticmethod
    @lru_cache()
    def get_instance():
        return PredictionSettings()


class HdfsSettings(BaseSettings):
    hdfs_endpoint: Optional[str] = ""
    hdfs_output_dir: Optional[str] = "enel_service"
    hdfs_data_subdir: Optional[str] = "data"
    hdfs_pretrained_models_subdir: Optional[str] = "pretrained_models"

    @staticmethod
    @lru_cache()
    def get_instance():
        return HdfsSettings()


class KubernetesSettings(BaseSettings):
    kubernetes_endpoint: Optional[str] = ""
    kubernetes_namespace: Optional[str] = "enel_service"
    kubernetes_api_key: Optional[str] = ""
    kubernetes_api_key_prefix: Optional[str] = "Bearer"

    @staticmethod
    @lru_cache()
    def get_instance():
        return KubernetesSettings()


class MongoSettings(BaseSettings):
    mongodb_endpoint: Optional[str] = ""
    mongodb_port: Optional[int] = 27017
    mongodb_database: Optional[str] = "enel_service"
    mongodb_username: Optional[str] = "root"
    mongodb_password: Optional[str] = "servicerootpassword"
    mongodb_application_execution_collection: Optional[str] = "application_execution"
    mongodb_job_execution_collection: Optional[str] = "job_execution"

    @staticmethod
    @lru_cache()
    def get_instance():
        return MongoSettings()


class EllisSettings(BaseSettings):
    py4j_address: Optional[str] = "localhost"
    py4j_port: Optional[int] = 25333

    @staticmethod
    @lru_cache()
    def get_instance():
        return EllisSettings()


class ListenerSettings(BaseSettings):
    drms_rest_timeout: Optional[int] = 30
    drms_service: Optional[str] = "admin-service"
    drms_online_scale_out_prediction_endpoint: Optional[str] = "prediction/online_scale_out_prediction"
    drms_update_information_endpoint: Optional[str] = "training/update_information"

    @staticmethod
    @lru_cache()
    def get_instance():
        return ListenerSettings()
