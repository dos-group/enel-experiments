import copy
from contextlib import redirect_stderr
from io import StringIO
from typing import List
from unittest import TestCase
from unittest.mock import MagicMock, Mock

from enel_service.modeling import request_id, job_database_obj
from enel_service.common.apis.hdfs_api import HdfsApi
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.async_utils import async_return, force_sync
from enel_service.common.configuration import HdfsSettings, MongoSettings
from enel_service.common.db_schemes import JobExecutionModel
from enel_service.config.onlinepredictor_config import OnlinePredictorConfig
from enel_service.modeling.handlers_runtime import handle_online_runtime_prediction
from enel_service.modeling.model_wrappers import OnlinePredictorModel
from enel_service.modeling.transforms import CustomCompose, TransformationHandler
from enel_service.modeling.utils import reset_artifact_cache, prepare_for_inference


# noinspection DuplicatedCode
class TestHandleOnlineRuntimePrediction(TestCase):

    def setUp(self) -> None:
        self.key_prefix: str = f"spark_grep_onlinepredictor"
        self.request_id: str = copy.deepcopy(request_id)
        self.database_obj = copy.deepcopy(job_database_obj)

        self.mongo_api = MongoApi()
        self.mongo_settings = MongoSettings.get_instance()
        self.hdfs_api = HdfsApi()
        self.hdfs_settings = HdfsSettings.get_instance()
        # reset cache
        reset_artifact_cache()

    def test_prediction_error_no_history(self):
        onlinepredictor_config: OnlinePredictorConfig = OnlinePredictorConfig()
        onlinepredictor_config.model_setup["epochs"] = [20, 20]
        onlinepredictor_config.early_stopping["patience"] = 5

        # create artifacts
        data_transformer: CustomCompose = CustomCompose(transformer_specs=onlinepredictor_config.transformer_specs)
        model_wrapper = OnlinePredictorModel(onlinepredictor_config)
        torchscript_model = model_wrapper.get_torchscript_model_instance()

        # fit data transformer
        successor_jobs: List[JobExecutionModel] = [
            JobExecutionModel(**{**self.database_obj, "job_id": self.database_obj["job_id"] + 1}),
            JobExecutionModel(**{**self.database_obj, "job_id": self.database_obj["job_id"] + 2})
        ]
        _, eval_data, db_data = prepare_for_inference(JobExecutionModel(**self.database_obj), "online", successor_jobs)
        data_transformer.fit(eval_data)

        # mock certain behavior
        self.mongo_api.find_one = MagicMock(return_value=async_return(self.database_obj))
        self.hdfs_api.exists_file = MagicMock(return_value=True)
        self.hdfs_api.load = Mock()
        self.hdfs_api.load.side_effect = [
            ({}, False),  # empty checkpoint
            (data_transformer, False),
            (model_wrapper, False),
            (torchscript_model, False)
        ]

        self.mongo_api.aggregate = MagicMock(return_value=async_return([]))

        self.mongo_api.find = Mock()
        self.mongo_api.find.side_effect = [
            async_return([
                {**self.database_obj, "job_id": self.database_obj["job_id"] + 1},
                {**self.database_obj, "job_id": self.database_obj["job_id"] + 2}
            ])
        ]

        with redirect_stderr(StringIO()) as _:
            response = force_sync(handle_online_runtime_prediction(self.request_id, self.hdfs_api, self.mongo_api))
            self.assertEqual(0, len(response.scale_outs))
            self.assertEqual(0, len(response.predicted_job_dict))
            self.assertEqual(0.0, response.fit_time)
            self.assertEqual(0.0, response.predict_time)
            self.assertTrue(response.abort)
            self.mongo_api.aggregate.assert_called_once()
            self.mongo_api.find.assert_not_called()

    def test_prediction_ok_with_history(self):
        onlinepredictor_config: OnlinePredictorConfig = OnlinePredictorConfig()
        onlinepredictor_config.model_setup["epochs"] = [20, 20]
        onlinepredictor_config.early_stopping["patience"] = 5

        # create artifacts
        data_transformer: CustomCompose = CustomCompose(transformer_specs=onlinepredictor_config.transformer_specs)
        model_wrapper = OnlinePredictorModel(onlinepredictor_config)
        torchscript_model = model_wrapper.get_torchscript_model_instance()

        # fit data transformer
        successor_jobs: List[JobExecutionModel] = [
            JobExecutionModel(**{**self.database_obj, "job_id": self.database_obj["job_id"] + 1}),
            JobExecutionModel(**{**self.database_obj, "job_id": self.database_obj["job_id"] + 2})
        ]
        _, eval_data, db_data = prepare_for_inference(JobExecutionModel(**self.database_obj), "online", successor_jobs)

        historic_data: List[TransformationHandler] = []
        for i in range(3):
            for m in successor_jobs + [JobExecutionModel(**self.database_obj)]:
                local_copy = copy.deepcopy(m)
                local_copy.start_scale_out -= i
                local_copy.end_scale_out -= i
                local_copy.application_id = f"new_application_id_{i}"
                for stage_id in sorted(local_copy.stages.keys(), key=lambda key: int(key)):
                    local_copy.stages.get(stage_id).start_scale_out -= i
                    local_copy.stages.get(stage_id).end_scale_out -= i
                historic_data.append(TransformationHandler(local_copy))
        data_transformer.fit(historic_data, suffix="training")

        unique_scale_outs: List[int] = list(set([el.end_scale_out for el in db_data]))

        # mock certain behavior
        self.mongo_api.find_one = MagicMock(return_value=async_return(self.database_obj))
        self.hdfs_api.exists_file = MagicMock(return_value=True)
        self.hdfs_api.load = Mock()
        self.hdfs_api.load.side_effect = [
            ({}, False),  # empty checkpoint
            (data_transformer, False),
            (model_wrapper, False),
            (torchscript_model, False)
        ]
        # we find historical data for predecessor jobs
        self.mongo_api.find = MagicMock(return_value=async_return([self.database_obj,
                                                                   {**self.database_obj,
                                                                    "_id": "another_id1",
                                                                    "job_id": self.database_obj["job_id"] + 1},
                                                                   # now: to "related executions"
                                                                   {**self.database_obj,
                                                                    "application_execution_id": "other1",
                                                                    "application_id": "other1",
                                                                    "_id": "another_id2"},
                                                                   {**self.database_obj,
                                                                    "application_execution_id": "other1",
                                                                    "application_id": "other1",
                                                                    "_id": "another_id3",
                                                                    "job_id": self.database_obj["job_id"] + 1}]))

        # we find historical data for successor jobs
        self.mongo_api.aggregate = MagicMock(return_value=async_return([el.dict() for el in successor_jobs]))

        with redirect_stderr(StringIO()) as _:
            response = force_sync(handle_online_runtime_prediction(self.request_id, self.hdfs_api, self.mongo_api))
            self.assertEqual(len(unique_scale_outs), len(response.scale_outs))
            self.assertTrue(all([isinstance(element, int)] for element in response.scale_outs))
            self.assertEqual(len(successor_jobs), len(response.predicted_job_dict))
            for k, v in response.predicted_job_dict.items():
                self.assertTrue(all([isinstance(elem, tuple) for elem in v]))
            self.assertTrue(isinstance(response.fit_time, float))
            self.assertTrue(isinstance(response.predict_time, float))
            self.assertTrue(isinstance(response.abort, bool))
            self.mongo_api.aggregate.assert_called_once()
            self.mongo_api.find.assert_called_once()
