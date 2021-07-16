import copy
from unittest import TestCase
from unittest.mock import MagicMock

import pymongo
from fastapi import HTTPException, status

from enel_service.modeling import request_id, job_database_obj, application_database_obj
from enel_service.common.apis.fs_api import FsApi
from enel_service.common.apis.hdfs_api import HdfsApi
from enel_service.common.apis.kubernetes_api import KubernetesApi
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.async_utils import async_return, force_sync
from enel_service.common.configuration import MongoSettings, HdfsSettings
from enel_service.modeling.handlers_training import handle_trigger_model_training
from enel_service.modeling.schemes import TriggerModelTrainingRequest
from enel_service.modeling.utils import reset_artifact_cache


class TestHandleTriggerModelTraining(TestCase):

    def setUp(self) -> None:
        self.fs_api = FsApi()
        self.mongo_api = MongoApi()
        self.mongo_settings = MongoSettings.get_instance()
        self.hdfs_api = HdfsApi()
        self.hdfs_settings = HdfsSettings.get_instance()
        self.kubernetes_api = KubernetesApi()

        self.request = {
            "system_name": "spark",
            "algorithm_name": "grep",
            "model_name": "onlinepredictor",
            "experiment_name": "trivial"
        }

        self.request_id: str = copy.deepcopy(request_id)
        self.job_db_element = copy.deepcopy(job_database_obj)
        self.app_db_element = copy.deepcopy(application_database_obj)

        # reset cache
        reset_artifact_cache()

    def test_wrong_model_name(self):
        self.request["model_name"] = "lala"
        request = TriggerModelTrainingRequest(**self.request)

        with self.assertRaises(HTTPException) as exc:
            force_sync(handle_trigger_model_training(request, self.mongo_api, self.hdfs_api))
        self.assertEqual(exc.exception.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(exc.exception.detail,
                         "Invalid Model-Type specified. Must be one of ['onlinepredictor'].")

    def test_not_data_available(self):
        request = TriggerModelTrainingRequest(**self.request)

        self.mongo_api.find = MagicMock(return_value=async_return([]))

        with self.assertRaises(HTTPException) as exc:
            force_sync(handle_trigger_model_training(request, self.mongo_api, self.hdfs_api))
        self.assertEqual(exc.exception.status_code, status.HTTP_412_PRECONDITION_FAILED)
        self.assertEqual(exc.exception.detail, "No data for pretraining available.")
        self.mongo_api.find.assert_called_once_with(self.mongo_settings.mongodb_job_execution_collection,
                                                    filter={'start_time': {'$exists': True, '$ne': None},
                                                            'end_time': {'$exists': True, '$ne': None},
                                                            'global_specs.system_name': 'spark',
                                                            'global_specs.algorithm_name': 'grep',
                                                            'global_specs.experiment_name': 'trivial'},
                                                    sort=[("application_id", pymongo.ASCENDING),
                                                          ("job_id", pymongo.ASCENDING)])

    def test_trigger_online_ok(self):
        request = TriggerModelTrainingRequest(**self.request)

        self.hdfs_api.exists_file = MagicMock(return_value=True)
        self.mongo_api.find = MagicMock(return_value=async_return([self.job_db_element]))

        force_sync(handle_trigger_model_training(request, self.mongo_api, self.hdfs_api))

        self.mongo_api.find.assert_called_once_with(self.mongo_settings.mongodb_job_execution_collection,
                                                    filter={'start_time': {'$exists': True, '$ne': None},
                                                            'end_time': {'$exists': True, '$ne': None},
                                                            "global_specs.system_name": "spark",
                                                            "global_specs.algorithm_name": "grep",
                                                            'global_specs.experiment_name': 'trivial'},
                                                    sort=[("application_id", pymongo.ASCENDING),
                                                          ("job_id", pymongo.ASCENDING)])

    def test_trigger_offline_ok(self):
        request = TriggerModelTrainingRequest(**self.request)

        self.hdfs_api.exists_file = MagicMock(return_value=True)
        self.mongo_api.find = MagicMock(return_value=async_return([self.app_db_element]))

        force_sync(handle_trigger_model_training(request, self.mongo_api, self.hdfs_api))

        self.mongo_api.find.assert_called_once_with(self.mongo_settings.mongodb_application_execution_collection,
                                                    filter={'start_time': {'$exists': True, '$ne': None},
                                                            'end_time': {'$exists': True, '$ne': None},
                                                            "global_specs.system_name": "spark",
                                                            "global_specs.algorithm_name": "grep",
                                                            'global_specs.experiment_name': 'trivial'})
