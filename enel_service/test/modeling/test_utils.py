import asyncio
import copy
from unittest import TestCase
from unittest.mock import MagicMock, call, Mock

from fastapi import HTTPException, status

from enel_service.modeling import request_id, application_database_obj
from enel_service.common.apis.hdfs_api import HdfsApi
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.async_utils import async_return
from enel_service.common.configuration import HdfsSettings, MongoSettings
from enel_service.common.db_schemes import ApplicationExecutionModel
from enel_service.modeling.utils import reset_artifact_cache, get_all_artifacts, get_by_execution_id


class TestGetAllArtifacts(TestCase):
    def setUp(self) -> None:
        self.model_name: str = "onlinepredictor"
        self.key_prefix: str = f"spark_grep_{self.model_name}"
        self.db_element: ApplicationExecutionModel = ApplicationExecutionModel(**application_database_obj)
        self.hdfs_api = HdfsApi()
        self.hdfs_settings = HdfsSettings.get_instance()
        # reset cache
        reset_artifact_cache()

    def test_checkpoint_precondition_failed(self):
        self.hdfs_api.exists_file = MagicMock(return_value=False)

        with self.assertRaises(HTTPException) as exc:
            get_all_artifacts(self.model_name, self.db_element, self.hdfs_api)
        self.assertEqual(exc.exception.status_code, status.HTTP_412_PRECONDITION_FAILED)
        self.assertEqual(exc.exception.detail, "No training checkpoint yet available.")

        self.hdfs_api.exists_file.assert_has_calls([
            call(f"{self.key_prefix}_checkpoint.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_data_transformer.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_model_wrapper.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_torchscript_model.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir)
        ])

    def test_data_transformer_precondition_failed(self):
        self.hdfs_api.exists_file = Mock()
        self.hdfs_api.exists_file.side_effect = [True, False, False, False]
        self.hdfs_api.load = Mock()
        self.hdfs_api.load.side_effect = [({}, False), None, None, None]

        with self.assertRaises(HTTPException) as exc:
            get_all_artifacts(self.model_name, self.db_element, self.hdfs_api)
        self.assertEqual(exc.exception.status_code, status.HTTP_412_PRECONDITION_FAILED)
        self.assertEqual(exc.exception.detail, "No fitted data-transformer yet available.")

        self.hdfs_api.exists_file.assert_has_calls([
            call(f"{self.key_prefix}_checkpoint.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_data_transformer.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_model_wrapper.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_torchscript_model.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir)
        ])

        self.hdfs_api.load.assert_called_once_with(f"{self.key_prefix}_checkpoint.pt",
                                                   target_dir=self.hdfs_settings.hdfs_pretrained_models_subdir)

    def test_model_wrapper_precondition_failed(self):
        self.hdfs_api.exists_file = Mock()
        self.hdfs_api.exists_file.side_effect = [True, True, False, False]
        self.hdfs_api.load = Mock()
        self.hdfs_api.load.side_effect = [({}, False), (lambda x: x, False), None, None]

        with self.assertRaises(HTTPException) as exc:
            get_all_artifacts(self.model_name, self.db_element, self.hdfs_api)
        self.assertEqual(exc.exception.status_code, status.HTTP_412_PRECONDITION_FAILED)
        self.assertEqual(exc.exception.detail, "No fitted model wrapper yet available.")

        self.hdfs_api.exists_file.assert_has_calls([
            call(f"{self.key_prefix}_checkpoint.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_data_transformer.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_model_wrapper.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_torchscript_model.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir)
        ])

        self.hdfs_api.load.assert_has_calls([
            call(f"{self.key_prefix}_checkpoint.pt", target_dir=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_data_transformer.pt", target_dir=self.hdfs_settings.hdfs_pretrained_models_subdir)
        ])

    def test_torchscript_model_precondition_failed(self):
        self.hdfs_api.exists_file = Mock()
        self.hdfs_api.exists_file.side_effect = [True, True, True, False]
        self.hdfs_api.load = Mock()
        self.hdfs_api.load.side_effect = [({}, False), (lambda x: x, False), ({}, False), None]

        with self.assertRaises(HTTPException) as exc:
            get_all_artifacts(self.model_name, self.db_element, self.hdfs_api)
        self.assertEqual(exc.exception.status_code, status.HTTP_412_PRECONDITION_FAILED)
        self.assertEqual(exc.exception.detail, "No fitted model yet available.")

        self.hdfs_api.exists_file.assert_has_calls([
            call(f"{self.key_prefix}_checkpoint.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_data_transformer.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_model_wrapper.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_torchscript_model.pt", directory=self.hdfs_settings.hdfs_pretrained_models_subdir)
        ])

        self.hdfs_api.load.assert_has_calls([
            call(f"{self.key_prefix}_checkpoint.pt", target_dir=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_data_transformer.pt", target_dir=self.hdfs_settings.hdfs_pretrained_models_subdir),
            call(f"{self.key_prefix}_model_wrapper.pt", target_dir=self.hdfs_settings.hdfs_pretrained_models_subdir)
        ])


class TestGetByExecutionId(TestCase):
    def setUp(self) -> None:
        self.db_element_id: str = copy.deepcopy(request_id)

        self.mongo_api = MongoApi()
        self.mongo_settings = MongoSettings.get_instance()

    def test_precondition_failed(self):
        self.mongo_api.find_one = MagicMock(return_value=async_return(None))

        with self.assertRaises(HTTPException) as exc:
            asyncio.run(get_by_execution_id(self.db_element_id, self.mongo_api))
        self.assertEqual(exc.exception.status_code, status.HTTP_412_PRECONDITION_FAILED)
        self.assertEqual(exc.exception.detail, "No element found in DB with specified Id.")
        self.mongo_api.find_one.assert_called_once_with("application_execution", {"_id": self.db_element_id})

        self.mongo_api.find_one.reset_mock()

        with self.assertRaises(HTTPException) as exc:
            asyncio.run(get_by_execution_id(self.db_element_id, self.mongo_api, scope="job"))
        self.assertEqual(exc.exception.status_code, status.HTTP_412_PRECONDITION_FAILED)
        self.assertEqual(exc.exception.detail, "No element found in DB with specified Id.")
        self.mongo_api.find_one.assert_called_once_with("job_execution", {"_id": self.db_element_id})
