import copy
import datetime
from unittest import TestCase
from unittest.mock import MagicMock, call, Mock, patch, AsyncMock, ANY

from enel_service.modeling import request_id, application_database_obj, job_database_obj
from enel_service.common.apis.kubernetes_api import update_dict_func
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.async_utils import async_return, force_sync
from enel_service.common.configuration import MongoSettings
from enel_service.common.db_schemes import JobExecutionModel, ApplicationExecutionModel
from enel_service.modeling.handlers_updating import handle_update_information, update_application_object, update_job_object
from enel_service.modeling.schemes import UpdateInformationRequest


class TestHandleOfflineUpdateInformation(TestCase):

    def setUp(self) -> None:
        self.mongo_api = MongoApi()
        self.mongo_settings = MongoSettings.get_instance()

        self.application_execution_id: str = copy.deepcopy(request_id)
        self.application_id: str = "application123"
        self.app_db_element = copy.deepcopy(application_database_obj)

        self.request = {
            "application_execution_id": self.application_execution_id,
            "application_id": self.application_id,
            "update_event": "APPLICATION_START",
            "updates": {
                "start_time": datetime.datetime.now()
            }
        }

    def test_no_element_found(self):
        request = UpdateInformationRequest(**self.request)

        with patch('motor.motor_asyncio.AsyncIOMotorClient', new_callable=AsyncMock) as client:
            self.mongo_api.client = client
            self.mongo_api.find_one = MagicMock(return_value=async_return(None))
            self.mongo_api.update_one = MagicMock(return_value=async_return(True))
            session_mock = AsyncMock()
            client.configure_mock(**{
                "start_session.return_value.__aexit__.return_value": session_mock,
                "start_session.return_value.__aenter__.return_value": MagicMock(**{
                    "with_transaction.return_value": AsyncMock(return_value=force_sync(update_application_object(request, self.mongo_api, session_mock))),
                })
            })

            force_sync(handle_update_information(request, self.mongo_api))

            self.mongo_api.find_one.assert_called_once_with(
                self.mongo_settings.mongodb_application_execution_collection,
                {"_id": self.application_execution_id},
                catch_error=False, session=ANY)

            self.mongo_api.update_one.assert_not_called()

    def test_update_ok(self):
        request = UpdateInformationRequest(**self.request)

        with patch('motor.motor_asyncio.AsyncIOMotorClient', new_callable=AsyncMock) as client:
            self.mongo_api.client = client
            self.mongo_api.find_one = MagicMock(return_value=async_return(self.app_db_element))
            self.mongo_api.update_one = MagicMock(return_value=async_return(True))
            session_mock = AsyncMock()
            client.configure_mock(**{
                "start_session.return_value.__aexit__.return_value": session_mock,
                "start_session.return_value.__aenter__.return_value": MagicMock(**{
                    "with_transaction.return_value": AsyncMock(
                        return_value=force_sync(update_application_object(request, self.mongo_api, session_mock))),
                })
            })

            force_sync(handle_update_information(request, self.mongo_api))

            self.mongo_api.find_one.assert_called_once_with(
                self.mongo_settings.mongodb_application_execution_collection,
                {"_id": self.application_execution_id},
                catch_error=False, session=ANY)

            app_db_element: ApplicationExecutionModel = ApplicationExecutionModel(**self.app_db_element)
            app_db_element: dict = update_dict_func(app_db_element.dict(), self.request["updates"],
                                                    check_existence=False)
            app_db_element: ApplicationExecutionModel = ApplicationExecutionModel(**app_db_element)

            self.mongo_api.update_one.assert_called_once_with(
                self.mongo_settings.mongodb_application_execution_collection,
                {"_id": request.application_execution_id},
                {"$set": app_db_element.dict(exclude_none=True, exclude={"id"})},
                catch_error=False, session=ANY
            )


class TestHandleOnlineUpdateInformation(TestCase):

    def setUp(self) -> None:
        self.mongo_api = MongoApi()
        self.mongo_settings = MongoSettings.get_instance()

        self.application_execution_id: str = copy.deepcopy(request_id)
        self.application_id: str = "application123"
        self.job_id: int = 1
        self.app_db_element = copy.deepcopy(application_database_obj)
        self.job_db_element = copy.deepcopy(job_database_obj)

        self.request = {
            "application_execution_id": self.application_execution_id,
            "application_id": self.application_id,
            "job_id": self.job_id,
            "update_event": "JOB_END",
            "updates": {
                "stages": {
                    "1": {
                        "stage_id": 1,
                        "stage_name": "stage 1",
                        "num_tasks": 15,
                        "parent_stage_ids": [],
                        "start_time": datetime.datetime(2020, 5, 12, 20, 0),
                        "end_time": datetime.datetime(2020, 5, 12, 20, 20),
                        "start_scale_out": 5,
                        "end_scale_out": 5,
                        "rdd_num_partitions": 10,
                        "rdd_num_cached_partitions": 2,
                        "rdd_mem_size": 12000,
                        "rdd_disk_size": 0,
                        "metrics": {
                            "cpu_utilization": 0.7,
                            "gc_time_ratio": 0.01,
                            "shuffle_rw_ratio": 1.1,
                            "data_io_ratio": 3.0,
                            "memory_spill_ratio": 0.05
                        },
                        "rescaling_time_ratio": 0.0
                    }
                },
            }
        }

    def test_no_element_found(self):
        request = UpdateInformationRequest(**self.request)

        self.mongo_api.find_one = MagicMock(return_value=async_return(None))

        with patch('motor.motor_asyncio.AsyncIOMotorClient', new_callable=AsyncMock) as client:
            self.mongo_api.client = client
            self.mongo_api.find_one = MagicMock(return_value=async_return(None))
            self.mongo_api.update_one = MagicMock(return_value=async_return(True))
            session_mock = AsyncMock()
            client.configure_mock(**{
                "start_session.return_value.__aexit__.return_value": session_mock,
                "start_session.return_value.__aenter__.return_value": MagicMock(**{
                    "with_transaction.return_value": AsyncMock(
                        return_value=force_sync(update_job_object(request, self.mongo_api, session_mock))),
                })
            })

            force_sync(handle_update_information(request, self.mongo_api))

        self.mongo_api.find_one.assert_has_calls([
            call(self.mongo_settings.mongodb_application_execution_collection,
                 {"_id": self.application_execution_id}, catch_error=False, session=ANY)
        ])

    def test_update_ok(self):
        request = UpdateInformationRequest(**self.request)

        with patch('motor.motor_asyncio.AsyncIOMotorClient', new_callable=AsyncMock) as client:
            self.mongo_api.client = client
            self.mongo_api.find_one = Mock()
            self.mongo_api.find_one.side_effect = [async_return(self.app_db_element), async_return(self.job_db_element)]
            self.mongo_api.update_one = MagicMock(return_value=async_return(True))
            session_mock = AsyncMock()
            client.configure_mock(**{
                "start_session.return_value.__aexit__.return_value": session_mock,
                "start_session.return_value.__aenter__.return_value": MagicMock(**{
                    "with_transaction.return_value": AsyncMock(
                        return_value=force_sync(update_job_object(request, self.mongo_api, session_mock))),
                })
            })

            force_sync(handle_update_information(request, self.mongo_api))

            self.mongo_api.find_one.assert_has_calls([
                call(self.mongo_settings.mongodb_application_execution_collection,
                     {"_id": self.application_execution_id}, catch_error=False, session=ANY),
                call(self.mongo_settings.mongodb_job_execution_collection, {
                    "application_execution_id": self.application_execution_id,
                    "application_id": self.application_id,
                    "job_id": self.job_id
                }, catch_error=False, session=ANY)
            ])

            job_db_element: dict = update_dict_func(self.job_db_element, self.request["updates"], check_existence=False)
            job_db_element["application_execution_id"] = self.application_execution_id
            job_db_element["job_id"] = self.job_id
            job_db_element: JobExecutionModel = JobExecutionModel(**job_db_element)
            self.mongo_api.update_one.assert_called_once_with(
                self.mongo_settings.mongodb_job_execution_collection,
                {"application_execution_id": self.application_execution_id,
                 "application_id": self.application_id,
                 "job_id": self.job_id},
                {"$set": job_db_element.dict(by_alias=True, exclude_none=True)},
                upsert=True, catch_error=False, session=ANY, create=False
            )
