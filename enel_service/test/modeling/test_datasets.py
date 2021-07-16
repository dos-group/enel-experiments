import copy
from typing import List
from unittest import TestCase
from unittest.mock import MagicMock

from enel_service.config.onlinepredictor_config import OnlinePredictorConfig
from enel_service.modeling import request_id, job_database_obj
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.async_utils import async_return, force_sync
from enel_service.common.configuration import MongoSettings
from enel_service.modeling.datasets import ExecutionDataset, ExecutionSubset
from enel_service.modeling.transforms import CustomCompose


class TestDatasets(TestCase):

    def setUp(self) -> None:
        self.request_id: str = copy.deepcopy(request_id)
        self.job_database_obj = copy.deepcopy(job_database_obj)

        self.mongo_api = MongoApi()
        self.mongo_settings = MongoSettings.get_instance()

    def test_dataset(self):
        self.mongo_api.find = MagicMock(return_value=async_return([self.job_database_obj]))
        dataset = force_sync(ExecutionDataset.from_config("job",
                                                          self.mongo_api,
                                                          self.mongo_settings.mongodb_job_execution_collection,
                                                          query={"filter": {}}))

        self.mongo_api.find.assert_called_once_with(self.mongo_settings.mongodb_job_execution_collection, filter={})
        self.assertEqual(len(dataset), 1)

        subset = dataset[:1]
        self.assertEqual(len(subset), 1)

        subset = dataset[1:]
        self.assertEqual(len(subset), 0)

    def test_subset(self):
        objects: List[dict] = [copy.deepcopy(self.job_database_obj) for _ in range(3)]
        for i, o in enumerate(objects):
            o["_id"] = str(i)
            o["job_id"] = i
            o["application_execution_id"] = "0"

        data_transformer: CustomCompose = CustomCompose(transformer_specs=OnlinePredictorConfig().transformer_specs)

        self.mongo_api.find = MagicMock(return_value=async_return(objects))
        dataset: ExecutionDataset = force_sync(
            ExecutionDataset.from_config("job",
                                         self.mongo_api,
                                         self.mongo_settings.mongodb_job_execution_collection,
                                         query={"filter": {}},
                                         pre_transform=data_transformer))

        self.mongo_api.find.assert_called_once_with(self.mongo_settings.mongodb_job_execution_collection, filter={})
        self.assertEqual(len(dataset.raw_file_names), 3)

        subset1 = ExecutionSubset.from_parent(dataset, ["0"])
        self.assertEqual(len(subset1.sub_ids), 1)
        self.assertEqual(len(subset1), 1)
        self.assertTrue(objects[0]["job_id"] in subset1.get(0)["job_id"].reshape(-1).tolist())
        self.assertTrue(objects[1]["job_id"] in subset1.get(0)["job_id"].reshape(-1).tolist())
        self.assertTrue(objects[2]["job_id"] in subset1.get(0)["job_id"].reshape(-1).tolist())

        with self.assertRaises(ValueError):
            ExecutionSubset.from_parent(dataset, ["1"])
