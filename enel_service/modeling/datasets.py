import copy
import os
from typing import List, Union
import torch
from torch_geometric.data import InMemoryDataset

from enel_service.common import id_generator, create_dirs
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.configuration import GeneralSettings
from enel_service.common.db_schemes import JobExecutionModel, ApplicationExecutionModel
from enel_service.modeling.transforms import CustomData, CustomCompose, TransformationHandler

general_settings: GeneralSettings = GeneralSettings.get_instance()


class ExecutionDataset(InMemoryDataset):
    def __init__(self, scope: str, dataset_id: str, path_to_dataset: str, raw_file_names: List[str],
                 mongo_api: MongoApi, collection: str, query: dict,
                 pre_transform: CustomCompose = None, suffix="inference"):

        self.mongo_api: MongoApi = mongo_api
        self.collection: str = collection
        self.query: dict = query

        self.scope = scope
        self.suffix: str = suffix

        self.__raw_file_names__: List[str] = raw_file_names
        self.__processed_file_names__: List[str] = [f"main_{dataset_id}.pt"]

        super(ExecutionDataset, self).__init__(path_to_dataset, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @staticmethod
    async def from_config(scope: str, mongo_api: MongoApi, collection: str, query: dict, *args, **kwargs):
        dataset_id: str = id_generator(30)
        path_to_dataset = os.path.join(general_settings.temp_dir, dataset_id)
        create_dirs(path_to_dataset)
        create_dirs(os.path.join(path_to_dataset, "raw"))
        create_dirs(os.path.join(path_to_dataset, "processed"))

        db_elements: List[dict] = await mongo_api.find(collection, **query)
        new_raw_file_names: List[str] = []
        for db_element in db_elements:
            data_model: Union[ApplicationExecutionModel, JobExecutionModel] = JobExecutionModel(**db_element) \
                if scope != "application" else ApplicationExecutionModel(**db_element)
            torch.save(data_model, os.path.join(path_to_dataset, "raw", f"{scope}_{data_model.id}.pt"))
            new_raw_file_names.append(f"{scope}_{data_model.id}.pt")

        return ExecutionDataset(scope, dataset_id, path_to_dataset,
                                new_raw_file_names,
                                mongo_api, collection, query, *args, **kwargs)

    @property
    def data_transformer(self):
        return self.pre_transform

    @data_transformer.setter
    def data_transformer(self, value):
        self.pre_transform = value

    @property
    def raw_file_names(self):
        return self.__raw_file_names__

    @property
    def processed_file_names(self):
        return self.__processed_file_names__

    def process(self):
        data_model_list: List[Union[ApplicationExecutionModel, JobExecutionModel]] = [
            torch.load(os.path.join(self.raw_dir, file_name))
            for file_name in self.raw_file_names]

        data_list: List[TransformationHandler] = [TransformationHandler(el) for el in data_model_list]
        real_data_list: List[CustomData] = [el.to_data_object() for el in data_list]
        if self.pre_transform is not None and isinstance(self.pre_transform, CustomCompose):
            if hasattr(self.pre_transform, "fit"):
                self.pre_transform.fit(data_list, suffix=self.suffix)
            real_data_list = [self.pre_transform(el) for el in data_list]

            if all([type(model) == JobExecutionModel for model in data_model_list]):
                # batch jobs of each app together in correct order
                real_data_list = CustomData.to_app_batch_list(real_data_list)

        if not len(real_data_list):
            raise ValueError("Dataset is empty.")

        data, slices = self.collate(real_data_list)
        torch.save((data, slices), self.processed_paths[0])


class ExecutionSubset(InMemoryDataset):
    def __init__(self, parent: ExecutionDataset,
                 ids: List[str], pre_transform: CustomCompose = None, suffix="inference", **kwargs):
        subset_id: str = id_generator(30)
        self.sub_ids: List[str] = ids

        self.scope = parent.scope
        self.suffix: str = suffix

        self.__raw_file_names__: List[str] = parent.raw_file_names
        self.__processed_file_names__: List[str] = [f"sub_{self.suffix}_{subset_id}.pt"]

        super(ExecutionSubset, self).__init__(parent.root, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @staticmethod
    def from_parent(parent: ExecutionDataset, ids: List[str], suffix: str = "training", **kwargs):
        data_transformer: CustomCompose = copy.deepcopy(parent.data_transformer)
        if suffix == "training":
            data_transformer = CustomCompose.from_ref(parent.data_transformer)

        return ExecutionSubset(
            parent,
            ids,
            data_transformer,
            suffix=suffix,
            **kwargs
        )

    @property
    def data_transformer(self):
        return self.pre_transform

    @data_transformer.setter
    def data_transformer(self, value):
        self.pre_transform = value

    @property
    def raw_file_names(self):
        return self.__raw_file_names__

    @property
    def processed_file_names(self):
        return self.__processed_file_names__

    def process(self):
        data_model_list: List[Union[ApplicationExecutionModel, JobExecutionModel]] = [
            torch.load(os.path.join(self.raw_dir, file_name))
            for file_name in self.raw_file_names]
        data_model_list = [model for model in data_model_list if
                           model.dict().get("application_execution_id" if
                                            self.scope == "job" else "_id", "") in self.sub_ids]

        data_list: List[TransformationHandler] = [TransformationHandler(el) for el in data_model_list]
        real_data_list: List[CustomData] = [el.to_data_object() for el in data_list]
        if self.pre_transform is not None and isinstance(self.pre_transform, CustomCompose):
            if hasattr(self.pre_transform, "fit"):
                self.pre_transform.fit(data_list, suffix=self.suffix)
            real_data_list = [self.pre_transform(el) for el in data_list]

            if all([type(model) == JobExecutionModel for model in data_model_list]):
                # batch jobs of each app together in correct order
                real_data_list = CustomData.to_app_batch_list(real_data_list)

        if not len(real_data_list):
            raise ValueError("Subset is empty.")

        data, slices = self.collate(real_data_list)
        torch.save((data, slices), self.processed_paths[0])
