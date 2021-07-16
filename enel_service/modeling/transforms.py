from __future__ import annotations

import copy

import numpy as np
import datetime
import logging
import math
from collections import OrderedDict
from typing import Union, Optional, List, Tuple, Any, Dict, Set
from itertools import product

import torch
from sklearn.feature_extraction.text import HashingVectorizer as HashVec
from torch_geometric.data import Data, Batch, DataLoader

from enel_service.common.configuration import PredictionSettings
from enel_service.common.db_schemes import JobExecutionModel, \
    StageDataModel, ApplicationExecutionModel, GlobalSpecsModel, OptionalSpecsModel

prediction_settings: PredictionSettings = PredictionSettings.get_instance()


class CustomData(Data):
    def __init__(self, **kwargs):
        self.prev_job_batch = None
        self.stage_context_batch = None
        self.context_opt_batch = None
        self.context_emb_batch = None

        self.application_id: Optional[str] = None
        self.job_id: Optional[int] = None

        kwargs.setdefault("context_emb", torch.tensor([]))
        kwargs.setdefault("context_opt", torch.tensor([]))
        kwargs.setdefault("stage_context", torch.tensor([]))
        kwargs.setdefault("edge_index", None)
        kwargs.setdefault("face", None)

        super().__init__(**kwargs)

    def __inc__(self, key, value):
        if key == 'context_emb_batch':
            return torch.max(self.context_emb_batch) + 1 if len(self.context_emb_batch) else 0
        if key == 'context_opt_batch':
            return torch.max(self.context_opt_batch) + 1 if len(self.context_opt_batch) else 0
        if key == 'stage_context_batch':
            return torch.max(self.stage_context_batch) + 1 if len(self.stage_context_batch) else 0
        if key == 'prev_job_batch':
            return self.num_nodes
        if key == 'real_nodes_batch':
            return self.num_nodes
        else:
            return super().__inc__(key, value)

    @staticmethod
    def prepare_batch_object(element: Batch):
        # special treatment for online predictor
        if hasattr(element, "real_ptr"):
            element.ptr = getattr(element, "real_ptr")
        if hasattr(element, "real_batch"):
            element.batch = getattr(element, "real_batch")
        if hasattr(element, "real_num_nodes"):
            element.num_nodes = getattr(element, "real_num_nodes")
        return element

    @staticmethod
    def to_app_batch_list(elements: List[CustomData]):
        """Only use with List of CustomData that originates from JobExecutionModel."""
        root_dict: Dict[str, Dict[str, CustomData]] = {}
        for element in elements:
            if element.application_id not in root_dict:
                root_dict[element.application_id] = {}
            root_dict[element.application_id][f"{element.job_id}"] = element

        result_list: List[Batch] = []
        for application_id in sorted(root_dict.keys()):
            job_dict: Dict[str, CustomData] = root_dict[application_id]
            values_list: List[CustomData] = [job_dict[k] for k in sorted(job_dict.keys(), key=lambda key: int(key))]
            batch_element: Batch = next(iter(DataLoader(values_list, batch_size=len(values_list), shuffle=False)))

            batch_element.real_ptr = batch_element.ptr
            batch_element.real_batch = batch_element.batch
            hasattr(batch_element, "ptr") and delattr(batch_element, "ptr")
            hasattr(batch_element, "batch") and delattr(batch_element, "batch")
            batch_element.real_num_nodes = sum(batch_element.__num_nodes_list__)
            batch_element.num_nodes = batch_element.real_num_nodes

            result_list.append(batch_element)
        return result_list


class TransformationHandler(object):
    def __init__(self, model: Union[ApplicationExecutionModel, JobExecutionModel]):
        self.__internal_model__: Union[ApplicationExecutionModel, JobExecutionModel] = model
        self.__result_dict__: OrderedDict = OrderedDict()

    @staticmethod
    def collapse(tuple_list: List[Tuple[str, Any]]) -> Any:
        if len(tuple_list):
            first_tuple: Tuple[str, Any] = tuple_list[0]
            if isinstance(first_tuple, tuple) and len(first_tuple) == 2:
                return first_tuple[1]

    def set_dict_value(self, key: str, value: Any):
        self.__result_dict__[key] = value

    def get_sub_keys_by_key(self, key_string: str):
        sub_keys: List[str] = [key for key in list(self.__result_dict__.keys()) if key_string in key]
        return sorted(sub_keys)

    def get_values(self, key_string: str) -> List[Tuple[str, Any]]:
        if key_string in self.__result_dict__:
            return [(key_string, self.__result_dict__[key_string])]

        result_list: List[Optional[Tuple[str, Any]]] = [("", self.__internal_model__)]
        key_list: List[str] = key_string.split(".")

        while len(key_list):
            first_key: str = key_list.pop(0)
            result_list_length: int = len(result_list)  # as we extend the list on the fly, get start length here
            for i in range(result_list_length):
                new_tuple: Tuple[str, Any] = result_list[i]
                if new_tuple is None:
                    continue
                new_key, new_value = new_tuple
                if first_key != "$":  # this is not a dict
                    if hasattr(new_value, first_key):
                        extended_key: str = f"{new_key}{'.' if len(new_key) else ''}{first_key}"
                        result_list[i] = (extended_key, getattr(new_value, first_key))
                    else:
                        result_list[i] = None
                else:  # this is a dict
                    for k, v in new_value.items():
                        extended_key: str = f"{new_key}{'.' if len(new_key) else ''}{k}"
                        result_list.append((extended_key, v))
                    result_list[i] = None

        return [my_tuple for my_tuple in result_list if my_tuple is not None]

    def to_data_object(self):
        result_dict: OrderedDict = copy.deepcopy(self.__result_dict__)
        for k, v in result_dict.items():
            if isinstance(v, (int, float)) and k != "num_nodes":
                result_dict[k] = torch.tensor([[v]], dtype=torch.double)
        result_dict.update({"id": self.__internal_model__.id})
        if type(self.__internal_model__) == JobExecutionModel:
            result_dict.update({
                "application_execution_id": self.__internal_model__.application_execution_id,
                "application_id": self.__internal_model__.application_id,
                "job_id": self.__internal_model__.job_id
            })
        for k in list(result_dict.keys()):
            if "." in k:
                result_dict.pop(k, None)
        return CustomData(**result_dict)


class DictCamelCaser(object):
    @staticmethod
    def __transform_key__(key: str):
        parts: List[str] = key.split("_")
        # capitalize all first chars
        parts: List[str] = [f"{el[0].upper()}{el[1:].lower()}" for el in parts]
        if len(parts):
            # lower case first element
            parts[0] = parts[0].lower()
        return "".join(parts)

    def __camel_case__(self, data_dict: dict):
        for k, v in list(data_dict.items()):
            new_k = self.__transform_key__(k)
            if isinstance(v, dict):
                data_dict[new_k] = self.__camel_case__(v)
            else:
                data_dict[new_k] = v
        return data_dict

    def __call__(self, data_dict: dict):
        result_dict = self.__camel_case__(data_dict)
        return result_dict


class BaseTransformer(object):
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


# noinspection SpellCheckingInspection
class ContextEmbedder(BaseTransformer):
    def __init__(self, context_emb_columns: list = None, context_opt_columns: list = None):
        if context_emb_columns is None:
            context_emb_columns = []
        if context_opt_columns is None:
            context_opt_columns = []

        self.context_emb_columns = context_emb_columns
        self.context_opt_columns = context_opt_columns

    def __call__(self, handler: TransformationHandler):
        # concat required values
        emb_vals: List[Tuple[str, Any]] = sum([handler.get_values(col) for col in self.context_emb_columns], [])
        emb_vals: List[Any] = [el[1] for el in emb_vals]
        context_emb = torch.tensor([])
        if len(emb_vals):
            context_emb = torch.cat(emb_vals, 0).reshape(len(emb_vals), -1)

        # concat optional values
        opt_vals: List[Tuple[str, Any]] = sum([handler.get_values(col) for col in self.context_opt_columns], [])
        opt_vals: List[Any] = [el[1] for el in opt_vals]
        context_opt = torch.tensor([])
        if len(opt_vals):
            context_opt = torch.cat(opt_vals, 0).reshape(len(opt_vals), -1)

        # set fields
        handler.set_dict_value("context_emb", context_emb)
        handler.set_dict_value("context_opt", context_opt)
        return handler


# noinspection SpellCheckingInspection
class HashingVectorizer(BaseTransformer):
    def __init__(self, target_fields: list = None, out_dim: int = 40, **kwargs):
        if target_fields is None:
            target_fields = []
        self.target_fields = target_fields

        self.hashing_vectorizer = HashVec(n_features=out_dim - 1, **kwargs)
        self.vocabulary = list("abcdefghijklmnopqrstuvwxyz0123456789.:,;_-+#@")

    def __transform__(self, value: str) -> List[float]:
        char_list = list(str(value))
        # remove undesired characters
        char_list = [c for c in char_list if c in self.vocabulary]
        value = "".join(char_list)
        # get encoding from hashing vectorizer
        base_encoding: List[float] = self.hashing_vectorizer.transform([value]).toarray().reshape(-1).tolist()

        return [0.0] + base_encoding  # category "hasher"

    def __call__(self, handler: TransformationHandler):
        for field in self.target_fields:
            source_value_list: List[Tuple[str, Any]] = handler.get_values(field)
            for key, source_value in source_value_list:
                if isinstance(source_value, list):
                    source_value = " ".join(source_value)
                target_value = self.__transform__(source_value)
                if isinstance(target_value, list) and all([isinstance(el, list) for el in target_value]):
                    target_value = sum(target_value, [])
                handler.set_dict_value(key, torch.tensor(target_value).reshape(1, -1))
        return handler


# noinspection SpellCheckingInspection
class MinMaxScaler(BaseTransformer):
    def __init__(self, target_fields: list = None, target_min: float = 0., target_max: float = 1.):
        if target_fields is None:
            target_fields = []
        self.target_fields = target_fields

        self.target_min = target_min
        self.target_max = target_max

        self.min_values = {}
        self.max_values = {}

    def _transform(self, tensor, min_vals, max_vals):
        denom = max_vals - min_vals
        denom[denom == 0] = 1  # Prevent division by 0

        nom = (tensor - min_vals) * (self.target_max - self.target_min)
        tensor = (self.target_min + nom) / denom
        return tensor

    def fit(self, handler: TransformationHandler):
        for field in self.target_fields:
            temp_list: List[Tuple[str, Any]] = handler.get_values(field)
            for key, temp_value in temp_list:
                if not isinstance(temp_value, torch.Tensor):
                    if isinstance(temp_value, (int, float)):
                        temp_value = torch.tensor([temp_value]).reshape(1, -1)
                    else:
                        logging.debug(f"Unhandled case so far with value={temp_value}")
                min_vals = torch.min(temp_value, dim=0, keepdim=True)[0]
                max_vals = torch.max(temp_value, dim=0, keepdim=True)[0]

                if key in self.min_values:
                    min_vals = torch.min(torch.cat((min_vals, self.min_values[key]), dim=0), dim=0, keepdim=True)[0]
                self.min_values[key] = min_vals

                if key in self.max_values:
                    max_vals = torch.max(torch.cat((max_vals, self.max_values[key]), dim=0), dim=0, keepdim=True)[0]
                self.max_values[key] = max_vals
        return handler

    def __call__(self, handler: TransformationHandler):
        for field in self.target_fields:
            temp_list: List[Tuple[str, Any]] = handler.get_values(field)
            for key, temp_value in temp_list:
                handler.set_dict_value(key,
                                       self._transform(temp_value, self.min_values[key], self.max_values[key]))
        return handler


class BinaryTransformer(BaseTransformer):
    def __init__(self, target_fields: list = None, out_dim: int = 40):
        if target_fields is None:
            target_fields = []
        self.target_fields = target_fields
        self.out_dim = out_dim

    @staticmethod
    def __resolve_value__(x: Union[int, float, str]):
        if isinstance(x, str):
            x = x.lower()
            if not x.isdecimal():
                if x[-1] == "m":  # specification in MB? e.g. memory
                    x = x[:-1]
                else:
                    raise ValueError("Expand method.")

        return int(float(x))

    def __get_bin__(self, x: Union[int, float, str]):
        x = 0 if (isinstance(x, str) and len(x) == 0) else self.__resolve_value__(x)
        value_list = list(map(int, list(format(x, 'b').zfill(self.out_dim - 1))[-(self.out_dim - 1):]))
        return [1] + value_list  # category "binarizer"

    def __call__(self, handler: TransformationHandler):
        for field in self.target_fields:
            temp_list: List[Tuple[str, Any]] = handler.get_values(field)
            for key, temp_value in temp_list:
                handler.set_dict_value(key, torch.tensor(self.__get_bin__(temp_value)).reshape(1, -1))
        return handler


# noinspection SpellCheckingInspection
class ScaleOutEnricher(BaseTransformer):
    def __init__(self, tol: float = 0.000001):
        self.tol = tol

    def __enrich__(self, data_value: Union[int, float]):
        log_value = math.log1p(float(data_value))
        div_value = 1 - (1. / (float(data_value) + self.tol))
        return torch.tensor([float(data_value), log_value, div_value]).reshape(1, -1)

    def __call__(self, handler: TransformationHandler):

        if type(handler.__internal_model__) == ApplicationExecutionModel:
            handler.set_dict_value("app_start_scale_out_vec",
                                   self.__enrich__(handler.collapse(handler.get_values("start_scale_out"))))
            handler.set_dict_value("app_end_scale_out_vec",
                                   self.__enrich__(handler.collapse(handler.get_values("end_scale_out"))))

        if type(handler.__internal_model__) == JobExecutionModel:
            handler.set_dict_value("job_start_scale_out_vec",
                                   self.__enrich__(handler.collapse(handler.get_values("start_scale_out"))))
            handler.set_dict_value("job_end_scale_out_vec",
                                   self.__enrich__(handler.collapse(handler.get_values("end_scale_out"))))

            stages: Dict[str, StageDataModel] = handler.collapse(handler.get_values("stages"))

            start_scale_out_vec_tensor_list: List[torch.Tensor] = []
            end_scale_out_vec_tensor_list: List[torch.Tensor] = []
            for stage_id in sorted(stages.keys(), key=lambda key: int(key)):
                start_scale_out: Union[int, float] = stages.get(stage_id).start_scale_out
                start_scale_out_vec_tensor_list.append(self.__enrich__(start_scale_out))
                end_scale_out: Union[int, float] = stages.get(stage_id).end_scale_out
                end_scale_out_vec_tensor_list.append(self.__enrich__(end_scale_out))

            handler.set_dict_value("stage_start_scale_out_vec",
                                   torch.cat(start_scale_out_vec_tensor_list or [torch.tensor([])], dim=0))
            handler.set_dict_value("stage_end_scale_out_vec",
                                   torch.cat(end_scale_out_vec_tensor_list or [torch.tensor([])], dim=0))
        return handler


class StageGraphCreator(BaseTransformer):
    @staticmethod
    def __create_edge_index__(stages: Dict[str, StageDataModel]) -> Tuple[torch.Tensor, int, List[int]]:
        all_stage_ids: List[List[int]] = [s.parent_stage_ids + [s.stage_id] for s in stages.values()]
        all_stage_ids: List[int] = sorted(list(set(sum(all_stage_ids, []))))
        edge_index_list: List[Tuple[int, int]] = []
        root_nodes: List[int] = []
        for stage_id in sorted(stages.keys(), key=lambda key: int(key)):
            for parent_stage_id in stages.get(stage_id).parent_stage_ids:
                edge_index_list.append((all_stage_ids.index(parent_stage_id), all_stage_ids.index(int(stage_id))))
            if len(stages.get(stage_id).parent_stage_ids) == 0:
                root_nodes.append(all_stage_ids.index(int(stage_id)))

        return torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(), len(all_stage_ids), root_nodes

    @staticmethod
    def __create_metrics_vector__(stage: StageDataModel) -> torch.Tensor:
        return torch.tensor(list(stage.metrics.dict().values())).reshape(1, -1)

    @staticmethod
    def __create_context_tensor__(stage: StageDataModel, handler: TransformationHandler) -> torch.Tensor:
        lookup_key: str = f"stages.{stage.stage_id}."
        corresponding_keys: List[str] = handler.get_sub_keys_by_key(lookup_key)
        # concat required values
        stage_vals: List[Tuple[str, Any]] = sum([handler.get_values(col) for col in corresponding_keys], [])
        stage_vals: List[Any] = [el[1] for el in stage_vals]
        stage_context = torch.tensor([])
        if len(stage_vals):
            stage_context = torch.cat(stage_vals, 0).reshape(len(stage_vals), -1)
        return stage_context

    def __call__(self, handler: TransformationHandler):
        stages: Dict[str, StageDataModel] = handler.collapse(handler.get_values("stages"))

        edge_index, num_nodes, root_nodes = StageGraphCreator.__create_edge_index__(stages)
        handler.set_dict_value("edge_index", edge_index)
        handler.set_dict_value("num_nodes", num_nodes)
        handler.set_dict_value("root_nodes", root_nodes)

        stage_runtime_list: List[float] = []
        stage_rescaling_time_ratio_list: List[float] = []
        stage_context_list: List[torch.Tensor] = []
        stage_context_batch_list: List[torch.Tensor] = []
        stage_metrics_list: List[torch.Tensor] = []
        for idx, stage_id in enumerate(sorted(stages.keys(), key=lambda key: int(key))):
            stage_rescaling_time_ratio_list.append(stages.get(stage_id).rescaling_time_ratio)

            runtime: float = 0
            if stages.get(stage_id).start_time is not None and stages.get(stage_id).end_time is not None:
                runtime = (stages.get(stage_id).end_time - stages.get(stage_id).start_time).total_seconds()

            stage_runtime_list.append(runtime)
            stage_context_list.append(StageGraphCreator.__create_context_tensor__(stages.get(stage_id), handler))
            stage_context_batch_list.append(torch.tensor([idx] * len(stage_context_list[-1]), dtype=torch.long))
            stage_metrics_list.append(StageGraphCreator.__create_metrics_vector__(stages.get(stage_id)))
        handler.set_dict_value("stage_rescaling_time_ratio",
                               torch.tensor(stage_rescaling_time_ratio_list).reshape(-1, 1))
        handler.set_dict_value("stage_runtime", torch.tensor(stage_runtime_list).reshape(-1, 1))
        handler.set_dict_value("stage_context", torch.cat(stage_context_list, dim=0))
        handler.set_dict_value("stage_context_batch", torch.cat(stage_context_batch_list, dim=0))
        handler.set_dict_value("stage_metrics", torch.cat(stage_metrics_list, dim=0))
        return handler


class SubDataHolder(object):
    def __init__(self, handler: TransformationHandler):
        self.edge_index: torch.Tensor = handler.collapse(handler.get_values("edge_index")).clone()
        self.num_nodes: int = handler.collapse(handler.get_values("num_nodes"))
        self.root_nodes: List[int] = handler.collapse(handler.get_values("root_nodes"))

        self.stage_runtime: torch.Tensor = handler.collapse(handler.get_values("stage_runtime")).clone()
        self.stage_rescaling_time_ratio: torch.Tensor = handler.collapse(handler.get_values(
            "stage_rescaling_time_ratio")).clone()
        self.stage_context: torch.Tensor = handler.collapse(handler.get_values("stage_context")).clone()
        self.stage_context_batch: torch.Tensor = handler.collapse(handler.get_values("stage_context_batch")).clone()
        self.stage_metrics: torch.Tensor = handler.collapse(handler.get_values("stage_metrics")).clone()
        self.stage_start_scale_out_vec: torch.Tensor = handler.collapse(
            handler.get_values("stage_start_scale_out_vec")).clone()
        self.stage_end_scale_out_vec: torch.Tensor = handler.collapse(
            handler.get_values("stage_end_scale_out_vec")).clone()
        self.job_start_scale_out_vec: torch.Tensor = handler.collapse(
            handler.get_values("job_start_scale_out_vec")).clone()
        self.job_end_scale_out_vec: torch.Tensor = handler.collapse(
            handler.get_values("job_end_scale_out_vec")).clone()
        self.job_rescaling_time_ratio: float = handler.collapse(
            handler.get_values("rescaling_time_ratio"))

        self.__handler__ = handler

    def __global_specs_string__(self):
        specs: GlobalSpecsModel = self.__handler__.__internal_model__.global_specs
        return ','.join([f'{k}={v}' for k, v in specs.dict(exclude_none=True, exclude_defaults=True).items()])

    def __optional_specs_string__(self):
        specs: OptionalSpecsModel = self.__handler__.__internal_model__.optional_specs
        return ','.join([f'{k}={v}' for k, v in specs.dict(exclude_none=True, exclude_defaults=True).items()])

    def __app_repr__(self):
        return f"global_specs={self.__global_specs_string__()} & " \
               f"optional_specs={self.__optional_specs_string__()} & " \
               f"app_signature={self.__handler__.__internal_model__.application_signature}"

    def app_id_repr(self):
        return f"{self.__app_repr__()} & app_id={self.__handler__.__internal_model__.application_id}"

    def job_id_repr(self, job_id: Optional[int] = None):
        job_id: int = job_id if job_id is not None else self.__handler__.__internal_model__.job_id
        return f"{self.__app_repr__()} & job_id={job_id}"

    def full_repr(self):
        return str(self.__handler__.__internal_model__.id)


class NeighborScanner(BaseTransformer):
    def __init__(self):
        self.application_dict: Dict[str, Dict[str, SubDataHolder]] = {}
        self.job_dict: Dict[str, Dict[str, SubDataHolder]] = {}

        self.inference_applications: Set[str] = set()

    def fit(self, handler: TransformationHandler, suffix: str = "training"):
        holder: SubDataHolder = SubDataHolder(handler)
        app_id_str: str = holder.app_id_repr()
        job_id_str: str = holder.job_id_repr()

        app_entry: Dict[str, SubDataHolder] = self.application_dict.get(app_id_str, {})
        if job_id_str not in app_entry:
            app_entry[job_id_str] = holder
        self.application_dict[app_id_str] = app_entry

        job_entry: Dict[str, SubDataHolder] = self.job_dict.get(job_id_str, {})
        if app_id_str not in job_entry:
            job_entry[app_id_str] = holder
        self.job_dict[job_id_str] = job_entry

        if suffix == "inference":
            self.inference_applications.add(app_id_str)

        return handler

    @staticmethod
    def filter_neighbors(target_holder: SubDataHolder, related_prev_holders: List[SubDataHolder]):

        if not len(related_prev_holders):
            return []

        def compute_from_holder(holder: SubDataHolder) -> float:
            return torch.mean(torch.cat([
                holder.job_start_scale_out_vec[:1, 0],
                holder.job_end_scale_out_vec[:1, 0]
            ], dim=-1)).item()

        target_value: float = compute_from_holder(target_holder)
        related_values: List[float] = [compute_from_holder(rph) for rph in related_prev_holders]
        partially_sorted_indices = np.argpartition(np.abs(target_value - np.array(related_values)),
                                                   min(len(related_prev_holders) - 1,
                                                       prediction_settings.prediction_num_neighbors))

        best_indices = partially_sorted_indices[:prediction_settings.prediction_num_neighbors]
        filtered_related_prev_holders = [related_prev_holders[i] for i in best_indices]

        print("Related Prev-Holders:",
              f"Before={len(related_prev_holders)}",
              f"After={len(filtered_related_prev_holders)}",
              f"Filter-Indices={best_indices}")

        return filtered_related_prev_holders

    def __call__(self, handler: TransformationHandler):
        # variable definitions
        prev_metrics: torch.Tensor
        prev_stage_context: torch.Tensor
        prev_job_rescaling_time_ratio: torch.Tensor
        prev_job_start_scale_out_vec: torch.Tensor
        prev_job_end_scale_out_vec: torch.Tensor
        related_prev_metrics: torch.Tensor
        related_prev_stage_context: torch.Tensor
        related_prev_job_rescaling_time_ratio: torch.Tensor
        related_prev_job_start_scale_out_vec: torch.Tensor
        related_prev_job_end_scale_out_vec: torch.Tensor

        holder: SubDataHolder = SubDataHolder(handler)
        app_id_str: str = holder.app_id_repr()
        job_id_str: str = holder.job_id_repr(handler.__internal_model__.job_id - 1)

        prev_holder: Optional[SubDataHolder] = self.application_dict.get(app_id_str, {}).get(job_id_str, None)
        prev_holder_not_none: bool = prev_holder is not None

        related_prev_holders: List[SubDataHolder] = list(self.job_dict.get(job_id_str, {}).values())
        if prev_holder_not_none:
            related_prev_holders = [el for el in related_prev_holders if el.full_repr() != prev_holder.full_repr()]
            prev_metrics = torch.mean(prev_holder.stage_metrics, dim=0, keepdim=True)
            prev_stage_context = prev_holder.stage_context
            prev_job_rescaling_time_ratio = torch.tensor([[prev_holder.job_rescaling_time_ratio]])
            prev_job_start_scale_out_vec = prev_holder.job_start_scale_out_vec
            prev_job_end_scale_out_vec = prev_holder.job_end_scale_out_vec
        else:
            prev_metrics = torch.zeros_like(holder.stage_metrics).mean(dim=0, keepdim=True)
            prev_stage_context = torch.zeros_like(holder.stage_context)
            prev_job_rescaling_time_ratio = torch.tensor([[0.0]])
            prev_job_start_scale_out_vec = torch.zeros_like(holder.job_start_scale_out_vec)
            prev_job_end_scale_out_vec = torch.zeros_like(holder.job_end_scale_out_vec)

        # if inference application: only use data from test data
        if holder.app_id_repr() in self.inference_applications:
            related_prev_holders = [el for el in related_prev_holders
                                    if el.app_id_repr() in self.inference_applications]
        # if data from training. Only use such data
        else:
            related_prev_holders = [el for el in related_prev_holders
                                    if el.app_id_repr() not in self.inference_applications]

        # do not consider all neighbors
        if prev_holder_not_none:
            related_prev_holders = NeighborScanner.filter_neighbors(prev_holder, related_prev_holders)
        else:  # in this case, use scale-out from current job
            related_prev_holders = NeighborScanner.filter_neighbors(holder, related_prev_holders)
        print("Direct Prev-Holder not None:", prev_holder_not_none, "\n")

        if len(related_prev_holders):
            related_prev_metrics = torch.mean(torch.cat([el.stage_metrics for el in related_prev_holders], dim=0),
                                              dim=0, keepdim=True)
            related_prev_stage_context = torch.mean(torch.stack([el.stage_context for el in related_prev_holders],
                                                                dim=0), dim=0)
            related_prev_job_rescaling_time_ratio = torch.mean(
                torch.tensor([el.job_rescaling_time_ratio for el in related_prev_holders])).reshape(-1, 1)
            related_prev_job_start_scale_out_vec = torch.mean(
                torch.cat([el.job_start_scale_out_vec for el in related_prev_holders], dim=0),
                dim=0, keepdim=True)
            related_prev_job_end_scale_out_vec = torch.mean(
                torch.cat([el.job_end_scale_out_vec for el in related_prev_holders], dim=0),
                dim=0, keepdim=True)
        else:
            related_prev_metrics = torch.zeros_like(holder.stage_metrics).mean(dim=0, keepdim=True)
            related_prev_stage_context = torch.zeros_like(holder.stage_context)
            related_prev_job_rescaling_time_ratio = torch.tensor([[0.0]])
            related_prev_job_start_scale_out_vec = torch.zeros_like(holder.job_start_scale_out_vec)
            related_prev_job_end_scale_out_vec = torch.zeros_like(holder.job_end_scale_out_vec)

        # BELOW: update values nested in handler
        handler.set_dict_value("prev_job_batch", torch.tensor([0], dtype=torch.long))
        handler.set_dict_value("real_nodes_batch", torch.tensor(list(range(2,
                                                                           2 + holder.num_nodes)),
                                                                dtype=torch.long))
        new_edge_index: torch.Tensor = torch.tensor(list(product([0, 1], [el + 2 for el in holder.root_nodes])),
                                                    dtype=torch.long).t().contiguous()
        handler.set_dict_value("edge_index", torch.cat([new_edge_index, holder.edge_index + 2],
                                                       dim=-1))
        handler.set_dict_value("num_nodes", holder.num_nodes + 2)
        handler.set_dict_value("stage_runtime", torch.cat([torch.tensor([[0] for _ in range(2)]),
                                                           holder.stage_runtime], dim=0))
        handler.set_dict_value("stage_context_batch", torch.cat([
            torch.tensor([0] * len(prev_stage_context) + [1] * len(related_prev_stage_context), dtype=torch.long),
            holder.stage_context_batch + 2
        ], dim=-1))
        handler.set_dict_value("stage_rescaling_time_ratio", torch.cat([
            prev_job_rescaling_time_ratio,
            related_prev_job_rescaling_time_ratio,
            holder.stage_rescaling_time_ratio
        ], dim=0))
        handler.set_dict_value("stage_context", torch.cat([
            prev_stage_context,
            related_prev_stage_context,
            holder.stage_context
        ], dim=0))
        handler.set_dict_value("stage_metrics", torch.cat([
            prev_metrics,
            related_prev_metrics,
            holder.stage_metrics
        ], dim=0))
        handler.set_dict_value("stage_start_scale_out_vec", torch.cat([
            prev_job_start_scale_out_vec,
            related_prev_job_start_scale_out_vec,
            holder.stage_start_scale_out_vec
        ], dim=0))
        handler.set_dict_value("stage_end_scale_out_vec", torch.cat([
            prev_job_end_scale_out_vec,
            related_prev_job_end_scale_out_vec,
            holder.stage_end_scale_out_vec
        ], dim=0))

        return handler


class OnlinePredictorFinalizer(BaseTransformer):
    def __call__(self, handler: TransformationHandler):
        start_time: Optional[datetime.datetime] = handler.collapse(handler.get_values("start_time"))
        end_time: Optional[datetime.datetime] = handler.collapse(handler.get_values("end_time"))
        if start_time is not None and end_time is not None:
            handler.set_dict_value("job_runtime", (end_time - start_time).total_seconds())
        else:
            handler.set_dict_value("job_runtime", 0)

        handler.set_dict_value("job_rescaling_time_ratio",
                               handler.collapse(handler.get_values("rescaling_time_ratio")))

        job_context_emb: torch.Tensor = handler.collapse(handler.get_values("context_emb"))
        job_context_opt: torch.Tensor = handler.collapse(handler.get_values("context_opt"))

        # repeat so that each real node has the context at hand
        num_nodes: int = handler.collapse(handler.get_values("num_nodes"))
        job_context_emb = job_context_emb.repeat(num_nodes, 1)
        job_context_opt = job_context_opt.repeat(num_nodes, 1)
        handler.set_dict_value("context_emb", job_context_emb)
        handler.set_dict_value("context_opt", job_context_opt)

        # set batch
        emb_batch_list = sum([[i] * int(len(job_context_emb) / num_nodes) for i in range(num_nodes)], [])
        handler.set_dict_value("context_emb_batch",
                               torch.tensor(emb_batch_list, dtype=torch.long))

        opt_batch_list = sum([[i] * int(len(job_context_opt) / num_nodes) for i in range(num_nodes)], [])
        handler.set_dict_value("context_opt_batch",
                               torch.tensor(opt_batch_list, dtype=torch.long))
        return handler


# keep this one updated
# noinspection SpellCheckingInspection
transformer_map: dict = {
    "HashingVectorizer": HashingVectorizer,
    "ContextEmbedder": ContextEmbedder,
    "MinMaxScaler": MinMaxScaler,
    "BinaryTransformer": BinaryTransformer,
    "ScaleOutEnricher": ScaleOutEnricher,
    "StageGraphCreator": StageGraphCreator,
    "NeighborScanner": NeighborScanner,
    "OnlinePredictorFinalizer": OnlinePredictorFinalizer
}


class CustomCompose(object):
    """Composes several transforms together."""

    def __init__(self, *args, **kwargs):
        # define fields and their types
        self.transformer_specs: Optional[list] = None
        # use kwargs to set fields (maybe not all)
        kwargs.setdefault('transformer_specs', [])
        self.__dict__.update(kwargs)

        self.transforms: List[BaseTransformer] = self._get_transforms()
        self.__is_fitted__: bool = False

    @staticmethod
    def from_ref(ref: CustomCompose):
        return CustomCompose(transformer_specs=ref.transformer_specs)

    def _get_transforms(self):
        """Retrieves a list of transforms."""

        transformer_list: List[BaseTransformer] = []
        for idx, transformer_spec in enumerate(self.transformer_specs):
            transformer_class = transformer_map.get(transformer_spec["transformer_class"])
            transformer_obj = transformer_class(**transformer_spec.get("transformer_args", {}))
            transformer_list.append(transformer_obj)

        return transformer_list

    def fit(self, data_list: List[TransformationHandler], suffix="inference") -> None:
        if self.__is_fitted__:
            if all([t.__class__.__name__ != "NeighborScanner" for t in self.transforms]):
                return
            else:
                input_data_list = copy.deepcopy(data_list)
                for data in input_data_list:
                    for t in self.transforms:
                        if t.__class__.__name__ != "NeighborScanner":
                            data = t(data)
                        else:
                            data = t.fit(data, suffix=suffix)
                return

        fit_indices = [i for i, t in enumerate(self.transforms) if "fit" in t.__class__.__dict__.keys()]
        for ft in fit_indices:
            input_data_list = copy.deepcopy(data_list)
            for data in input_data_list:
                for i, t in enumerate(self.transforms):
                    if i < ft:
                        data = t(data)
                    elif i == ft:
                        if t.__class__.__name__ != "NeighborScanner":
                            data = t.fit(data)
                        else:
                            data = t.fit(data, suffix=suffix)
                    elif i > ft:
                        continue
        self.__is_fitted__ = True

    def __call__(self, handler: TransformationHandler) -> CustomData:
        for t in self.transforms:
            handler = t(handler)
        return handler.to_data_object()

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))
