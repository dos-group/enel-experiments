import copy
import logging
from functools import partial
from typing import List, Any, Type, Tuple

import pymongo
from fastapi import HTTPException, status

from enel_service.common.apis.hdfs_api import HdfsApi
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.configuration import GeneralSettings, MongoSettings
from enel_service.config.base_model_config import BaseModelConfig
from enel_service.config.onlinepredictor_config import OnlinePredictorConfig
from enel_service.modeling.datasets import ExecutionDataset, ExecutionSubset
from enel_service.modeling.hyper_optimization import HyperOptimizer
from enel_service.modeling.model_wrappers import OnlinePredictorModel
from enel_service.modeling.schemes import TriggerModelTrainingRequest
from enel_service.modeling.transforms import CustomCompose
from enel_service.modeling.utils import save_artifact

# load settings
general_settings: GeneralSettings = GeneralSettings.get_instance()
mongo_settings: MongoSettings = MongoSettings.get_instance()
# load model configs
onlinepredictor_config: OnlinePredictorConfig = OnlinePredictorConfig()


async def handle_trigger_model_training(request: TriggerModelTrainingRequest,
                                        mongo_api: MongoApi,
                                        hdfs_api: HdfsApi):
    # PREPARATION
    # extract request parameters
    system_name: str = request.system_name
    algorithm_name: str = request.algorithm_name
    model_name: str = request.model_name

    key_prefix: str = f"{system_name}_{algorithm_name}_{model_name}"

    shallow_model: Type[OnlinePredictorModel]
    config: BaseModelConfig
    mongo_query: dict
    historical_data: List[Any]
    target_collection: str

    # retrieve model, config, and data
    if model_name == "onlinepredictor":
        shallow_model = OnlinePredictorModel
        config = onlinepredictor_config
        target_collection = mongo_settings.mongodb_job_execution_collection
    else:
        logging.error("Invalid Model-Type specified. Must be one of ['onlinepredictor'].")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid Model-Type specified. Must be one of ['onlinepredictor'].")

    mongo_query: dict = {
        "filter": {
            'start_time': {'$exists': True, '$ne': None},
            'end_time': {'$exists': True, '$ne': None},
            "global_specs.system_name": system_name,
            "global_specs.algorithm_name": algorithm_name,
            "global_specs.experiment_name": request.experiment_name
        }
    }
    if model_name == "onlinepredictor":
        mongo_query["sort"] = [("application_id", pymongo.ASCENDING), ("job_id", pymongo.ASCENDING)]

    # init model
    shallow_model: OnlinePredictorModel = shallow_model(config)
    # create transformer
    data_transformer: CustomCompose = CustomCompose(transformer_specs=config.transformer_specs)

    try:
        historical_dataset: ExecutionDataset = await ExecutionDataset. \
            from_config("application" if model_name == "offlinepredictor" else "job",
                        mongo_api,
                        target_collection,
                        mongo_query,
                        data_transformer,
                        suffix="training")
    except ValueError:
        logging.error("No data for pretraining available.")
        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED,
                            detail="No data for pretraining available.")

    train_and_save(shallow_model, algorithm_name, historical_dataset,
                   config,
                   key_prefix,
                   hdfs_api)


def train_and_save(shallow_model: OnlinePredictorModel,
                   algorithm_name: str,
                   historical_dataset: ExecutionDataset,
                   config: BaseModelConfig,
                   key_prefix: str,
                   hdfs_api: HdfsApi,):
    send_artifact = partial(save_artifact, hdfs_api)

    main_data_transformer: CustomCompose = copy.deepcopy(historical_dataset.data_transformer)

    historical_subsets: Tuple[ExecutionSubset, ExecutionSubset] = shallow_model.split_data(historical_dataset)

    # TRAINING
    # prepare training / hyperparameter optimization
    hyperoptimizer_instance = HyperOptimizer(**shallow_model.__dict__,
                                             pre_augmentation_function=shallow_model.pre_augmentation_function,
                                             post_augmentation_function=shallow_model.post_augmentation_function,
                                             output_transform_function=shallow_model.output_transform_function,
                                             job_identifier=algorithm_name)

    logging.info("Start training / hyperparameter optimization...")
    checkpoint = HyperOptimizer.perform_optimization(hyperoptimizer_instance,
                                                     hyperoptimizer_instance.epochs[0],
                                                     historical_subsets,
                                                     config=config)

    # save checkpoint dict
    logging.info("Save best checkpoint...")
    send_artifact(checkpoint, f"{key_prefix}_checkpoint.pt")

    # save shallow model
    logging.info("Save shallow model...")
    shallow_model.incorporate_checkpoint(checkpoint)
    send_artifact(shallow_model, f"{key_prefix}_model_wrapper.pt")

    # save fitted pipeline
    logging.info("Save data-transformer...")
    send_artifact(main_data_transformer, f"{key_prefix}_data_transformer.pt")

    # create torchscript model
    logging.info("Create torchscript model...")
    torchscript_model = shallow_model.get_torchscript_model_instance(checkpoint,
                                                                     log_meta_information=True,
                                                                     disable_autoencoder=True)

    # save torchscript model
    logging.info("Save torchscript model...")
    send_artifact(torchscript_model, f"{key_prefix}_torchscript_model.pt")
