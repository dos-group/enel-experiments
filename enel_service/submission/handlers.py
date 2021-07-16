import logging
import os

from bson import ObjectId
from fastapi import HTTPException, status

from enel_service.common.apis.fs_api import FsApi
from enel_service.common.apis.hdfs_api import HdfsApi
from enel_service.common.apis.kubernetes_api import KubernetesApi, generate_template_code, update_dict_func
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.configuration import GeneralSettings, HdfsSettings, MongoSettings, ListenerSettings
from enel_service.modeling.schemes import OfflineScaleOutPredictionResponse
from enel_service.modeling.transforms import DictCamelCaser
from enel_service.modeling.handlers_scale_out_initial import handle_initial_scale_out_prediction
from .schemes import *
from .ellis_utils import handle_ellis_initial_scale_out_prediction

hdfs_settings: HdfsSettings = HdfsSettings.get_instance()
general_settings: GeneralSettings = GeneralSettings.get_instance()
mongo_settings: MongoSettings = MongoSettings.get_instance()
listener_settings: ListenerSettings = ListenerSettings.get_instance()


def extend_spark_config(document: ApplicationExecutionModel):
    # enrich with enel_service information
    if document.global_specs.system_name == "spark":
        # prepare and retrieve default config (mostly from environment variables)
        default_config: dict = prepare_default_config(document.id, document.start_scale_out)
        # update default config with values in request object
        document.spark_template_values["scale_out_tuner"] = \
            update_dict_func(default_config,
                             document.spark_template_values.get("scale_out_tuner", {}),
                             check_existence=False)
    return document


async def alter_submission_model(document: ApplicationExecutionModel, hdfs_api: HdfsApi, mongo_api: MongoApi):
    if document.global_specs.solution_name == "enel" and \
            document.spark_template_values.get("scale_out_tuner", {}).get("config", {}).get("is_adaptive", False):
        logging.info("Get initial scale-out from Enel (with help of Bell)...")
        response: OfflineScaleOutPredictionResponse = await handle_initial_scale_out_prediction(document,
                                                                                                hdfs_api, mongo_api)
        document = response.prepared_db_entry
    else:
        logging.info("Get initial scale-out from Ellis...")
        document, _ = await handle_ellis_initial_scale_out_prediction(document)
    # set scale out for worker property
    document.worker_specs.scale_out = document.start_scale_out
    # remember scale-out prediction
    document.predicted_scale_out = document.start_scale_out
    # if spark system: possibly enrich config
    document = extend_spark_config(document)

    if document.global_specs.solution_name == "enel":
        # insert into DB
        target_collection: str = mongo_settings.mongodb_application_execution_collection
        application_execution_id: Optional[str] = await mongo_api.insert_one(target_collection,
                                                                             document.dict(by_alias=True))
        if application_execution_id is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Could not insert into DB.")

    return document


def prepare_default_config(application_execution_id: str, initial_executors: int) -> dict:
    config: dict = {
        "method": "enel",
        "rest_timeout": listener_settings.drms_rest_timeout,
        "service": listener_settings.drms_service,
        "port": general_settings.port,
        "online_scale_out_prediction_endpoint": listener_settings.drms_online_scale_out_prediction_endpoint,
        "update_information_endpoint": listener_settings.drms_update_information_endpoint,
        "application_execution_id": application_execution_id,
        "is_adaptive": False,
        "initial_executors": initial_executors
    }
    tuner: dict = {
        "config": config
    }

    return tuner


async def handle_submit_application(request: ApplicationSubmissionRequest,
                                    kubernetes_api: KubernetesApi,
                                    fs_api: FsApi,
                                    hdfs_api: HdfsApi,
                                    mongo_api: MongoApi):
    # unpack request
    application_name: str = f"{request.global_specs.system_name}-app-{request.global_specs.template_version}"

    # prepare response
    response_obj: dict = {}

    source_folder = os.path.join(general_settings.config_dir, "job_templates", application_name)
    # load default values associated with desired job template
    source_values_file = os.path.join(source_folder, "values.yaml")
    source_values, _ = fs_api.load(source_values_file)

    target_values_file: Optional[str] = ""
    target_template_file: Optional[str] = ""

    if len(source_values):

        new_document: ApplicationExecutionModel = ApplicationExecutionModel(**{
            **request.dict(),
            "_id": str(ObjectId()),
            "start_scale_out": request.worker_specs.scale_out,  # will be overridden eventually
            "end_scale_out": request.worker_specs.scale_out  # will be overridden eventually
        })

        new_document = await alter_submission_model(new_document, hdfs_api, mongo_api)

        response_obj["db_entry"] = new_document.dict()

        # update values yaml with values from payload
        template_values = DictCamelCaser()(new_document.dict())
        source_values = [update_dict_func(source_values[0], template_values)]

        # save locally for further use
        target_values_file = os.path.join(general_settings.temp_dir, f"{new_document.id}_values.yaml")
        error = fs_api.save(source_values, target_values_file)

        if not error and target_values_file:
            # generate the template code
            release_name = request.spark_template_values["release_name"]
            template_code, error = generate_template_code(source_folder, target_values_file, release_name=release_name)
            if template_code:
                response_obj["template_code"] = template_code
                # save locally for further use
                target_template_file = os.path.join(general_settings.temp_dir, f"{new_document.id}_defs.yaml")
                error = fs_api.save(template_code, target_template_file)

                if not error and target_template_file:
                    api_response, error = kubernetes_api.post(target_template_file)
                    response_obj["k8s_api_response"] = api_response

    # clean up
    try:
        os.path.exists(target_values_file) and os.remove(target_values_file)
        os.path.exists(target_template_file) and os.remove(target_template_file)
    except BaseException as e:
        logging.error(e)
    if not isinstance(response_obj.get("k8s_api_response", {}), dict):
        response_obj["k8s_api_response"] = {"response": response_obj["k8s_api_response"]}
    return response_obj
