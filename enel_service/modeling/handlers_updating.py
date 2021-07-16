import logging
from typing import Optional, Union
from bson import ObjectId

from enel_service.common.apis.kubernetes_api import update_dict_func
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.configuration import MongoSettings
from enel_service.common.db_schemes import JobExecutionModel, ApplicationExecutionModel
from enel_service.modeling.schemes import UpdateInformationRequest, RootDataUpdateModel

mongo_settings: MongoSettings = MongoSettings.get_instance()


# update events: ["APPLICATION_START", "JOB_END"]

async def update_application_object(request: UpdateInformationRequest, mongo_api: MongoApi, session):
    target_collection: str = mongo_settings.mongodb_application_execution_collection

    new_app_db_element: Optional[ApplicationExecutionModel] = None
    ack: bool = False

    old_app_db_result: Optional[dict] = await mongo_api.find_one(target_collection,
                                                                 {"_id": request.application_execution_id},
                                                                 catch_error=False,
                                                                 session=session)
    if old_app_db_result is None:
        logging.error(f"Could not find App DB-element with Id '{request.application_execution_id}'.")
        return None, False

    old_app_db_element: ApplicationExecutionModel = ApplicationExecutionModel(**old_app_db_result)
    # update with remote data
    new_app_db_result: dict = update_dict_func(old_app_db_element.dict(),
                                               request.updates.dict(exclude_none=True, exclude_defaults=True),
                                               check_existence=False)
    new_app_db_element = ApplicationExecutionModel(**new_app_db_result)

    if str(old_app_db_element.dict()) != str(new_app_db_element.dict()):
        logging.info(f"Update App DB-element with Id '{request.application_execution_id}'...")
        ack = await mongo_api.update_one(target_collection,
                                         {"_id": request.application_execution_id},
                                         {"$set": new_app_db_element.dict(exclude={"id"}, exclude_none=True)},
                                         catch_error=False,
                                         session=session)
    else:
        ack = True

    return new_app_db_element, ack


async def update_job_object(request: UpdateInformationRequest, mongo_api: MongoApi, session):
    job_collection: str = mongo_settings.mongodb_job_execution_collection
    app_collection: str = mongo_settings.mongodb_application_execution_collection

    new_job_db_element: Optional[JobExecutionModel] = None
    ack: bool = False

    # get parent app
    app_db_result: Optional[dict] = await mongo_api.find_one(app_collection,
                                                             {"_id": request.application_execution_id},
                                                             catch_error=False,
                                                             session=session)
    if app_db_result is None:
        logging.error(f"Could not find App DB-element with "
                      f"Application-Execution-Id '{request.application_execution_id}'.")
        return None, False

    app_db_element: ApplicationExecutionModel = ApplicationExecutionModel(**app_db_result)

    # first: check if we have an entry for this job id
    old_job_db_result: Optional[dict] = await mongo_api.find_one(job_collection, {
        "application_execution_id": request.application_execution_id,
        "application_id": request.application_id,
        "job_id": request.job_id
    }, catch_error=False, session=session)
    create: bool = False
    # else: craft from corresponding app (id)
    if old_job_db_result is None:
        create = True
        old_job_db_result = app_db_element.dict()
        old_job_db_result["job_id"] = request.job_id
        old_job_db_result["application_execution_id"] = request.application_execution_id
        old_job_db_result["_id"] = str(ObjectId())
        for dict_key in ["attempt_id", "fit_time", "predict_time", "preparation_time",
                         "predicted_scale_out", "created_at", "updated_at"]:
            old_job_db_result.pop(dict_key, None)

    old_job_db_element: JobExecutionModel = JobExecutionModel(**old_job_db_result)
    # update with remote data
    new_job_db_result: dict = update_dict_func(old_job_db_element.dict(),
                                               request.updates.dict(exclude_none=True, exclude_defaults=True),
                                               check_existence=False)
    new_job_db_element = JobExecutionModel(**new_job_db_result)

    if str(old_job_db_element.dict()) != str(new_job_db_element.dict()):
        logging.info(f"Update Job DB-element with "
                     f"Application-Execution-Id '{request.application_execution_id}'...")

        _ = await mongo_api.update_one(job_collection, {
            "application_execution_id": request.application_execution_id,
            "application_id": request.application_id,
            "job_id": request.job_id
        }, {"$set": new_job_db_element.dict(by_alias=True, exclude_none=True)},
                                         create=create,
                                         upsert=True, catch_error=False, session=session)

        if app_db_element.end_time is None or \
                new_job_db_element.end_time.replace(tzinfo=None) > app_db_element.end_time.replace(tzinfo=None):
            # use end_time and end_scale_out from this job
            logging.info(f"Update App DB-element with Id '{request.application_execution_id}'...")
            ack = await mongo_api.update_one(app_collection,
                                             {"_id": request.application_execution_id},
                                             {"$set": {
                                                 "end_time": new_job_db_element.end_time,
                                                 "end_scale_out": new_job_db_element.end_scale_out
                                             }},
                                             catch_error=False,
                                             session=session)
        else:
            ack = True
    else:
        ack = True

    return new_job_db_element, ack


async def handle_update_information(request: UpdateInformationRequest, mongo_api: MongoApi):
    # extract infos
    application_execution_id: str = request.application_execution_id
    application_id: str = request.application_id
    job_id: Optional[int] = request.job_id
    update_event: str = request.update_event
    updates: RootDataUpdateModel = request.updates

    logging_prefix: str = f"[Application-Execution-Id: {application_execution_id},  " \
                          f"Application-Id: {application_id}, " \
                          f"{'' if job_id is None else f'Job-Id: {job_id}'}]"

    logging.info(f"{logging_prefix} Event = '{update_event}', "
                 f"Updates = '{updates.dict(exclude_none=True, exclude_defaults=True)}'.")

    db_element: Union[ApplicationExecutionModel, JobExecutionModel, None] = None
    ack: bool = False

    async with await mongo_api.get_client().start_session() as session:
        if update_event == "APPLICATION_START":
            db_element, ack = await session.with_transaction(lambda s: update_application_object(request, mongo_api, s))
        elif update_event == "JOB_END":
            db_element, ack = await session.with_transaction(lambda s: update_job_object(request, mongo_api, s))
        else:
            logging.error(f"Unknown case! Update-Event = {update_event}")
    await mongo_api.close_client()

    if db_element is not None and ack:
        return db_element
