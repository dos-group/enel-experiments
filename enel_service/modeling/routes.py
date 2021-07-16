from fastapi import APIRouter, status, BackgroundTasks, Depends

from enel_service.common.schemes import DefaultResponse
from enel_service.common.apis.hdfs_api import HdfsApi
from .handlers_training import *
from .handlers_runtime import *
from .handlers_scale_out import *
from .handlers_updating import handle_update_information
from .schemes import *
from .utils import preload_artifacts

prediction_router = APIRouter(
    prefix="/prediction",
    tags=["prediction"]
)

training_router = APIRouter(
    prefix="/training",
    tags=["training"]
)

prediction_metadata: dict = {
    "name": "prediction",
    "description": "Endpoints for retrieving various predictions."
}

training_metadata: dict = {
    "name": "training",
    "description": "Endpoints for triggering training of models and updating information."
}


@training_router.post('/trigger_model_training',
                      status_code=status.HTTP_200_OK)
async def trigger_model_training(request: TriggerModelTrainingRequest,
                                 mongo_api: MongoApi = Depends(),
                                 hdfs_api: HdfsApi = Depends()):
    return await handle_trigger_model_training(request, mongo_api, hdfs_api)


@training_router.post('/update_information',
                      response_model=DefaultResponse,
                      status_code=status.HTTP_200_OK)
async def update_information(request: UpdateInformationRequest,
                             background_tasks: BackgroundTasks,
                             mongo_api: MongoApi = Depends()):
    background_tasks.add_task(handle_update_information, request, mongo_api)
    return DefaultResponse(message="Added the update of information to the list of background tasks.")


@prediction_router.get('/preload/{base_name}',
                       status_code=status.HTTP_200_OK)
async def preload(base_name: str, hdfs_api: HdfsApi = Depends()):
    return await preload_artifacts(base_name, hdfs_api)


@prediction_router.post('/online_scale_out_prediction',
                        response_model=OnlineScaleOutPredictionResponse,
                        status_code=status.HTTP_200_OK)
async def online_scale_out_prediction(request: OnlineScaleOutPredictionRequest,
                                      background_tasks: BackgroundTasks,
                                      mongo_api: MongoApi = Depends(),
                                      hdfs_api: HdfsApi = Depends()):
    return await handle_online_scale_out_prediction(request, background_tasks, hdfs_api, mongo_api)
