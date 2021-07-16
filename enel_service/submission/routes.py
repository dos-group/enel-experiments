from fastapi import APIRouter, status, Depends

from .handlers import *
from .schemes import *
from enel_service.common.apis.fs_api import FsApi
from enel_service.common.apis.mongo_api import MongoApi
from enel_service.common.apis.kubernetes_api import KubernetesApi

router = APIRouter(
    prefix="/submission",
    tags=["submission"]
)

metadata: dict = {
    "name": "submission",
    "description": "Endpoints for submitting spark-applications and fetching information."
}


@router.post("/submit",
             response_model=ApplicationSubmissionResponse,
             status_code=status.HTTP_200_OK)
async def submit_application(request: ApplicationSubmissionRequest,
                             fs_api: FsApi = Depends(),
                             hdfs_api: HdfsApi = Depends(),
                             mongo_api: MongoApi = Depends(),
                             kubernetes_api: KubernetesApi = Depends()):
    return await handle_submit_application(request, kubernetes_api, fs_api, hdfs_api, mongo_api)
