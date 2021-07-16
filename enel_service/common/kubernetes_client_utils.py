import logging
from typing import Optional, TypeVar, Generic

import kubernetes
from pydantic import BaseModel

T = TypeVar('T')


class KubernetesApiResponse(BaseModel, Generic[T]):
    success: bool = True
    exception: Optional[Exception]
    result: Optional[T]

    class Config:
        arbitrary_types_allowed = True


class KubernetesApiHelper:
    @staticmethod
    def call_k8s_api(fn, raise_exception: bool = False, **kwargs) -> KubernetesApiResponse:
        """
        call k8s api and handling exceptions
        :param fn: function to call
        :param raise_exception: whether raise exception if exceptions encountered
        :param kwargs:
        :return:
        """

        api_response = KubernetesApiResponse()
        try:
            api_response.result = fn(**kwargs)
        except kubernetes.client.exceptions.ApiException as exc:
            if raise_exception:
                raise exc
            logging.error("Kubernetes api call exception,", exc_info=exc)
            api_response.success = False
            api_response.exception = exc
        except Exception as exc:
            if raise_exception:
                raise exc
            logging.error("Kubernetes api call exception,", exc_info=exc)
            api_response.success = False
            api_response.exception = exc
        return api_response
