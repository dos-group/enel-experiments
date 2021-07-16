import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from requests.exceptions import ConnectionError as InternalConnectionError
from starlette import status

from .response_utils import ResponseHelper


async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    request validation exception handler
    :param request:
    :param exc:
    :return:
    """
    logging.error(f"Request parameters validation error for request {request.url}", exc_info=exc)
    error_msg = {"error_type": "request parameter error", "detail": exc.errors(), "body": exc.body}
    return ResponseHelper().exception(error_msg=error_msg, status_code=status.HTTP_400_BAD_REQUEST)


async def internal_connection_error_handler(request: Request, exc: InternalConnectionError):
    """
    internal connection error handler
    :param request:
    :param exc:
    :return:
    """
    logging.error(f"Internal connection error for request {request.url}", exc_info=exc)
    error_msg = f"internal connection error {str(exc)}"
    return ResponseHelper().exception(error_msg=error_msg, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


async def base_exception_handler(request: Request, exc: Exception):
    """
    base exception handler
    :param request:
    :param exc:
    :return:
    """
    logging.error(f"Internal  error for request {request.url}", exc_info=exc)
    error_msg = {
        "error_msg": f"internal error",
        "exception type": str(type(exc)),
        "exception_detail": str(exc)
    }
    return ResponseHelper().exception(error_msg, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


def register_exception_handler(server):
    """
    register exceptin handler
    :param server:
    :return:
    """
    server.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    server.add_exception_handler(InternalConnectionError, internal_connection_error_handler)
    server.add_exception_handler(Exception, base_exception_handler)
