from typing import TypeVar, Generic, Optional, Union

from pydantic.generics import GenericModel
from starlette.responses import JSONResponse

T = TypeVar('T')


class ApiResponse(GenericModel, Generic[T]):
    """
    API Response Model
    """
    success: bool = True
    data: Optional[T] = None
    error_msg: Optional[Union[str, dict]] = None


class ResponseHelper(Generic[T]):
    """
    API Response Helper
    """

    def __init__(self):
        self.resp = ApiResponse[T]()

    def succeed(self, data=None) -> T:
        if data is not None:
            self.resp.data = data
        return self.resp

    def fail(self, error_msg) -> T:
        self.resp.success = False
        self.resp.error_msg = error_msg
        return self.resp

    def exception(self, error_msg, status_code) -> JSONResponse:
        self.resp.success = False
        self.resp.error_msg = error_msg
        return JSONResponse(self.resp.dict(), status_code=status_code)
