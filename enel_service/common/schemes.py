from typing import Optional
from pydantic import BaseModel


class DefaultResponse(BaseModel):
    message: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "message": ""
            }
        }
