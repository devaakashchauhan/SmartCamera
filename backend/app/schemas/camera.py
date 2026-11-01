from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class CameraBase(BaseModel):
    name: str
    source: str = "0"
    location: str = "Main Area"
    status: str = "ONLINE"


class CameraCreate(CameraBase):
    pass


class CameraResponse(CameraBase):
    id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
