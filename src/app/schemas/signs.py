from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SignBase(BaseModel):
    character: str
    cloudinary_url: str
    notes: str | None = None


class SignCreate(SignBase):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    updated_by: int | None = None


class SignUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cloudinary_url: str | None = None
    notes: str | None = None
    version: int | None = None
    updated_by: int | None = None


class SignRead(BaseModel):
    id: int
    character: str
    cloudinary_url: str
    version: int
    updated_by: int | None
    notes: str | None
    created_at: datetime
    updated_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class UpdateURLRequest(BaseModel):
    cloudinary_url: str
    notes: str | None = None
