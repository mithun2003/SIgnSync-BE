from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from ..core.schemas import CommonResponse, PersistentDeletion, TimestampSchema, UUIDSchema


class UserBase(BaseModel):
    first_name: Annotated[str | None, Field(max_length=50, examples=["Admin"])] = None
    last_name: Annotated[str | None, Field(max_length=50, examples=["User"])] = None
    username: Annotated[
        str,
        Field(min_length=2, max_length=20, pattern=r"^[a-z0-9_]+$", examples=["userson"]),
    ]
    email: Annotated[EmailStr, Field(examples=["userson@example.com"])]


class User(TimestampSchema, UserBase, UUIDSchema, PersistentDeletion):
    profile_image_url: Annotated[str | None, Field(default="https://www.profileimageurl.com")] = None
    hashed_password: str
    is_superuser: bool = False
    tier_id: int | None = None


class UserRead(BaseModel):
    id: int
    first_name: str | None
    last_name: str | None
    username: str
    email: EmailStr
    profile_image_url: str | None
    bio: str | None
    country: str | None
    language: str
    two_factor_enabled: bool
    tier_id: int | None


class UserCreate(UserBase):
    model_config = ConfigDict(extra="forbid")

    password: Annotated[
        str,
        Field(
            pattern=r"^.{8,}|[0-9]+|[A-Z]+|[a-z]+|[^a-zA-Z0-9]+$",
            examples=["Str1ngst!"],
        ),
    ]


class UserCreateInternal(UserBase):
    hashed_password: str


class UserUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: str | None = None
    last_name: str | None = None
    username: str | None = None
    email: EmailStr | None = None
    bio: str | None = None
    country: str | None = None
    language: str | None = None
    two_factor_enabled: bool | None = None
    profile_image_url: str | None = None


class UserUpdateInternal(UserUpdate):
    updated_at: datetime


class UserTierUpdate(BaseModel):
    tier_id: int


class UserDelete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_deleted: bool
    deleted_at: datetime


class UserRestoreDeleted(BaseModel):
    is_deleted: bool


class UserResponse(CommonResponse):
    data: UserRead
