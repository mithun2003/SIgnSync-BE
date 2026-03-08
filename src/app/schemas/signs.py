from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SignBase(BaseModel):
    character: str
    cloudinary_url: str
    notes: str | None = None


class SignCreate(SignBase):
    model_config = ConfigDict(extra="forbid")

    cloudinary_public_id: str | None = None
    file_size: int | None = None
    width: int | None = None
    height: int | None = None
    mime_type: str | None = None
    version: int = 1
    updated_by: int | None = None


class SignUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cloudinary_url: str | None = None
    cloudinary_public_id: str | None = None
    file_size: int | None = None
    width: int | None = None
    height: int | None = None
    mime_type: str | None = None
    notes: str | None = None
    version: int | None = None
    updated_by: int | None = None


class SignRead(BaseModel):
    id: int
    character: str
    cloudinary_url: str
    cloudinary_public_id: str | None = None
    file_size: int | None = None
    width: int | None = None
    height: int | None = None
    mime_type: str | None = None
    version: int
    updated_by: int | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


# ── Request bodies ─────────────────────────────────────────────────────────────


class UpdateURLRequest(BaseModel):
    """Manually set the active URL (no file upload)."""

    cloudinary_url: str
    notes: str | None = None


class SetActiveImageRequest(BaseModel):
    """Choose an existing Cloudinary image (already in the folder) as the active one."""

    cloudinary_public_id: str
    notes: str | None = None


# ── Cloudinary image info (for listing versions) ───────────────────────────────


class CloudinaryImageInfo(BaseModel):
    """Represents one image version stored in Cloudinary."""

    public_id: str
    url: str
    file_size: int
    width: int
    height: int
    mime_type: str
    created_at: str | None = None
    version: int | None = None  # sequential upload version; None for legacy random-id images


class BulkUploadResult(BaseModel):
    """Result summary for a bulk-upload operation."""

    uploaded: list[dict]
    skipped: list[dict]  # files whose name could not be mapped to a valid character
