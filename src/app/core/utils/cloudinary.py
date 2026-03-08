import os

import cloudinary
import cloudinary.api
import cloudinary.search
import cloudinary.uploader

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

SIGNS_FOLDER = "asl-signs"

_MIME_MAP: dict[str, str] = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}


def _mime_from_format(fmt: str) -> str:
    return _MIME_MAP.get(fmt.lower(), f"image/{fmt.lower()}")


def upload_sign_image(file, character: str, version: int) -> dict:
    """Upload a file to Cloudinary with an explicit, version-stamped public_id: ``asl-signs/{CHARACTER}/{version}``
    (e.g. ``asl-signs/A/3``).

    The version number must be determined by the caller (typically
    ``len(list_sign_images(character)) + 1``) so that every upload gets a
    permanent, sequential slot that is never overwritten.
    """
    public_id = f"{SIGNS_FOLDER}/{character.upper()}/{version}"
    result = cloudinary.uploader.upload(
        file,
        public_id=public_id,
        folder=f"{SIGNS_FOLDER}/{character.upper()}",
        resource_type="image",
        overwrite=False,
        tags=[f"character_{character.upper()}"],
    )
    return {
        "url": result["secure_url"],
        "public_id": result["public_id"],
        "file_size": result["bytes"],
        "width": result["width"],
        "height": result["height"],
        "mime_type": _mime_from_format(result.get("format", "jpeg")),
    }


def list_sign_images(character: str) -> list[dict]:
    """Return all images stored under asl-signs/{CHARACTER}/ in Cloudinary, sorted by version number ascending.
    Never raises – returns [] on any error.

    For images uploaded with the versioned scheme the ``version`` field is the
    integer extracted from the public_id's last path segment
    (e.g. ``asl-signs/A/3`` → version 3).  Older images whose public_id ends
    with a random string will have ``version=None``.
    """
    prefix = f"{SIGNS_FOLDER}/{character.upper()}/"
    try:
        result = cloudinary.api.resources(
            type="upload",
            prefix=prefix,
            max_results=500,
            resource_type="image",
        )
        out = []
        for r in result.get("resources", []):
            pid = r["public_id"]
            last_segment = pid.split("/")[-1]
            version = int(last_segment) if last_segment.isdigit() else None
            out.append(
                {
                    "public_id": pid,
                    "url": r["secure_url"],
                    "file_size": r.get("bytes", 0),
                    "width": r.get("width", 0),
                    "height": r.get("height", 0),
                    "mime_type": _mime_from_format(r.get("format", "jpeg")),
                    "created_at": r.get("created_at"),
                    "version": version,
                }
            )
        # Sort versioned images first (ascending), then unversioned by created_at
        out.sort(key=lambda x: (x["version"] is None, x["version"] or 0))
        return out
    except Exception:
        return []


def get_image_details(public_id: str) -> dict | None:
    """Fetch metadata for a single Cloudinary asset.

    Returns None on error.
    """
    try:
        r = cloudinary.api.resource(public_id)
        return {
            "url": r["secure_url"],
            "public_id": r["public_id"],
            "file_size": r.get("bytes", 0),
            "width": r.get("width", 0),
            "height": r.get("height", 0),
            "mime_type": _mime_from_format(r.get("format", "jpeg")),
        }
    except Exception:
        return None


def delete_image(public_id: str) -> bool:
    """Delete a single image from Cloudinary by its public_id.

    Returns True if the image was deleted, False if it was not found. Raises on unexpected errors.
    """
    result = cloudinary.uploader.destroy(public_id, resource_type="image")
    return result.get("result") == "ok"


# ── Legacy helper kept for backward compatibility ──────────────────────────────
def upload_image(file, folder: str) -> dict:
    result = cloudinary.uploader.upload(file, folder=folder)
    return {
        "url": result["secure_url"],
        "public_id": result["public_id"],
        "bytes": result["bytes"],
        "format": result["format"],
    }
