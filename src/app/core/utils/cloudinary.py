import os

import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)


def upload_image(file, folder: str):
    result = cloudinary.uploader.upload(file, folder=folder)

    return {
        "url": result["secure_url"],
        "public_id": result["public_id"],
        "bytes": result["bytes"],
        "format": result["format"],
    }


def delete_image(public_id: str):
    cloudinary.uploader.destroy(public_id)
