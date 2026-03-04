import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .admin.initialize import create_admin_interface
from .api import router
from .core.config import settings
from .core.ml.predict import load_ml_model
from .core.setup import create_application, lifespan_factory

admin = create_admin_interface()


@asynccontextmanager
async def lifespan_with_admin(app: FastAPI) -> AsyncGenerator[None, None]:
    """Custom lifespan that includes admin initialization."""
    # Get the default lifespan
    default_lifespan = lifespan_factory(settings)

    # Run the default lifespan initialization and our admin initialization
    async with default_lifespan(app):
        # 🔥 Load ML model here
        # print("🚀 Loading ML model at startup...")
        # success = load_ml_model()
        # if success:
        #     print("✅ ML Model loaded successfully")
        # else:
        #     print("❌ ML Model failed to load")
        # Initialize admin interface if it exists
        if admin:
            # Initialize admin database and setup
            await admin.initialize()

        yield


app = create_application(router=router, settings=settings, lifespan=lifespan_with_admin)

media_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media")
os.makedirs(media_path, exist_ok=True)
app.mount("/media", StaticFiles(directory=media_path), name="media")

# Mount admin interface if enabled
if admin:
    app.mount(settings.CRUD_ADMIN_MOUNT_PATH, admin.app)
