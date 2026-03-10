import asyncio
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
        # Load MLP model in a thread pool so it doesn't block the event loop.
        # onnxruntime session creation can take a moment on first load.
        print("🚀 Loading MLP model at startup...")
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, load_ml_model)
        if success:
            print("✅ MLP Model loaded successfully")
        else:
            print("❌ MLP Model failed to load — train it first:")
            print("   Open src/app/core/ml/sign_model_mlp/train_mlp_signsync.ipynb in Google Colab")
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
