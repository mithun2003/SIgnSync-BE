import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.core.config import settings
from src.app.core.db.database import async_session
from src.app.core.security import get_password_hash
from src.app.models.user import User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_first_user(session: AsyncSession) -> None:
    try:
        email = settings.ADMIN_EMAIL
        username = settings.ADMIN_USERNAME
        hashed_password = get_password_hash(settings.ADMIN_PASSWORD)

        result = await session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if user:
            logger.info(f"Admin user '{username}' already exists.")
            return

        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            is_superuser=True,
            is_active=True,
            created_at=datetime.now(UTC),
        )
        session.add(user)
        await session.commit()

        logger.info(f"✅ Admin user '{username}' created successfully.")

    except Exception as e:
        await session.rollback()
        logger.error(f"❌ Error creating admin user: {e}")
        raise


async def main():
    async with async_session() as session:
        await create_first_user(session)


if __name__ == "__main__":
    asyncio.run(main())
