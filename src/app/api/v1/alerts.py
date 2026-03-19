# app/api/v1/alerts.py
import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.db.database import async_get_db
from ...core.exceptions.http_exceptions import BadRequestException, NotFoundException
from ...crud.crud_users import crud_users
from ...schemas.user import HelpMailData, HelpMailResponse
from ...services.email_service import (
    HELP_ALERT_COOLDOWN_SECONDS,
    dispatch_help_sign_alerts,
)
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post("/help-mail", response_model=HelpMailResponse)
async def send_help_mail(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(async_get_db)],
):
    """Send emergency help email to user's configured emergency contacts.

    Emergency contacts can be managed via the /user/me/emergency-contacts endpoints. If no emergency contacts are
    configured, this endpoint will return an error.
    """
    db_user = await crud_users.get(db=db, id=current_user["id"])
    if not db_user:
        raise NotFoundException("User not found")

    dispatch_result = await dispatch_help_sign_alerts(db_user)
    if not dispatch_result.recipients:
        raise BadRequestException(
            "No emergency contacts configured. Please add emergency contacts via /user/me/emergency-contacts"
        )

    if dispatch_result.skipped_by_cooldown:
        logger.info(
            "Help mail skipped due to %d-second cooldown for user_id=%s",
            HELP_ALERT_COOLDOWN_SECONDS,
            current_user["id"],
        )
        return HelpMailResponse(
            data=HelpMailData(
                message="Emergency alert email was sent recently. Skipping duplicate notification.",
                recipients=dispatch_result.recipients,
                sent_count=0,
            )
        )

    if dispatch_result.sent_count == 0:
        raise BadRequestException("Failed to send emergency alert email to any recipient")

    logger.info(
        "Help mail triggered by user_id=%s, sent to %d/%d recipients",
        current_user["id"],
        dispatch_result.sent_count,
        len(dispatch_result.recipients),
    )

    if dispatch_result.failed_recipients:
        logger.warning("Failed to send to: %s", ", ".join(dispatch_result.failed_recipients))

    return HelpMailResponse(
        data=HelpMailData(
            message=(f"Emergency alert email sent successfully to {dispatch_result.sent_count} recipient(s)"),
            recipients=dispatch_result.recipients,
            sent_count=dispatch_result.sent_count,
        )
    )
