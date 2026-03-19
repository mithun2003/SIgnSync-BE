"""Email service for sending notifications via SMTP."""

import asyncio
import logging
import smtplib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from ..core.config import settings

logger = logging.getLogger(__name__)

HELP_ALERT_COOLDOWN_SECONDS = 30
_help_alert_expiry: dict[int, float] = {}
_help_alert_lock = asyncio.Lock()


@dataclass(slots=True)
class HelpAlertDispatchResult:
    recipients: list[str]
    sent_count: int
    failed_recipients: list[str]
    skipped_by_cooldown: bool = False


def _normalize_name_part(value: Any) -> str:
    part = str(value or "").strip()
    if part.lower() in {"none", "null"}:
        return ""
    return part


def resolve_user_display_name(user: Mapping[str, Any]) -> str:
    """Resolve display name for alert emails from full name or username."""
    first_name = _normalize_name_part(user.get("first_name"))
    last_name = _normalize_name_part(user.get("last_name"))
    full_name = f"{first_name} {last_name}".strip()
    username = _normalize_name_part(user.get("username"))
    return full_name or username or "User"


@dataclass(slots=True)
class HelpAlertUserIdentity:
    display_name: str
    full_name: str
    username: str
    email: str


def resolve_help_alert_user_identity(user: Mapping[str, Any]) -> HelpAlertUserIdentity:
    """Build normalized identity details for help alert emails."""
    first_name = _normalize_name_part(user.get("first_name"))
    last_name = _normalize_name_part(user.get("last_name"))
    full_name = f"{first_name} {last_name}".strip()
    username = _normalize_name_part(user.get("username"))
    email = _normalize_name_part(user.get("email"))
    display_name = full_name or username or email or "User"
    return HelpAlertUserIdentity(
        display_name=display_name,
        full_name=full_name,
        username=username,
        email=email,
    )


def get_emergency_recipients(user: Mapping[str, Any]) -> list[str]:
    """Return normalized emergency contact recipients for a user."""
    emergency_contacts = user.get("emergency_contacts") or []
    return sorted({str(email).strip().lower() for email in emergency_contacts if str(email).strip()})


async def _acquire_help_alert_cooldown(user_id: int | None) -> bool:
    if user_id is None:
        return True

    now = time.monotonic()
    async with _help_alert_lock:
        current_expiry = _help_alert_expiry.get(user_id, 0.0)
        if current_expiry > now:
            return False

        _help_alert_expiry[user_id] = now + HELP_ALERT_COOLDOWN_SECONDS
        stale_user_ids = [uid for uid, expiry in _help_alert_expiry.items() if expiry <= now]
        for stale_user_id in stale_user_ids:
            _help_alert_expiry.pop(stale_user_id, None)
        return True


async def _clear_help_alert_cooldown(user_id: int | None) -> None:
    if user_id is None:
        return

    async with _help_alert_lock:
        _help_alert_expiry.pop(user_id, None)


async def dispatch_help_sign_alerts(user: Mapping[str, Any]) -> HelpAlertDispatchResult:
    """Send help-alert emails for a user with short duplicate suppression."""
    recipients = get_emergency_recipients(user)
    if not recipients:
        return HelpAlertDispatchResult(recipients=[], sent_count=0, failed_recipients=[])

    raw_user_id = user.get("id")
    user_id = raw_user_id if isinstance(raw_user_id, int) else None
    if not await _acquire_help_alert_cooldown(user_id):
        return HelpAlertDispatchResult(
            recipients=recipients,
            sent_count=0,
            failed_recipients=[],
            skipped_by_cooldown=True,
        )

    identity = resolve_help_alert_user_identity(user)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            None,
            email_service.send_help_sign_alert,
            recipient,
            identity.display_name,
            timestamp,
            identity.full_name,
            identity.username,
            identity.email,
        )
        for recipient in recipients
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    sent_count = sum(1 for result in results if result is True)
    failed_recipients = [recipients[index] for index, result in enumerate(results) if result is not True]

    if sent_count == 0:
        await _clear_help_alert_cooldown(user_id)

    return HelpAlertDispatchResult(
        recipients=recipients,
        sent_count=sent_count,
        failed_recipients=failed_recipients,
    )


class EmailService:
    """Service for sending emails via SMTP."""

    @staticmethod
    def send_help_sign_alert(
        to_email: str,
        user_name: str,
        timestamp: str,
        user_full_name: str = "",
        user_username: str = "",
        user_email: str = "",
    ) -> bool:
        """Send help sign alert email to emergency contact.

        Args:
            to_email: Recipient email address
            user_name: Primary display name of the user who showed help sign
            timestamp: Timestamp when the help sign was detected
            user_full_name: Full name of the user
            user_username: Username of the user
            user_email: Email of the user

        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        if not settings.EMAIL_ENABLED:
            logger.warning("Email service is disabled. Skipping email notification.")
            return False

        if not to_email:
            logger.warning("No emergency contact email provided. Skipping email notification.")
            return False

        identity_items_html: list[str] = []
        identity_items_text: list[str] = []

        if user_full_name:
            identity_items_html.append(f"<li><strong>Full name:</strong> {user_full_name}</li>")
            identity_items_text.append(f"- Full name: {user_full_name}")

        if user_username:
            identity_items_html.append(f"<li><strong>Username:</strong> {user_username}</li>")
            identity_items_text.append(f"- Username: {user_username}")

        if user_email:
            identity_items_html.append(f"<li><strong>Email:</strong> {user_email}</li>")
            identity_items_text.append(f"- Email: {user_email}")

        if not identity_items_html:
            identity_items_html.append("<li><strong>User:</strong> Not available</li>")
            identity_items_text.append("- User: Not available")

        identity_html = "\n                        ".join(identity_items_html)
        identity_text = "\n        ".join(identity_items_text)

        subject = f"🚨 Emergency Alert from {user_name}"

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 10px;
                }}
                .header {{
                    background-color: #dc3545;
                    color: white;
                    padding: 20px;
                    border-radius: 10px 10px 0 0;
                    text-align: center;
                }}
                .content {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                }}
                .alert-icon {{
                    font-size: 48px;
                    margin-bottom: 10px;
                }}
                .timestamp {{
                    color: #666;
                    font-size: 14px;
                    margin-top: 20px;
                }}
                .footer {{
                    margin-top: 20px;
                    text-align: center;
                    color: #666;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="alert-icon">🚨</div>
                    <h1>Emergency Alert</h1>
                </div>
                <div class="content">
                    <p><strong>{user_name}</strong> has shown a <strong>HELP sign</strong> during their
                    session on SignSync.</p>

                    <p>This is an automated alert sent because you are listed as their emergency contact.</p>

                    <p><strong>User details:</strong></p>
                    <ul>
                        {identity_html}
                    </ul>

                    <p><strong>What to do:</strong></p>
                    <ul>
                        <li>Check in with {user_name} immediately</li>
                        <li>Ensure they are safe and okay</li>
                        <li>Provide assistance if needed</li>
                    </ul>

                    <p class="timestamp">Alert triggered at: {timestamp}</p>
                </div>
                <div class="footer">
                    <p>This is an automated message from SignSync. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_body = f"""
        EMERGENCY ALERT

        {user_name} has shown a HELP sign during their session on SignSync.

        This is an automated alert sent because you are listed as their emergency contact.

        User details:
        {identity_text}

        What to do:
        - Check in with {user_name} immediately
        - Ensure they are safe and okay
        - Provide assistance if needed

        Alert triggered at: {timestamp}

        ---
        This is an automated message from SignSync. Please do not reply to this email.
        """

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM_EMAIL}>"
            msg["To"] = to_email

            part1 = MIMEText(text_body, "plain")
            part2 = MIMEText(html_body, "html")

            msg.attach(part1)
            msg.attach(part2)

            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                server.starttls()
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.send_message(msg)

            logger.info("Help sign alert email sent successfully to %s", to_email)
            return True

        except Exception as e:
            logger.exception("Failed to send help sign alert email to %s: %s", to_email, e)
            return False


email_service = EmailService()
