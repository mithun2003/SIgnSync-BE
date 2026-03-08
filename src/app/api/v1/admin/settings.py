import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from ....api.dependencies import get_current_superuser
from ....schemas.admin import SettingsUpdate

SETTINGS_FILE = Path("/app/data/settings.json")

DEFAULT_SETTINGS: dict[str, Any] = {
    "general": {
        "app_name": "SignSync",
        "app_description": "AI-powered sign language detection and learning platform",
        "timezone": "UTC",
        "language": "en",
    },
    "security": {
        "min_password_length": 8,
        "require_uppercase": True,
        "require_numbers": True,
        "require_special_chars": False,
        "session_timeout": 30,
        "enable_2fa": False,
    },
    "notifications": {
        "email": {
            "new_users": True,
            "system_alerts": True,
            "backup_completion": False,
        },
        "in_app": {
            "show_badges": True,
            "auto_dismiss": True,
        },
    },
    "system": {
        "cache_duration": 2,
        "debug_mode": False,
        "maintenance_mode": False,
        "auto_backup": True,
    },
}

router = APIRouter(tags=["Admin Settings"])


def _load_settings() -> dict[str, Any]:
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text())
    except Exception:
        pass
    return {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULT_SETTINGS.items()}


def _save_settings(data: dict[str, Any]) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge patch into base (one extra level for nested dicts)."""
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@router.get(
    "/",
    dependencies=[Depends(get_current_superuser)],
)
async def get_settings() -> dict[str, Any]:
    """Return current application settings."""
    return _load_settings()


@router.patch(
    "/",
    dependencies=[Depends(get_current_superuser)],
)
async def update_settings(updates: SettingsUpdate) -> dict[str, Any]:
    """Partially update application settings."""
    current = _load_settings()
    patch = updates.model_dump(exclude_unset=True)

    # Remove None values from nested dicts produced by model_dump
    def strip_none(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: strip_none(v) for k, v in d.items() if v is not None}
        return d

    clean_patch = strip_none(patch)
    merged = _deep_merge(current, clean_patch)
    _save_settings(merged)
    return merged
