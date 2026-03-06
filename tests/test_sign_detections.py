"""Unit tests for sign detection endpoints."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.core.exceptions.http_exceptions import ForbiddenException, NotFoundException


@pytest.fixture
def sample_detection_read():
    """A valid SignDetectionRead dict returned by the CRUD layer."""
    return {
        "id": 1,
        "user_id": 1,
        "detected_sign": "A",
        "confidence": 0.95,
        "is_correct": True,
        "session_id": "sess_abc123",
        "duration_seconds": 1.5,
        "created_at": datetime.now(UTC),
    }


@pytest.fixture
def detection_create_payload():
    """Valid payload for POST /detection/log."""
    return {
        "detected_sign": "B",
        "confidence": 0.88,
        "is_correct": True,
        "session_id": "sess_xyz",
        "duration_seconds": 2.0,
    }


class TestLogDetection:
    """Tests for POST /detection/log."""

    @pytest.mark.asyncio
    async def test_log_detection_success(self, mock_db, current_user_dict, detection_create_payload, sample_detection_read):
        """Authenticated user can log a sign detection."""
        from src.app.api.v1.sign_detections import log_detection
        from src.app.schemas.sign_detection import SignDetectionCreate

        payload = SignDetectionCreate(**detection_create_payload)

        with patch("src.app.api.v1.sign_detections.crud_sign_detections") as mock_crud:
            mock_crud.create = AsyncMock(return_value=sample_detection_read)

            result = await log_detection(MagicMock(), payload, current_user_dict, mock_db)

            assert result is not None
            mock_crud.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_detection_sets_user_id(self, mock_db, current_user_dict, detection_create_payload, sample_detection_read):
        """The user_id from the auth token is injected into the detection record."""
        from src.app.api.v1.sign_detections import log_detection
        from src.app.schemas.sign_detection import SignDetectionCreate

        payload = SignDetectionCreate(**detection_create_payload)

        with patch("src.app.api.v1.sign_detections.crud_sign_detections") as mock_crud:
            mock_crud.create = AsyncMock(return_value=sample_detection_read)

            await log_detection(MagicMock(), payload, current_user_dict, mock_db)

            call_kwargs = mock_crud.create.call_args.kwargs
            created_object = call_kwargs.get("object")
            assert created_object is not None
            assert created_object.user_id == current_user_dict["id"]


class TestGetDetection:
    """Tests for GET /detection/{id}."""

    @pytest.mark.asyncio
    async def test_get_own_detection_success(self, mock_db, current_user_dict, sample_detection_read):
        """User can retrieve their own detection record."""
        from src.app.api.v1.sign_detections import get_detection

        with patch("src.app.api.v1.sign_detections.crud_sign_detections") as mock_crud:
            mock_crud.get = AsyncMock(return_value=sample_detection_read)

            result = await get_detection(detection_id=1, current_user=current_user_dict, db=mock_db)

            assert result is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent_detection_raises_404(self, mock_db, current_user_dict):
        """Getting a detection that does not exist raises NotFoundException."""
        from src.app.api.v1.sign_detections import get_detection

        with patch("src.app.api.v1.sign_detections.crud_sign_detections") as mock_crud:
            mock_crud.get = AsyncMock(return_value=None)

            with pytest.raises(NotFoundException):
                await get_detection(detection_id=999, current_user=current_user_dict, db=mock_db)

    @pytest.mark.asyncio
    async def test_get_other_users_detection_raises_403(self, mock_db, current_user_dict, sample_detection_read):
        """A user cannot read another user's detection record."""
        from src.app.api.v1.sign_detections import get_detection

        other_users_detection = {**sample_detection_read, "user_id": current_user_dict["id"] + 999}

        with patch("src.app.api.v1.sign_detections.crud_sign_detections") as mock_crud:
            mock_crud.get = AsyncMock(return_value=other_users_detection)

            with pytest.raises(ForbiddenException):
                await get_detection(detection_id=1, current_user=current_user_dict, db=mock_db)


class TestDeleteDetection:
    """Tests for DELETE /detection/{id}."""

    @pytest.mark.asyncio
    async def test_delete_own_detection_success(self, mock_db, current_user_dict, sample_detection_read):
        """User can soft-delete their own detection."""
        from src.app.api.v1.sign_detections import delete_detection

        with patch("src.app.api.v1.sign_detections.crud_sign_detections") as mock_crud:
            mock_crud.get = AsyncMock(return_value=sample_detection_read)
            mock_crud.delete = AsyncMock(return_value=None)

            result = await delete_detection(detection_id=1, current_user=current_user_dict, db=mock_db)

            mock_crud.delete.assert_called_once()
            assert result is not None

    @pytest.mark.asyncio
    async def test_delete_other_users_detection_raises_403(self, mock_db, current_user_dict, sample_detection_read):
        """A user cannot delete another user's detection."""
        from src.app.api.v1.sign_detections import delete_detection

        other_users_detection = {**sample_detection_read, "user_id": current_user_dict["id"] + 999}

        with patch("src.app.api.v1.sign_detections.crud_sign_detections") as mock_crud:
            mock_crud.get = AsyncMock(return_value=other_users_detection)

            with pytest.raises(ForbiddenException):
                await delete_detection(detection_id=1, current_user=current_user_dict, db=mock_db)
