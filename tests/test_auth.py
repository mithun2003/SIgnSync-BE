"""Unit tests for authentication endpoints (register, login, logout, refresh)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.core.exceptions.http_exceptions import DuplicateValueException, UnauthorizedException


class TestRegister:
    """Tests for POST /auth/register."""

    @pytest.mark.asyncio
    async def test_register_success(self, mock_db, sample_user_data):
        """New user can register with unique email and username."""
        from src.app.api.v1.auth import register_user
        from src.app.schemas.user import UserCreate

        user_input = UserCreate(**sample_user_data)

        with patch("src.app.api.v1.auth.crud_users") as mock_crud, \
             patch("src.app.api.v1.auth.get_password_hash", return_value="hashed_pw"):
            mock_crud.exists = AsyncMock(side_effect=[False, False])
            mock_crud.create = AsyncMock(return_value={"id": 1, "email": user_input.email})

            result = await register_user(MagicMock(), user_input, mock_db)

            assert result["email"] == user_input.email
            assert mock_crud.create.called

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, mock_db, sample_user_data):
        """Registration fails when email is already taken."""
        from src.app.api.v1.auth import register_user
        from src.app.schemas.user import UserCreate

        user_input = UserCreate(**sample_user_data)

        with patch("src.app.api.v1.auth.crud_users") as mock_crud:
            mock_crud.exists = AsyncMock(return_value=True)

            with pytest.raises(DuplicateValueException, match="Email already registered"):
                await register_user(MagicMock(), user_input, mock_db)

    @pytest.mark.asyncio
    async def test_register_duplicate_username(self, mock_db, sample_user_data):
        """Registration fails when username is already taken."""
        from src.app.api.v1.auth import register_user
        from src.app.schemas.user import UserCreate

        user_input = UserCreate(**sample_user_data)

        with patch("src.app.api.v1.auth.crud_users") as mock_crud:
            # First exists() call (email) returns False, second (username) returns True
            mock_crud.exists = AsyncMock(side_effect=[False, True])

            with pytest.raises(DuplicateValueException, match="Username not available"):
                await register_user(MagicMock(), user_input, mock_db)

    @pytest.mark.asyncio
    async def test_register_hashes_password(self, mock_db, sample_user_data):
        """Plain-text password is never stored — only the hash."""
        from src.app.api.v1.auth import register_user
        from src.app.schemas.user import UserCreate

        user_input = UserCreate(**sample_user_data)

        with patch("src.app.api.v1.auth.crud_users") as mock_crud, \
             patch("src.app.api.v1.auth.get_password_hash", return_value="hashed_pw") as mock_hash:
            mock_crud.exists = AsyncMock(return_value=False)
            mock_crud.create = AsyncMock(return_value={"id": 1, "email": user_input.email})

            await register_user(MagicMock(), user_input, mock_db)

            mock_hash.assert_called_once_with(sample_user_data["password"])
            # The raw password must not be forwarded to the database layer
            call_kwargs = mock_crud.create.call_args
            created_object = call_kwargs.kwargs.get("object") or call_kwargs.args[0]
            assert not hasattr(created_object, "password") or created_object.password is None


class TestLogin:
    """Tests for POST /auth/login."""

    @pytest.mark.asyncio
    async def test_login_success(self, mock_db):
        """Valid credentials return an access token and set a refresh cookie."""
        from src.app.api.v1.login import login_for_access_token

        form_data = MagicMock()
        form_data.username = "testuser"
        form_data.password = "correct_password"

        response = MagicMock()
        response.set_cookie = MagicMock()

        mock_user = {"id": 1, "username": "testuser", "email": "test@example.com", "is_superuser": False}

        with patch("src.app.api.v1.login.authenticate_user", new_callable=AsyncMock, return_value=mock_user), \
             patch("src.app.api.v1.login.create_access_token", new_callable=AsyncMock, return_value="access123"), \
             patch("src.app.api.v1.login.create_refresh_token", new_callable=AsyncMock, return_value="refresh456"):

            result = await login_for_access_token(response, form_data, mock_db)

            assert result.success is True
            assert result.data["access_token"] == "access123"
            assert result.data["token_type"] == "bearer"
            response.set_cookie.assert_called_once()

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, mock_db):
        """Wrong credentials raise UnauthorizedException."""
        from src.app.api.v1.login import login_for_access_token

        form_data = MagicMock()
        form_data.username = "testuser"
        form_data.password = "wrong_password"

        with patch("src.app.api.v1.login.authenticate_user", new_callable=AsyncMock, return_value=None):
            with pytest.raises(UnauthorizedException, match="Wrong username, email or password"):
                await login_for_access_token(MagicMock(), form_data, mock_db)

    @pytest.mark.asyncio
    async def test_login_sets_httponly_cookie(self, mock_db):
        """Refresh token cookie must be set with httponly=True."""
        from src.app.api.v1.login import login_for_access_token

        form_data = MagicMock()
        form_data.username = "testuser"
        form_data.password = "correct_password"

        response = MagicMock()
        mock_user = {"id": 1, "username": "testuser", "email": "test@example.com", "is_superuser": False}

        with patch("src.app.api.v1.login.authenticate_user", new_callable=AsyncMock, return_value=mock_user), \
             patch("src.app.api.v1.login.create_access_token", new_callable=AsyncMock, return_value="tok"), \
             patch("src.app.api.v1.login.create_refresh_token", new_callable=AsyncMock, return_value="ref"):

            await login_for_access_token(response, form_data, mock_db)

            cookie_kwargs = response.set_cookie.call_args.kwargs
            assert cookie_kwargs.get("httponly") is True

    @pytest.mark.asyncio
    async def test_login_superuser_role(self, mock_db):
        """Superuser gets 'admin' role in the response."""
        from src.app.api.v1.login import login_for_access_token

        form_data = MagicMock()
        form_data.username = "admin"
        form_data.password = "admin_pass"

        mock_user = {"id": 1, "username": "admin", "email": "admin@example.com", "is_superuser": True}

        with patch("src.app.api.v1.login.authenticate_user", new_callable=AsyncMock, return_value=mock_user), \
             patch("src.app.api.v1.login.create_access_token", new_callable=AsyncMock, return_value="tok"), \
             patch("src.app.api.v1.login.create_refresh_token", new_callable=AsyncMock, return_value="ref"):

            result = await login_for_access_token(MagicMock(), form_data, mock_db)

            assert result.data["user_role"] == "admin"
