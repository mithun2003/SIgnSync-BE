"""Unit tests for the Redis rate limiter utility."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from src.app.core.utils.rate_limit import RateLimiter


class TestRateLimiter:
    """Tests for RateLimiter.is_rate_limited()."""

    @pytest.mark.asyncio
    async def test_under_limit_not_rate_limited(self):
        """Request count below the limit returns False (not limited)."""
        limiter = RateLimiter()
        mock_client = MagicMock()
        mock_client.incr = AsyncMock(return_value=1)
        mock_client.expire = AsyncMock()

        with patch.object(limiter, "get_client", return_value=mock_client):
            result = await limiter.is_rate_limited(
                db=MagicMock(), user_id=1, path="/api/v1/test", limit=10, period=3600
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_over_limit_is_rate_limited(self):
        """Request count exceeding the limit returns True (limited)."""
        limiter = RateLimiter()
        mock_client = MagicMock()
        mock_client.incr = AsyncMock(return_value=11)
        mock_client.expire = AsyncMock()

        with patch.object(limiter, "get_client", return_value=mock_client):
            result = await limiter.is_rate_limited(
                db=MagicMock(), user_id=1, path="/api/v1/test", limit=10, period=3600
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_at_limit_boundary_is_rate_limited(self):
        """Request exactly at the limit (count == limit + 1) is blocked."""
        limiter = RateLimiter()
        mock_client = MagicMock()
        mock_client.incr = AsyncMock(return_value=11)  # limit=10, count=11 → blocked
        mock_client.expire = AsyncMock()

        with patch.object(limiter, "get_client", return_value=mock_client):
            result = await limiter.is_rate_limited(
                db=MagicMock(), user_id=1, path="/api/v1/test", limit=10, period=3600
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_redis_error_fails_open(self):
        """When Redis raises an error the rate limiter fails open (returns False)."""
        limiter = RateLimiter()
        mock_client = MagicMock()
        mock_client.incr = AsyncMock(side_effect=RedisConnectionError("connection refused"))

        with patch.object(limiter, "get_client", return_value=mock_client):
            result = await limiter.is_rate_limited(
                db=MagicMock(), user_id=1, path="/api/v1/test", limit=10, period=3600
            )

        # Must not raise — traffic should be allowed when Redis is unavailable
        assert result is False

    @pytest.mark.asyncio
    async def test_ttl_set_on_first_request(self):
        """expire() is called exactly once when the key is first created (count == 1)."""
        limiter = RateLimiter()
        mock_client = MagicMock()
        mock_client.incr = AsyncMock(return_value=1)
        mock_client.expire = AsyncMock()

        with patch.object(limiter, "get_client", return_value=mock_client):
            await limiter.is_rate_limited(
                db=MagicMock(), user_id=42, path="/api/v1/signs", limit=100, period=60
            )

        mock_client.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_ttl_not_reset_on_subsequent_requests(self):
        """expire() is NOT called again on follow-up requests within the same window."""
        limiter = RateLimiter()
        mock_client = MagicMock()
        mock_client.incr = AsyncMock(return_value=5)  # count > 1 → already exists
        mock_client.expire = AsyncMock()

        with patch.object(limiter, "get_client", return_value=mock_client):
            await limiter.is_rate_limited(
                db=MagicMock(), user_id=42, path="/api/v1/signs", limit=100, period=60
            )

        mock_client.expire.assert_not_called()
