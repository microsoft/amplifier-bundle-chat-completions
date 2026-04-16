"""Tests for amplifier_module_provider_chat_completions module.

Tests verify:
- Module metadata (__amplifier_module_type__, __all__ exports)
- mount() function is callable
- mount() returns a coroutine (async function)
- Error translation from OpenAI SDK exceptions to kernel error types
"""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

import amplifier_module_provider_chat_completions as module
from amplifier_core.llm_errors import (
    AccessDeniedError as KernelAccessDeniedError,
    AuthenticationError as KernelAuthenticationError,
    ContentFilterError as KernelContentFilterError,
    ContextLengthError as KernelContextLengthError,
    InvalidRequestError as KernelInvalidRequestError,
    LLMError as KernelLLMError,
    LLMTimeoutError as KernelLLMTimeoutError,
    NotFoundError as KernelNotFoundError,
    ProviderUnavailableError as KernelProviderUnavailableError,
    RateLimitError as KernelRateLimitError,
)


class TestModuleMetadata:
    """Verify the module's static metadata and exports."""

    def test_amplifier_module_type(self):
        """__amplifier_module_type__ must be 'provider'."""
        assert module.__amplifier_module_type__ == "provider"

    def test_all_exports(self):
        """__all__ must export exactly 'mount' and 'ChatCompletionsProvider'."""
        assert "mount" in module.__all__
        assert "ChatCompletionsProvider" in module.__all__

    def test_mount_is_callable(self):
        """mount must be a callable (async function)."""
        assert callable(module.mount)


class TestMountContract:
    """Verify the mount() function contract."""

    def test_mount_returns_coroutine(self):
        """mount() must return a coroutine when called."""
        config = MagicMock()
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()

        result = module.mount(config, coordinator)
        assert asyncio.iscoroutine(result), (
            "mount() must be an async function that returns a coroutine"
        )
        # Clean up the coroutine to avoid RuntimeWarning
        result.close()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_openai_error(cls, message="error", status_code=400):
    """Construct an OpenAI SDK error with the expected shape."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": message}}
    return cls(message, response=mock_response, body=None)


# ---------------------------------------------------------------------------
# Error translation tests
# ---------------------------------------------------------------------------


class TestErrorTranslation:
    """Verify _translate_error maps all SDK exceptions correctly."""

    def _get_provider(self):
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        provider = ChatCompletionsProvider(
            config={"model": "test-model", "use_streaming": "false", "max_retries": "0"},
        )
        provider.coordinator = cast(MagicMock, FakeCoordinator())
        return provider

    def test_timeout_error(self):
        provider = self._get_provider()
        err = provider._translate_error(openai.APITimeoutError(request=MagicMock()))
        assert isinstance(err, KernelLLMTimeoutError)
        assert err.provider == "chat-completions"
        assert err.model == "test-model"
        assert err.retryable is True

    def test_connection_error(self):
        provider = self._get_provider()
        err = provider._translate_error(openai.APIConnectionError(request=MagicMock()))
        assert isinstance(err, KernelProviderUnavailableError)
        assert err.retryable is True

    def test_rate_limit_error(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(openai.RateLimitError, "rate limited", 429)
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelRateLimitError)
        assert err.provider == "chat-completions"
        assert err.retryable is True

    def test_bad_request_context_length(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(
            openai.BadRequestError, "context length exceeded", 400
        )
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelContextLengthError)
        assert err.retryable is False

    def test_bad_request_too_many_tokens(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(
            openai.BadRequestError, "too many tokens", 400
        )
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelContextLengthError)

    def test_bad_request_content_filter(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(
            openai.BadRequestError, "content blocked by safety filter", 400
        )
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelContentFilterError)
        assert err.retryable is False

    def test_bad_request_blocked(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(
            openai.BadRequestError, "request blocked", 400
        )
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelContentFilterError)

    def test_bad_request_generic(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(
            openai.BadRequestError, "invalid model name", 400
        )
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelInvalidRequestError)
        assert err.retryable is False

    def test_authentication_error(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(
            openai.AuthenticationError, "invalid key", 401
        )
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelAuthenticationError)
        assert err.retryable is False

    def test_permission_denied_error(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(
            openai.PermissionDeniedError, "forbidden", 403
        )
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelAccessDeniedError)
        assert err.retryable is False

    def test_not_found_error(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(openai.NotFoundError, "not found", 404)
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelNotFoundError)
        assert err.retryable is False

    def test_5xx_server_error(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(
            openai.InternalServerError, "internal error", 500
        )
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelProviderUnavailableError)
        assert err.retryable is True

    def test_asyncio_timeout(self):
        provider = self._get_provider()
        err = provider._translate_error(asyncio.TimeoutError())
        assert isinstance(err, KernelLLMTimeoutError)
        assert err.retryable is True

    def test_generic_exception(self):
        provider = self._get_provider()
        err = provider._translate_error(RuntimeError("unexpected"))
        assert isinstance(err, KernelLLMError)
        assert err.retryable is True

    def test_all_errors_carry_provider_and_model(self):
        provider = self._get_provider()
        test_cases = [
            openai.APITimeoutError(request=MagicMock()),
            openai.APIConnectionError(request=MagicMock()),
            _make_openai_error(openai.RateLimitError, "x", 429),
            _make_openai_error(openai.BadRequestError, "x", 400),
            _make_openai_error(openai.AuthenticationError, "x", 401),
            _make_openai_error(openai.NotFoundError, "x", 404),
            _make_openai_error(openai.InternalServerError, "x", 500),
            asyncio.TimeoutError(),
            RuntimeError("boom"),
        ]
        for exc in test_cases:
            err = provider._translate_error(exc)
            assert err.provider == "chat-completions", f"Failed for {type(exc)}"
            assert err.model == "test-model", f"Failed for {type(exc)}"

    def test_cause_chain_preserved(self):
        provider = self._get_provider()
        original = _make_openai_error(openai.RateLimitError, "x", 429)
        err = provider._translate_error(original)
        assert err.__cause__ is original
