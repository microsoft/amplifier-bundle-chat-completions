"""Tests for amplifier_module_provider_chat_completions module.

Tests verify:
- Module metadata (__amplifier_module_type__, __all__ exports)
- mount() function is callable
- mount() returns a coroutine (async function)
- Error translation from OpenAI SDK exceptions to kernel error types
- Message conversion (inbound: messages -> wire format)
- Response building (outbound: API response -> ChatResponse)
- get_info() returns ProviderInfo with correct fields
- close() works safely with and without a client initialized
- Retry with backoff on retryable errors
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

import openai

from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)

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
            config={
                "model": "test-model",
                "use_streaming": "false",
                "max_retries": "0",
            },
        )
        provider.coordinator = FakeCoordinator()
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
        sdk_err = _make_openai_error(openai.BadRequestError, "too many tokens", 400)
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
        sdk_err = _make_openai_error(openai.BadRequestError, "request blocked", 400)
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelContentFilterError)

    def test_bad_request_generic(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(openai.BadRequestError, "invalid model name", 400)
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelInvalidRequestError)
        assert err.retryable is False

    def test_authentication_error(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(openai.AuthenticationError, "invalid key", 401)
        err = provider._translate_error(sdk_err)
        assert isinstance(err, KernelAuthenticationError)
        assert err.retryable is False

    def test_permission_denied_error(self):
        provider = self._get_provider()
        sdk_err = _make_openai_error(openai.PermissionDeniedError, "forbidden", 403)
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
        sdk_err = _make_openai_error(openai.InternalServerError, "internal error", 500)
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


# ---------------------------------------------------------------------------
# Message conversion tests (inbound: internal -> wire format)
# ---------------------------------------------------------------------------


class TestMessageConversionInbound:
    """Tests for _convert_messages_to_wire: internal Message list -> OpenAI wire dicts."""

    def _get_provider(self):
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        return ChatCompletionsProvider(config={"model": "test-model"})

    def test_string_content(self):
        """String content passes through as-is with the original role."""
        provider = self._get_provider()
        msgs = [Message(role="user", content="hello")]
        result = provider._convert_messages_to_wire(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_text_blocks_joined(self):
        """Multiple TextBlocks in content list are joined with newline."""
        provider = self._get_provider()
        msgs = [
            Message(
                role="user",
                content=[TextBlock(text="hello"), TextBlock(text="world")],
            )
        ]
        result = provider._convert_messages_to_wire(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello\nworld"

    def test_thinking_blocks_dropped(self):
        """ThinkingBlocks are silently omitted from the wire format."""
        provider = self._get_provider()
        msgs = [
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="internal reasoning"),
                    TextBlock(text="final answer"),
                ],
            )
        ]
        result = provider._convert_messages_to_wire(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "final answer"
        # No trace of the thinking block
        assert "thinking" not in str(result[0])

    def test_tool_call_blocks_become_tool_calls(self):
        """ToolCallBlocks produce a tool_calls array with JSON-string arguments."""
        provider = self._get_provider()
        msgs = [
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(id="call_1", name="my_func", input={"arg": "val"})
                ],
            )
        ]
        result = provider._convert_messages_to_wire(msgs)
        assert len(result) == 1
        wire = result[0]
        assert wire["role"] == "assistant"
        assert "tool_calls" in wire
        assert len(wire["tool_calls"]) == 1
        tc = wire["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "my_func"
        # arguments must be a JSON string, not a dict
        assert isinstance(tc["function"]["arguments"], str)
        assert json.loads(tc["function"]["arguments"]) == {"arg": "val"}

    def test_tool_result_becomes_tool_message(self):
        """ToolResultBlock in a message produces a tool-role wire message."""
        provider = self._get_provider()
        msgs = [
            Message(
                role="tool",
                content=[ToolResultBlock(tool_call_id="call_1", output="42")],
            )
        ]
        result = provider._convert_messages_to_wire(msgs)
        assert len(result) == 1
        assert result[0] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "42",
        }

    def test_developer_role_becomes_system(self):
        """'developer' role is remapped to 'system' for OpenAI compatibility."""
        provider = self._get_provider()
        msgs = [Message(role="developer", content="You are a helpful assistant.")]
        result = provider._convert_messages_to_wire(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."

    def test_system_role_passthrough(self):
        """'system' role is preserved unchanged."""
        provider = self._get_provider()
        msgs = [Message(role="system", content="system prompt")]
        result = provider._convert_messages_to_wire(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "system prompt"


# ---------------------------------------------------------------------------
# Response building tests (outbound: OpenAI response -> ChatResponse)
# ---------------------------------------------------------------------------


def _make_mock_completion(
    content=None,
    tool_calls=None,
    reasoning_content=None,
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=5,
    total_tokens=15,
):
    """Build a minimal mock of an OpenAI ChatCompletion object."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.reasoning_content = reasoning_content  # None unless set

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestMessageConversionOutbound:
    """Tests for _build_response: OpenAI ChatCompletion -> ChatResponse."""

    def _get_provider(self):
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        return ChatCompletionsProvider(config={"model": "test-model"})

    def test_text_response(self):
        """message.content becomes a TextBlock in the ChatResponse."""
        provider = self._get_provider()
        response = _make_mock_completion(content="Hello there")
        result = provider._build_response(response)

        assert isinstance(result, ChatResponse)
        text_blocks = [b for b in result.content if isinstance(b, TextBlock)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "Hello there"
        # Usage is mapped from prompt/completion tokens
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15

    def test_tool_call_response(self):
        """message.tool_calls produce ToolCallBlock in content and ToolCall in tool_calls."""
        provider = self._get_provider()

        mock_tc = MagicMock()
        mock_tc.id = "call_abc"
        mock_tc.function.name = "search"
        mock_tc.function.arguments = json.dumps({"query": "test"})

        response = _make_mock_completion(content=None, tool_calls=[mock_tc])
        result = provider._build_response(response)

        assert isinstance(result, ChatResponse)
        # ToolCallBlock in content
        call_blocks = [b for b in result.content if isinstance(b, ToolCallBlock)]
        assert len(call_blocks) == 1
        assert call_blocks[0].id == "call_abc"
        assert call_blocks[0].name == "search"
        assert call_blocks[0].input == {"query": "test"}
        # ToolCall in tool_calls field
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "test"}

    def test_reasoning_content_becomes_thinking_block(self):
        """reasoning_content on the message becomes a ThinkingBlock in content."""
        provider = self._get_provider()
        response = _make_mock_completion(
            content="The answer is 42.",
            reasoning_content="Let me think step by step...",
        )
        result = provider._build_response(response)

        assert isinstance(result, ChatResponse)
        thinking_blocks = [b for b in result.content if isinstance(b, ThinkingBlock)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "Let me think step by step..."
        # Text content should still be present
        text_blocks = [b for b in result.content if isinstance(b, TextBlock)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "The answer is 42."


# ---------------------------------------------------------------------------
# get_info() tests
# ---------------------------------------------------------------------------


class TestGetInfo:
    """Verify get_info() returns a correctly populated ProviderInfo."""

    def _get_provider(self, model="gpt-4o"):
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        return ChatCompletionsProvider(config={"model": model})

    def test_returns_provider_info(self):
        """get_info() returns a ProviderInfo instance."""
        from amplifier_core.models import ProviderInfo

        provider = self._get_provider()
        info = provider.get_info()
        assert isinstance(info, ProviderInfo)

    def test_provider_id(self):
        """ProviderInfo.id must be 'chat-completions'."""
        provider = self._get_provider()
        info = provider.get_info()
        assert info.id == "chat-completions"

    def test_credential_env_vars(self):
        """credential_env_vars must include 'CHAT_COMPLETIONS_API_KEY'."""
        provider = self._get_provider()
        info = provider.get_info()
        assert "CHAT_COMPLETIONS_API_KEY" in info.credential_env_vars

    def test_has_all_config_fields(self):
        """All 10 config fields must be present."""
        expected_field_ids = {
            "api_key",
            "base_url",
            "model",
            "max_tokens",
            "temperature",
            "timeout",
            "max_retries",
            "min_retry_delay",
            "max_retry_delay",
            "use_streaming",
        }
        provider = self._get_provider()
        info = provider.get_info()
        actual_ids = {field.id for field in info.config_fields}
        assert expected_field_ids == actual_ids

    def test_base_url_has_env_var(self):
        """The base_url config field must have env_var='CHAT_COMPLETIONS_BASE_URL'."""
        provider = self._get_provider()
        info = provider.get_info()
        base_url_field = next(
            (f for f in info.config_fields if f.id == "base_url"), None
        )
        assert base_url_field is not None
        assert base_url_field.env_var == "CHAT_COMPLETIONS_BASE_URL"

    def test_api_key_is_secret_type(self):
        """The api_key config field must have field_type='secret'."""
        provider = self._get_provider()
        info = provider.get_info()
        api_key_field = next((f for f in info.config_fields if f.id == "api_key"), None)
        assert api_key_field is not None
        assert api_key_field.field_type == "secret"

    def test_defaults_include_model(self):
        """The defaults dict must include 'model' key matching self._model."""
        provider = self._get_provider(model="my-model")
        info = provider.get_info()
        assert "model" in info.defaults
        assert info.defaults["model"] == "my-model"

    def test_capabilities_include_tools_and_streaming(self):
        """capabilities must include 'tools' and 'streaming'."""
        provider = self._get_provider()
        info = provider.get_info()
        assert "tools" in info.capabilities
        assert "streaming" in info.capabilities


# ---------------------------------------------------------------------------
# close() tests
# ---------------------------------------------------------------------------


class TestClose:
    """Verify close() works safely with and without a client initialized."""

    @pytest.mark.asyncio
    async def test_close_with_no_client_does_not_raise(self):
        """close() must not raise when _client is None."""
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        provider = ChatCompletionsProvider(config={})
        assert provider._client is None
        # Should not raise
        await provider.close()

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self):
        """close() must call _client.close() when client is initialized."""
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        provider = ChatCompletionsProvider(config={})
        mock_client = AsyncMock()
        provider._client = mock_client

        await provider.close()

        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


class TestRetry:
    """Verify retry behaviour in complete()."""

    def _make_provider(self):
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        return ChatCompletionsProvider(
            config={
                "model": "test-model",
                "max_retries": "2",
                "min_retry_delay": "0.01",
                "max_retry_delay": "0.02",
            }
        )

    @pytest.mark.asyncio
    async def test_retries_on_retryable_error(self):
        """Retryable errors (APIConnectionError) cause retries up to max_retries."""
        provider = self._make_provider()
        mock_client = AsyncMock()
        provider._client = mock_client

        conn_error = openai.APIConnectionError(request=MagicMock())
        mock_response = _make_mock_completion(content="done")
        mock_client.chat.completions.create.side_effect = [
            conn_error,
            conn_error,
            mock_response,
        ]

        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            model="test-model",
        )
        result = await provider.complete(request)
        assert result is not None
        assert mock_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self):
        """Non-retryable errors (AuthenticationError) are raised immediately with no retries."""
        provider = self._make_provider()
        mock_client = AsyncMock()
        provider._client = mock_client

        auth_error = _make_openai_error(openai.AuthenticationError, "invalid key", 401)
        mock_client.chat.completions.create.side_effect = auth_error

        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            model="test-model",
        )

        with pytest.raises(KernelAuthenticationError):
            await provider.complete(request)

        assert mock_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_emits_provider_retry_event(self):
        """provider:retry event is emitted on each retry attempt."""
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        provider = ChatCompletionsProvider(
            config={
                "model": "test-model",
                "max_retries": "2",
                "min_retry_delay": "0.01",
                "max_retry_delay": "0.02",
            },
            coordinator=FakeCoordinator(),
        )
        mock_client = AsyncMock()
        provider._client = mock_client

        conn_error = openai.APIConnectionError(request=MagicMock())
        mock_response = _make_mock_completion(content="done")
        mock_client.chat.completions.create.side_effect = [
            conn_error,
            conn_error,
            mock_response,
        ]

        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            model="test-model",
        )
        await provider.complete(request)

        # Two retries should each emit one provider:retry event.
        assert provider.coordinator is not None
        hooks = provider.coordinator.hooks
        retry_events = [e for e in hooks.events if e[0] == "provider:retry"]
        assert len(retry_events) == 2

        # Verify payload shape on the first event.
        _, payload = retry_events[0]
        assert payload["provider"] == "chat-completions"
        assert payload["max_retries"] == 2
        assert "attempt" in payload
        assert "delay" in payload
        assert "error_type" in payload
        assert "error_message" in payload


# ---------------------------------------------------------------------------
# Tool repair tests
# ---------------------------------------------------------------------------


class TestToolRepair:
    """Tests for _repair_tool_sequence: orphaned tool call detection and repair."""

    def _make_provider(self):
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        return ChatCompletionsProvider(
            config={"model": "test-model"},
            coordinator=FakeCoordinator(),
        )

    @pytest.mark.asyncio
    async def test_missing_tool_result_is_repaired(self):
        """Assistant message with ToolCallBlock but no matching tool result gets repaired."""
        provider = self._make_provider()

        messages = [
            Message(role="user", content="Call a tool"),
            Message(
                role="assistant",
                content=[ToolCallBlock(id="call_1", name="my_func", input={})],
            ),
            # No tool result message follows
        ]

        result = await provider._repair_tool_sequence(messages)

        # Should have injected a synthetic tool result after the assistant message
        assert len(result) == 3
        synthetic = result[2]
        assert synthetic.role == "tool"
        assert synthetic.tool_call_id == "call_1"
        assert "[ERROR] Tool result missing" in str(synthetic.content)

        # Verify repair event was emitted
        events = provider.coordinator.hooks.events
        repair_events = [e for e in events if e[0] == "provider:tool_sequence_repaired"]
        assert len(repair_events) == 1
        _, payload = repair_events[0]
        assert payload["repaired_count"] == 1
        assert "call_1" in payload["repaired_tool_ids"]
        assert payload["provider"] == "chat-completions"
        assert "model" in payload

    @pytest.mark.asyncio
    async def test_repaired_ids_not_detected_again(self):
        """Already-repaired tool call IDs are not re-repaired on a second call."""
        provider = self._make_provider()

        messages = [
            Message(role="user", content="Call a tool"),
            Message(
                role="assistant",
                content=[ToolCallBlock(id="call_1", name="my_func", input={})],
            ),
        ]

        # First call repairs the orphaned call
        result1 = await provider._repair_tool_sequence(messages)
        assert len(result1) == 3

        # Second call with the same original messages (still without tool result)
        # should NOT re-repair because call_1 is already in _repaired_tool_ids
        result2 = await provider._repair_tool_sequence(messages)
        assert len(result2) == 2  # No additional injection

        # Only 1 repair event total (emitted during first call only)
        events = provider.coordinator.hooks.events
        repair_events = [e for e in events if e[0] == "provider:tool_sequence_repaired"]
        assert len(repair_events) == 1

    @pytest.mark.asyncio
    async def test_multiple_missing_tool_results(self):
        """Multiple orphaned tool calls in a single assistant message all get repaired."""
        provider = self._make_provider()

        messages = [
            Message(role="user", content="Call tools"),
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(id="call_1", name="func1", input={}),
                    ToolCallBlock(id="call_2", name="func2", input={}),
                    ToolCallBlock(id="call_3", name="func3", input={}),
                ],
            ),
            # No tool results follow
        ]

        result = await provider._repair_tool_sequence(messages)

        # 2 original messages + 3 injected synthetic tool results
        assert len(result) == 5

        # Verify the repair event
        events = provider.coordinator.hooks.events
        repair_events = [e for e in events if e[0] == "provider:tool_sequence_repaired"]
        assert len(repair_events) == 1
        _, payload = repair_events[0]
        assert payload["repaired_count"] == 3
        assert set(payload["repaired_tool_ids"]) == {"call_1", "call_2", "call_3"}


# ---------------------------------------------------------------------------
# Message edge cases tests
# ---------------------------------------------------------------------------


class TestMessageEdgeCases:
    """Tests for message conversion edge cases."""

    def _get_provider(self):
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        return ChatCompletionsProvider(config={"model": "test-model"})

    def test_developer_role_prepended_before_user(self):
        """Developer message (converted to system) is prepended before user messages."""
        provider = self._get_provider()
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="developer", content="You are a helpful assistant."),
        ]
        result = provider._convert_messages_to_wire(msgs)
        # developer message should come first as system, before the user message
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello"

    def test_image_block_url_format(self):
        """ImageBlock with base64 source produces correct data: URL format."""
        provider = self._get_provider()
        msgs = [
            Message(
                role="user",
                content=[
                    TextBlock(text="Look at this image"),
                    ImageBlock(
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc123",
                        }
                    ),
                ],
            )
        ]
        result = provider._convert_messages_to_wire(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        # Find the image_url item
        image_items = [item for item in content if item["type"] == "image_url"]
        assert len(image_items) == 1
        assert image_items[0]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_empty_content_string(self):
        """Empty string content doesn't crash and passes through as empty string."""
        provider = self._get_provider()
        msgs = [Message(role="user", content="")]
        result = provider._convert_messages_to_wire(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == ""

    def test_empty_content_list(self):
        """Empty content list handles gracefully with the message still having correct role."""
        provider = self._get_provider()
        msgs = [Message(role="user", content=[])]
        result = provider._convert_messages_to_wire(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_tool_result_block_in_content(self):
        """ToolResultBlock in content list produces role='tool' with correct tool_call_id."""
        provider = self._get_provider()
        msgs = [
            Message(
                role="tool",
                content=[
                    ToolResultBlock(tool_call_id="call_xyz", output="result data")
                ],
            )
        ]
        result = provider._convert_messages_to_wire(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_xyz"
        assert result[0]["content"] == "result data"


# ---------------------------------------------------------------------------
# Observability events tests
# ---------------------------------------------------------------------------


class TestObservabilityEvents:
    """Tests for llm:request and llm:response observability event emission."""

    def _make_provider(self, coordinator=None):
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        return ChatCompletionsProvider(
            config={
                "model": "test-model",
                "max_retries": "0",
                "min_retry_delay": "0.01",
                "max_retry_delay": "0.02",
            },
            coordinator=coordinator,
        )

    @pytest.mark.asyncio
    async def test_llm_request_event_emitted(self):
        """llm:request event is emitted before the API call with provider and model."""
        provider = self._make_provider(coordinator=FakeCoordinator())
        mock_client = AsyncMock()
        provider._client = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            content="hello"
        )

        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            model="test-model",
        )
        await provider.complete(request)

        hooks = provider.coordinator.hooks
        request_events = [e for e in hooks.events if e[0] == "llm:request"]
        assert len(request_events) >= 1, "llm:request event must be emitted"

        _, payload = request_events[0]
        assert "provider" in payload, "llm:request payload must have 'provider'"
        assert "model" in payload, "llm:request payload must have 'model'"
        assert payload["provider"] == "chat-completions"

    @pytest.mark.asyncio
    async def test_llm_response_event_emitted(self):
        """llm:response event is emitted after successful response with provider, usage, duration_ms."""
        provider = self._make_provider(coordinator=FakeCoordinator())
        mock_client = AsyncMock()
        provider._client = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            content="hello",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            model="test-model",
        )
        await provider.complete(request)

        hooks = provider.coordinator.hooks
        # Filter for non-error llm:response events
        response_events = [
            e
            for e in hooks.events
            if e[0] == "llm:response" and e[1].get("status") != "error"
        ]
        assert len(response_events) >= 1, "llm:response event must be emitted"

        _, payload = response_events[0]
        assert "provider" in payload, "llm:response payload must have 'provider'"
        assert "usage" in payload, "llm:response payload must have 'usage'"
        assert "duration_ms" in payload, "llm:response payload must have 'duration_ms'"
        assert payload["provider"] == "chat-completions"
        assert isinstance(payload["duration_ms"], float)

    @pytest.mark.asyncio
    async def test_events_skipped_when_no_hooks(self):
        """No crash when coordinator is None — events are silently skipped."""
        provider = self._make_provider(coordinator=None)
        mock_client = AsyncMock()
        provider._client = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            content="hello"
        )

        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            model="test-model",
        )
        # Must not raise even though coordinator is None
        result = await provider.complete(request)
        assert result is not None


# ---------------------------------------------------------------------------
# list_models() tests
# ---------------------------------------------------------------------------


class TestListModels:
    """Tests for list_models(): dynamic model listing with graceful fallback."""

    def _make_provider(self, model: str = "test-model"):
        from amplifier_module_provider_chat_completions import ChatCompletionsProvider

        return ChatCompletionsProvider(config={"model": model, "max_tokens": "4096"})

    @pytest.mark.asyncio
    async def test_returns_models_from_server(self):
        """list_models() returns ModelInfo objects from the server's /v1/models endpoint."""
        from amplifier_core.models import ModelInfo

        provider = self._make_provider()

        # Mock model object with id attribute
        mock_model = MagicMock()
        mock_model.id = "gemma26-long"

        # Mock page with .data list
        mock_page = MagicMock()
        mock_page.data = [mock_model]

        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=mock_page)
        provider._client = mock_client

        result = await provider.list_models()

        assert len(result) >= 1
        assert any(m.id == "gemma26-long" for m in result)
        assert all(isinstance(m, ModelInfo) for m in result)

    @pytest.mark.asyncio
    async def test_returns_fallback_on_failure(self):
        """list_models() returns fallback with configured model id on any exception."""
        from amplifier_core.models import ModelInfo

        provider = self._make_provider(model="my-configured-model")

        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(side_effect=Exception("Connection failed"))
        provider._client = mock_client

        result = await provider.list_models()

        assert len(result) == 1
        assert isinstance(result[0], ModelInfo)
        assert result[0].id == "my-configured-model"
