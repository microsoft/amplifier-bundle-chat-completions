"""Amplifier provider module for OpenAI-compatible chat completions.

This module mounts a ChatCompletionsProvider into the Amplifier coordinator,
making it available as the 'chat-completions' provider.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import openai

from amplifier_core.utils.retry import RetryConfig, retry_with_backoff
from amplifier_core.llm_errors import AccessDeniedError as KernelAccessDeniedError
from amplifier_core.llm_errors import AuthenticationError as KernelAuthenticationError
from amplifier_core.llm_errors import ContentFilterError as KernelContentFilterError
from amplifier_core.llm_errors import ContextLengthError as KernelContextLengthError
from amplifier_core.llm_errors import InvalidRequestError as KernelInvalidRequestError
from amplifier_core.llm_errors import LLMError as KernelLLMError
from amplifier_core.llm_errors import LLMTimeoutError as KernelLLMTimeoutError
from amplifier_core.llm_errors import NotFoundError as KernelNotFoundError
from amplifier_core.llm_errors import (
    ProviderUnavailableError as KernelProviderUnavailableError,
)
from amplifier_core.llm_errors import RateLimitError as KernelRateLimitError
from amplifier_core.models import ConfigField, ModelInfo, ProviderInfo
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResultBlock,
    ToolSpec,
    Usage,
)

__all__ = ["mount", "ChatCompletionsProvider"]
__amplifier_module_type__ = "provider"

logger = logging.getLogger(__name__)


class ChatCompletionsProvider:
    """Provider for OpenAI-compatible chat completions API."""

    name = "chat-completions"

    @staticmethod
    def _config_bool(value: Any) -> bool:
        """Parse config booleans from YAML or CLI string values."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _config_int(value: Any, default: int) -> int:
        """Parse an int config value with a safe fallback."""
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(
                "[PROVIDER] Invalid integer config value %r; using default %s",
                value,
                default,
            )
            return default

    @staticmethod
    def _config_float(value: Any, default: float) -> float:
        """Parse a float config value with a safe fallback."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning(
                "[PROVIDER] Invalid float config value %r; using default %s",
                value,
                default,
            )
            return default

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        coordinator: Any | None = None,
    ) -> None:
        """Initialise the provider with config and coordinator.

        Args:
            config: Provider configuration object.
            coordinator: Amplifier coordinator instance.
        """
        self.config = config or {}
        self.coordinator = coordinator

        # base_url: env var takes precedence over config, then default.
        self._base_url: str = os.environ.get(
            "CHAT_COMPLETIONS_BASE_URL",
            str(self.config.get("base_url", "http://localhost:8080/v1")),
        )

        # api_key: env var takes precedence over config, then "not-needed".
        # Empty string is rejected by the OpenAI client library, so we use
        # "not-needed" as a safe placeholder for local/keyless deployments.
        self._api_key: str = (
            os.environ.get("CHAT_COMPLETIONS_API_KEY")
            or self.config.get("api_key")
            or "not-needed"
        )

        self._model: str = str(self.config.get("model", "default"))
        self._client: openai.AsyncOpenAI | None = None
        self._timeout: float = self._config_float(
            self.config.get("timeout", 300.0), 300.0
        )
        self._temperature: float = self._config_float(
            self.config.get("temperature", 0.7), 0.7
        )
        self._max_tokens: int = self._config_int(
            self.config.get("max_tokens", 4096), 4096
        )
        self._max_retries: int = self._config_int(self.config.get("max_retries", 3), 3)
        self._min_retry_delay: float = self._config_float(
            self.config.get("min_retry_delay", 1.0), 1.0
        )
        self._max_retry_delay: float = self._config_float(
            self.config.get("max_retry_delay", 30.0), 30.0
        )
        self._repaired_tool_ids: set[str] = set()
        self._use_streaming: bool = self._config_bool(
            self.config.get("use_streaming", True)
        )

    def _translate_error(self, exc: Exception) -> KernelLLMError:
        """Translate an OpenAI SDK exception to a kernel error type.

        Maps OpenAI SDK exceptions to the shared kernel error vocabulary so that
        downstream code can catch rate limits, auth failures, etc. without
        provider-specific knowledge.

        Args:
            exc: The original exception from the OpenAI SDK or asyncio.

        Returns:
            A KernelLLMError subclass with provider, model, and retryable set.
            The __cause__ attribute is set to the original exception.
        """
        provider = self.name
        model = self._model
        err: KernelLLMError

        # Check specific OpenAI error types before the broader APIStatusError,
        # since many specific errors (RateLimitError, AuthenticationError, etc.)
        # are subclasses of APIStatusError.
        if isinstance(exc, openai.APITimeoutError):
            err = KernelLLMTimeoutError(
                str(exc),
                provider=provider,
                model=model,
                retryable=True,
            )
        elif isinstance(exc, openai.APIConnectionError):
            err = KernelProviderUnavailableError(
                str(exc),
                provider=provider,
                model=model,
                retryable=True,
            )
        elif isinstance(exc, openai.RateLimitError):
            err = KernelRateLimitError(
                str(exc),
                provider=provider,
                model=model,
                retryable=True,
            )
        elif isinstance(exc, openai.BadRequestError):
            msg = str(exc).lower()
            if "context length" in msg or "too many tokens" in msg:
                err = KernelContextLengthError(
                    str(exc),
                    provider=provider,
                    model=model,
                    retryable=False,
                )
            elif "content filter" in msg or "safety" in msg or "blocked" in msg:
                err = KernelContentFilterError(
                    str(exc),
                    provider=provider,
                    model=model,
                    retryable=False,
                )
            else:
                err = KernelInvalidRequestError(
                    str(exc),
                    provider=provider,
                    model=model,
                    retryable=False,
                )
        elif isinstance(exc, openai.AuthenticationError):
            err = KernelAuthenticationError(
                str(exc),
                provider=provider,
                model=model,
                retryable=False,
            )
        elif isinstance(exc, openai.PermissionDeniedError):
            err = KernelAccessDeniedError(
                str(exc),
                provider=provider,
                model=model,
                retryable=False,
            )
        elif isinstance(exc, openai.NotFoundError):
            err = KernelNotFoundError(
                str(exc),
                provider=provider,
                model=model,
                retryable=False,
            )
        elif isinstance(exc, openai.APIStatusError):
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is not None and 500 <= status_code < 600:
                err = KernelProviderUnavailableError(
                    str(exc),
                    provider=provider,
                    model=model,
                    status_code=status_code,
                    retryable=True,
                )
            else:
                err = KernelLLMError(
                    str(exc),
                    provider=provider,
                    model=model,
                )
        elif isinstance(exc, asyncio.TimeoutError):
            err = KernelLLMTimeoutError(
                str(exc),
                provider=provider,
                model=model,
                retryable=True,
            )
        else:
            err = KernelLLMError(
                str(exc),
                provider=provider,
                model=model,
                retryable=True,
            )

        err.__cause__ = exc
        return err

    async def _emit_event(self, name: str, payload: dict[str, Any]) -> None:
        """Emit an observability event via the coordinator's hooks, if available.

        Safely checks that self.coordinator is not None and has a hooks
        attribute before delegating to coordinator.hooks.emit.  This allows
        the provider to be used without a coordinator (e.g. in unit tests)
        without crashing.

        Args:
            name: The event name (e.g. 'llm:request', 'llm:response').
            payload: The event payload dict.
        """
        if self.coordinator is not None and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(name, payload)

    async def _repair_tool_sequence(self, messages: list[Message]) -> list[Message]:
        """Detect orphaned tool calls and inject synthetic tool results.

        Scans the message list for assistant messages containing ToolCallBlock
        entries whose IDs have no corresponding tool-role result message.
        For each orphaned ID that has not already been repaired, a synthetic
        error result message is injected immediately after the assistant message.

        Already-repaired IDs are tracked in ``self._repaired_tool_ids`` so
        that repeated calls with the same messages do not produce duplicate
        injections.

        Emits a ``provider:tool_sequence_repaired`` event when at least one
        repair is performed.

        Args:
            messages: Internal message list to inspect and (possibly) repair.

        Returns:
            The (possibly modified) message list with synthetic results injected.
        """
        # Collect all tool_call_ids that already have a matching result message.
        existing_result_ids: set[str] = set()
        for msg in messages:
            if msg.role == "tool":
                if msg.tool_call_id:
                    existing_result_ids.add(msg.tool_call_id)
                if isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, ToolResultBlock):
                            existing_result_ids.add(block.tool_call_id)

        result: list[Message] = []
        repaired_tool_ids: list[str] = []

        for msg in messages:
            result.append(msg)

            # Only assistant messages can carry ToolCallBlocks.
            if msg.role != "assistant" or not isinstance(msg.content, list):
                continue

            for block in msg.content:
                if not isinstance(block, ToolCallBlock):
                    continue
                tool_id = block.id
                if tool_id in existing_result_ids or tool_id in self._repaired_tool_ids:
                    continue
                # Orphaned tool call — inject a synthetic error result.
                synthetic = Message(
                    role="tool",
                    tool_call_id=tool_id,
                    content=(
                        "[ERROR] Tool result missing - this tool call was not executed."
                    ),
                )
                result.append(synthetic)
                self._repaired_tool_ids.add(tool_id)
                repaired_tool_ids.append(tool_id)

        if repaired_tool_ids:
            if self.coordinator is not None and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "model": self._model,
                        "repaired_count": len(repaired_tool_ids),
                        "repaired_tool_ids": repaired_tool_ids,
                    },
                )

        return result

    def _convert_messages_to_wire(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Convert internal Message list to OpenAI-compatible wire format.

        Maps roles, joins text blocks, drops thinking blocks, converts tool
        call blocks to the ``tool_calls`` array, and converts tool result blocks
        to ``role: 'tool'`` messages.  Images are converted to ``image_url``
        content items.

        Args:
            messages: Internal message list from the ChatRequest.

        Returns:
            List of dicts suitable for ``client.chat.completions.create``.
        """
        wire: list[dict[str, Any]] = []

        # Prepend developer messages (as system) before user messages in wire format.
        ordered = [m for m in messages if m.role == "developer"] + [
            m for m in messages if m.role != "developer"
        ]

        for message in ordered:
            role = "system" if message.role == "developer" else message.role
            content = message.content

            if isinstance(content, str):
                wire.append({"role": role, "content": content})
                continue

            # List of content blocks — iterate and classify.
            text_parts: list[str] = []
            tool_calls_wire: list[dict[str, Any]] = []
            image_parts: list[dict[str, Any]] = []
            tool_result_block: ToolResultBlock | None = None

            for block in content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingBlock):
                    pass  # Silently drop thinking blocks
                elif isinstance(block, ToolCallBlock):
                    tool_calls_wire.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    tool_result_block = block
                elif isinstance(block, ImageBlock):
                    src = block.source
                    url = f"data:{src['media_type']};base64,{src['data']}"
                    image_parts.append({"type": "image_url", "image_url": {"url": url}})

            if tool_result_block is not None:
                # ToolResultBlock overrides everything else in this message.
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result_block.tool_call_id,
                        "content": str(tool_result_block.output),
                    }
                )
                continue

            msg: dict[str, Any] = {"role": role}

            if image_parts:
                # Build a multimodal content array.
                content_array: list[dict[str, Any]] = []
                if text_parts:
                    content_array.append(
                        {"type": "text", "text": "\n".join(text_parts)}
                    )
                content_array.extend(image_parts)
                msg["content"] = content_array
            elif text_parts:
                msg["content"] = "\n".join(text_parts)
            else:
                msg["content"] = None

            if tool_calls_wire:
                msg["tool_calls"] = tool_calls_wire

            wire.append(msg)

        return wire

    def _convert_tools_to_wire(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Convert internal ToolSpec list to OpenAI function-calling wire format.

        Args:
            tools: List of ToolSpec from the ChatRequest.

        Returns:
            List of ``{type: 'function', function: {name, description, parameters}}``
            dicts.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _build_response(self, response: Any) -> ChatResponse:
        """Build a ChatResponse from an OpenAI ChatCompletion object.

        Args:
            response: The ChatCompletion returned by the OpenAI SDK.

        Returns:
            A ChatResponse with content blocks, tool calls, and usage.
        """
        choice = response.choices[0]
        message = choice.message

        content: list[Any] = []

        # Preserve any extended reasoning content as a ThinkingBlock.
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            content.append(ThinkingBlock(thinking=reasoning_content))

        # Text content → TextBlock.
        if message.content:
            content.append(TextBlock(text=message.content))

        # Tool calls → ToolCallBlock in content + ToolCall in tool_calls.
        tool_calls: list[ToolCall] | None = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                arguments = json.loads(tc.function.arguments)
                content.append(
                    ToolCallBlock(
                        id=tc.id,
                        name=tc.function.name,
                        input=arguments,
                    )
                )
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        # Usage mapping: OpenAI prompt/completion → kernel input/output.
        usage: Usage | None = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    async def _complete_non_streaming(
        self,
        wire_messages: list[dict[str, Any]],
        wire_tools: list[dict[str, Any]] | None,
        request: "ChatRequest",
    ) -> "ChatResponse":
        """Execute a non-streaming chat completion and return a ChatResponse.

        Args:
            wire_messages: Messages already converted to OpenAI wire format.
            wire_tools: Tools already converted to OpenAI wire format, or None.
            request: The original ChatRequest (used to resolve model).

        Returns:
            A ChatResponse built from the completed API response.
        """
        model = request.model or self._model
        response = await self._client.chat.completions.create(  # type: ignore[union-attr]
            model=model,
            messages=wire_messages,  # type: ignore[arg-type]
            tools=wire_tools,  # type: ignore[arg-type]
            stream=False,
        )
        return self._build_response(response)

    async def _complete_streaming(
        self,
        wire_messages: list[dict[str, Any]],
        wire_tools: list[dict[str, Any]] | None,
        request: "ChatRequest",
    ) -> "ChatResponse":
        """Execute a streaming chat completion and accumulate chunks into a ChatResponse.

        Calls the API with ``stream=True``, iterates the async chunk stream, and
        accumulates:
        - ``delta.content`` into a text buffer
        - ``delta.tool_calls`` by index (first chunk per index carries ``id`` and
          ``function.name``; subsequent chunks append to ``function.arguments``)
        - ``reasoning_content`` into a thinking buffer (via ``getattr`` for safety)
        - ``finish_reason`` from the last non-empty choice
        - ``usage`` from the final chunk if present

        Args:
            wire_messages: Messages already converted to OpenAI wire format.
            wire_tools: Tools already converted to OpenAI wire format, or None.
            request: The original ChatRequest (used to resolve model).

        Returns:
            A ChatResponse built from the accumulated streaming data.
        """
        model = request.model or self._model

        text_buffer: str = ""
        thinking_buffer: str = ""
        # Maps chunk index -> accumulated tool call data
        tool_call_accum: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: Any = None

        stream = await self._client.chat.completions.create(  # type: ignore[union-attr]
            model=model,
            messages=wire_messages,  # type: ignore[arg-type]
            tools=wire_tools,  # type: ignore[arg-type]
            stream=True,
        )

        async for chunk in stream:
            # Capture usage if present on any chunk (typically the final one).
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Track finish_reason from the last chunk that has one.
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            # Accumulate text content.
            if delta.content:
                text_buffer += delta.content

            # Accumulate reasoning/thinking content (provider-specific extension).
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                thinking_buffer += reasoning

            # Accumulate tool call deltas by index.
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_accum:
                        # First chunk for this index: capture id and name.
                        tool_call_accum[idx] = {
                            "id": tc_delta.id,
                            "name": tc_delta.function.name,
                            "arguments": "",
                        }
                    # All chunks: append to arguments.
                    if tc_delta.function.arguments:
                        tool_call_accum[idx]["arguments"] += tc_delta.function.arguments

        # Build content blocks from accumulated buffers.
        content: list[Any] = []

        if thinking_buffer:
            content.append(ThinkingBlock(thinking=thinking_buffer))

        if text_buffer:
            content.append(TextBlock(text=text_buffer))

        # Build tool calls from accumulated data (ordered by index).
        tool_calls: list[ToolCall] | None = None
        if tool_call_accum:
            tool_calls = []
            for idx in sorted(tool_call_accum.keys()):
                tc_data = tool_call_accum[idx]
                arguments = json.loads(tc_data["arguments"])
                content.append(
                    ToolCallBlock(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        input=arguments,
                    )
                )
                tool_calls.append(
                    ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=arguments,
                    )
                )

        # Map usage if captured from the stream.
        usage_obj: Usage | None = None
        if usage is not None:
            usage_obj = Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
            )

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage_obj,
            finish_reason=finish_reason,
        )

    def _ensure_client(self) -> openai.AsyncOpenAI:
        """Return the AsyncOpenAI client, creating it lazily if not yet initialised.

        Returns:
            The AsyncOpenAI client instance.
        """
        if self._client is None:
            client_kwargs: dict[str, Any] = {}
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**client_kwargs)
        return self._client

    async def list_models(self) -> list[ModelInfo]:
        """Return a list of models available on the server.

        Queries the server's ``/v1/models`` endpoint via the OpenAI-compatible
        client.  Each model returned by the server is converted to a
        :class:`~amplifier_core.models.ModelInfo`.  If the request fails for any
        reason, a single fallback entry using the configured model name is
        returned so that callers always receive at least one usable model.

        Returns:
            A list of :class:`~amplifier_core.models.ModelInfo` objects.  On
            failure, returns a one-element list containing the configured model.
        """
        try:
            response = await self._ensure_client().models.list()
            return [
                ModelInfo(
                    id=model.id,
                    display_name=model.id,
                    context_window=0,
                    max_output_tokens=self._max_tokens,
                    capabilities=["tools", "streaming"],
                )
                for model in response.data
            ]
        except Exception as exc:
            logger.warning("Failed to list models from server: %s", exc)
            return [
                ModelInfo(
                    id=self._model,
                    display_name=self._model,
                    context_window=0,
                    max_output_tokens=self._max_tokens,
                )
            ]

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Send a chat request and return a ChatResponse.

        Lazily initialises the AsyncOpenAI client on the first call.  Wraps
        each individual attempt in ``asyncio.timeout`` and retries retryable
        errors with exponential backoff via ``retry_with_backoff``.

        Args:
            request: The unified ChatRequest to execute.

        Returns:
            A ChatResponse with content, tool calls, and usage.

        Raises:
            KernelLLMError subclass on any provider failure.
        """
        # Ensure the AsyncOpenAI client is initialised before the retry loop.
        self._ensure_client()

        messages = await self._repair_tool_sequence(request.messages)
        wire_messages = self._convert_messages_to_wire(messages)
        wire_tools = (
            self._convert_tools_to_wire(request.tools) if request.tools else None
        )

        retry_config = RetryConfig(
            max_retries=self._max_retries,
            initial_delay=self._min_retry_delay,
            max_delay=self._max_retry_delay,
        )

        async def _on_retry(attempt: int, delay: float, error: Any) -> None:
            if self.coordinator is not None and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:retry",
                    {
                        "provider": self.name,
                        "attempt": attempt,
                        "delay": delay,
                        "max_retries": self._max_retries,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

        async def _single_attempt() -> ChatResponse:
            start_time = time.monotonic()
            await self._emit_event(
                "llm:request",
                {
                    "provider": self.name,
                    "model": self._model,
                },
            )
            try:
                async with asyncio.timeout(self._timeout):
                    if self._use_streaming:
                        chat_response = await self._complete_streaming(
                            wire_messages, wire_tools, request
                        )
                    else:
                        chat_response = await self._complete_non_streaming(
                            wire_messages, wire_tools, request
                        )
            except Exception as exc:
                duration_ms = (time.monotonic() - start_time) * 1000
                kernel_error = self._translate_error(exc)
                await self._emit_event(
                    "llm:response",
                    {
                        "provider": self.name,
                        "status": "error",
                        "duration_ms": duration_ms,
                        "error_type": type(exc).__name__,
                    },
                )
                raise kernel_error from exc

            duration_ms = (time.monotonic() - start_time) * 1000
            usage_dict: dict[str, Any] = {}
            if chat_response.usage:
                usage_dict = {
                    "input_tokens": chat_response.usage.input_tokens,
                    "output_tokens": chat_response.usage.output_tokens,
                    "total_tokens": chat_response.usage.total_tokens,
                }
            await self._emit_event(
                "llm:response",
                {
                    "provider": self.name,
                    "usage": usage_dict,
                    "duration_ms": duration_ms,
                    "stop_reason": chat_response.finish_reason,
                },
            )
            return chat_response

        return await retry_with_backoff(
            _single_attempt, config=retry_config, on_retry=_on_retry
        )

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Extract tool calls from a ChatResponse.

        Args:
            response: The ChatResponse returned by complete().

        Returns:
            The list of ToolCall objects, or an empty list if none present.
        """
        return response.tool_calls or []

    def get_info(self) -> ProviderInfo:
        """Return metadata describing this provider's capabilities and configuration.

        Returns:
            A ProviderInfo instance with the provider's id, display name,
            credential environment variables, capabilities, defaults, and
            all 10 config field declarations.
        """
        return ProviderInfo(
            id="chat-completions",
            display_name="Chat Completions",
            credential_env_vars=["CHAT_COMPLETIONS_API_KEY"],
            capabilities=["tools", "streaming", "json_mode"],
            defaults={
                "model": self._model,
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 300.0,
            },
            config_fields=[
                ConfigField(
                    id="api_key",
                    display_name="API Key",
                    field_type="secret",
                    prompt="Enter your API key",
                    env_var="CHAT_COMPLETIONS_API_KEY",
                    required=False,
                ),
                ConfigField(
                    id="base_url",
                    display_name="Base URL",
                    field_type="text",
                    prompt="Enter the API base URL",
                    env_var="CHAT_COMPLETIONS_BASE_URL",
                    required=False,
                    default="http://localhost:8080/v1",
                ),
                ConfigField(
                    id="model",
                    display_name="Model",
                    field_type="text",
                    prompt="Enter the model name to use",
                    required=False,
                ),
                ConfigField(
                    id="max_tokens",
                    display_name="Max Tokens",
                    field_type="text",
                    prompt="Maximum number of tokens to generate",
                    required=False,
                    default="4096",
                ),
                ConfigField(
                    id="temperature",
                    display_name="Temperature",
                    field_type="text",
                    prompt="Sampling temperature (0.0-2.0)",
                    required=False,
                    default="0.7",
                ),
                ConfigField(
                    id="timeout",
                    display_name="Timeout",
                    field_type="text",
                    prompt="Request timeout in seconds",
                    required=False,
                    default="300.0",
                ),
                ConfigField(
                    id="max_retries",
                    display_name="Max Retries",
                    field_type="text",
                    prompt="Maximum number of retry attempts",
                    required=False,
                    default="3",
                ),
                ConfigField(
                    id="min_retry_delay",
                    display_name="Min Retry Delay",
                    field_type="text",
                    prompt="Minimum delay between retries in seconds",
                    required=False,
                    default="1.0",
                ),
                ConfigField(
                    id="max_retry_delay",
                    display_name="Max Retry Delay",
                    field_type="text",
                    prompt="Maximum delay between retries in seconds",
                    required=False,
                    default="30.0",
                ),
                ConfigField(
                    id="use_streaming",
                    display_name="Use Streaming",
                    field_type="boolean",
                    prompt="Enable streaming responses",
                    required=False,
                    default="true",
                ),
            ],
        )

    async def close(self) -> None:
        """Release any resources held by this provider.

        Uses asyncio.shield so the client cleanup completes even if the
        enclosing task is cancelled.  All exceptions are suppressed.
        """
        if self._client is not None:
            try:
                await asyncio.shield(self._client.close())
            except (asyncio.CancelledError, Exception):
                pass


async def mount(coordinator: Any, config: dict[str, Any] | None = None) -> Any:
    """Mount the chat-completions provider into the coordinator.

    Creates a ChatCompletionsProvider instance, registers it with the
    coordinator under the 'providers' namespace, logs that mounting
    succeeded, and returns a cleanup callable.

    Args:
        coordinator: Amplifier coordinator instance (first arg per amplifier-core convention).
        config: Provider configuration dict, or None for defaults.

    Returns:
        An async cleanup function that closes the provider.
    """
    provider = ChatCompletionsProvider(config=config, coordinator=coordinator)
    await coordinator.mount("providers", provider, name="chat-completions")
    logger.info("chat-completions provider mounted")

    async def cleanup() -> None:
        await provider.close()

    return cleanup
