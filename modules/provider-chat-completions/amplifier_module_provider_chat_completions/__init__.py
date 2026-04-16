"""Amplifier provider module for OpenAI-compatible chat completions.

This module mounts a ChatCompletionsProvider into the Amplifier coordinator,
making it available as the 'chat-completions' provider.
"""

import asyncio
import json
import logging
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
        self._model: str = str(self.config.get("model", ""))
        self._client: openai.AsyncOpenAI | None = None
        self._timeout: float = float(self.config.get("timeout", 60))
        self._max_tokens: int = int(self.config.get("max_tokens", 4096))
        self._max_retries: int = int(self.config.get("max_retries", 3))
        self._min_retry_delay: float = float(self.config.get("min_retry_delay", 1.0))
        self._max_retry_delay: float = float(self.config.get("max_retry_delay", 60.0))
        self._repaired_tool_ids: set[str] = set()

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

    def _ensure_client(self) -> openai.AsyncOpenAI:
        """Return the AsyncOpenAI client, creating it lazily if not yet initialised.

        Returns:
            The AsyncOpenAI client instance.
        """
        if self._client is None:
            client_kwargs: dict[str, Any] = {}
            if self.config.get("api_key"):
                client_kwargs["api_key"] = self.config["api_key"]
            if self.config.get("base_url"):
                client_kwargs["base_url"] = self.config["base_url"]
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

    async def complete(self, request: ChatRequest) -> ChatResponse:
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

        model = request.model or self._model

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
                    response = await self._client.chat.completions.create(  # type: ignore[union-attr]
                        model=model,
                        messages=wire_messages,  # type: ignore[arg-type]
                        tools=wire_tools,  # type: ignore[arg-type]
                        stream=False,
                    )
                chat_response = self._build_response(response)
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
                    default="60.0",
                ),
                ConfigField(
                    id="use_streaming",
                    display_name="Use Streaming",
                    field_type="boolean",
                    prompt="Enable streaming responses",
                    required=False,
                    default="false",
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


async def mount(config: Any, coordinator: Any) -> Any:
    """Mount the chat-completions provider into the coordinator.

    Creates a ChatCompletionsProvider instance, registers it with the
    coordinator under the 'providers' namespace, logs that mounting
    succeeded, and returns a cleanup callable.

    Args:
        config: Provider configuration object.
        coordinator: Amplifier coordinator instance.

    Returns:
        An async cleanup function that closes the provider.
    """
    provider = ChatCompletionsProvider(config, coordinator)
    await coordinator.mount("providers", provider, name="chat-completions")
    logger.info("chat-completions provider mounted")

    async def cleanup() -> None:
        await provider.close()

    return cleanup
