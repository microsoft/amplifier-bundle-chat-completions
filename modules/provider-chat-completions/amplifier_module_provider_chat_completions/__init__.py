"""Amplifier provider module for OpenAI-compatible chat completions.

This module mounts a ChatCompletionsProvider into the Amplifier coordinator,
making it available as the 'chat-completions' provider.
"""

import asyncio
import json
import logging
from typing import Any

import openai

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
        provider = "chat-completions"
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

        for message in messages:
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

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request and return a ChatResponse.

        Lazily initialises the AsyncOpenAI client on the first call.  Wraps
        the API call in ``asyncio.timeout`` and translates any exception via
        ``_translate_error``.

        Args:
            request: The unified ChatRequest to execute.

        Returns:
            A ChatResponse with content, tool calls, and usage.

        Raises:
            KernelLLMError subclass on any provider failure.
        """
        # Lazy-initialize the AsyncOpenAI client.
        if self._client is None:
            client_kwargs: dict[str, Any] = {}
            if self.config.get("api_key"):
                client_kwargs["api_key"] = self.config["api_key"]
            if self.config.get("base_url"):
                client_kwargs["base_url"] = self.config["base_url"]
            self._client = openai.AsyncOpenAI(**client_kwargs)

        wire_messages = self._convert_messages_to_wire(request.messages)
        wire_tools = (
            self._convert_tools_to_wire(request.tools) if request.tools else None
        )

        model = request.model or self._model
        timeout = request.timeout or float(self.config.get("timeout", 60))

        try:
            async with asyncio.timeout(timeout):
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=wire_messages,  # type: ignore[arg-type]
                    tools=wire_tools,  # type: ignore[arg-type]
                    stream=False,
                )
            return self._build_response(response)
        except Exception as exc:
            raise self._translate_error(exc) from exc

    async def close(self) -> None:
        """Release any resources held by this provider."""
        if self._client is not None:
            await self._client.close()
            self._client = None


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
