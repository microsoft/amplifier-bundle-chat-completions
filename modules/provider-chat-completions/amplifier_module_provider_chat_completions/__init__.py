"""Amplifier provider module for OpenAI-compatible chat completions.

This module mounts a ChatCompletionsProvider into the Amplifier coordinator,
making it available as the 'chat-completions' provider.
"""

import logging

__all__ = ["mount", "ChatCompletionsProvider"]
__amplifier_module_type__ = "provider"

logger = logging.getLogger(__name__)


class ChatCompletionsProvider:
    """Provider for OpenAI-compatible chat completions API."""

    name = "chat-completions"

    def __init__(self, config, coordinator):
        """Initialise the provider with config and coordinator.

        Args:
            config: Provider configuration object.
            coordinator: Amplifier coordinator instance.
        """
        self.config = config
        self.coordinator = coordinator

    async def close(self):
        """Release any resources held by this provider."""


async def mount(config, coordinator):
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

    async def cleanup():
        await provider.close()

    return cleanup
