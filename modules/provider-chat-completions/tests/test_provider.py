"""Tests for amplifier_module_provider_chat_completions module skeleton.

Tests verify:
- Module metadata (__amplifier_module_type__, __all__ exports)
- mount() function is callable
- mount() returns a coroutine (async function)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import amplifier_module_provider_chat_completions as module


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
