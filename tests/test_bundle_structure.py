"""Tests for Amplifier bundle structure files created in task-2.

Tests verify:
- Files exist at exact paths
- bundle.md has valid YAML frontmatter with correct metadata and includes
- behaviors/chat-completions.yaml has correct structure and provider source URL
- context/provider-instructions.md has complete config reference table with all 10 fields
"""

from pathlib import Path

import pytest
import yaml

BUNDLE_ROOT = Path(__file__).parent.parent


# ─── File existence tests ───────────────────────────────────────────────────


def test_bundle_md_exists():
    assert (BUNDLE_ROOT / "bundle.md").exists(), "bundle.md must exist"


def test_behavior_yaml_exists():
    assert (BUNDLE_ROOT / "behaviors" / "chat-completions.yaml").exists(), (
        "behaviors/chat-completions.yaml must exist"
    )


def test_provider_instructions_exists():
    assert (BUNDLE_ROOT / "context" / "provider-instructions.md").exists(), (
        "context/provider-instructions.md must exist"
    )


# ─── bundle.md tests ────────────────────────────────────────────────────────


def _parse_bundle_md_frontmatter() -> dict:
    """Extract and parse the YAML frontmatter from bundle.md."""
    content = (BUNDLE_ROOT / "bundle.md").read_text()
    # Must start with --- and have a closing ---
    assert content.startswith("---\n"), "bundle.md must start with YAML frontmatter (---)"
    end = content.index("\n---\n", 4)
    yaml_text = content[4:end]
    return yaml.safe_load(yaml_text)


def test_bundle_md_has_valid_yaml_frontmatter():
    data = _parse_bundle_md_frontmatter()
    assert data is not None


def test_bundle_md_bundle_name():
    data = _parse_bundle_md_frontmatter()
    assert data["bundle"]["name"] == "chat-completions", (
        "bundle.name must be 'chat-completions'"
    )


def test_bundle_md_bundle_version():
    data = _parse_bundle_md_frontmatter()
    assert data["bundle"]["version"] == "1.0.0", "bundle.version must be '1.0.0'"


def test_bundle_md_bundle_description():
    data = _parse_bundle_md_frontmatter()
    assert "description" in data["bundle"], "bundle.description must be present"
    assert len(data["bundle"]["description"]) > 0, "bundle.description must not be empty"


def test_bundle_md_includes_behavior():
    data = _parse_bundle_md_frontmatter()
    assert "includes" in data, "bundle.md must have an 'includes' section"
    includes = data["includes"]
    assert isinstance(includes, list), "includes must be a list"
    bundle_refs = [item["bundle"] for item in includes if isinstance(item, dict) and "bundle" in item]
    assert any("chat-completions:behaviors/chat-completions" in ref for ref in bundle_refs), (
        "includes must reference chat-completions:behaviors/chat-completions"
    )


def test_bundle_md_body_has_context_reference():
    content = (BUNDLE_ROOT / "bundle.md").read_text()
    assert "@chat-completions:context/provider-instructions.md" in content, (
        "bundle.md body must reference @chat-completions:context/provider-instructions.md"
    )


# ─── behaviors/chat-completions.yaml tests ──────────────────────────────────


def _parse_behavior_yaml() -> dict:
    return yaml.safe_load((BUNDLE_ROOT / "behaviors" / "chat-completions.yaml").read_text())


def test_behavior_yaml_is_valid_yaml():
    data = _parse_behavior_yaml()
    assert data is not None


def test_behavior_yaml_bundle_name():
    data = _parse_behavior_yaml()
    assert data["bundle"]["name"] == "chat-completions-behavior"


def test_behavior_yaml_bundle_version():
    data = _parse_behavior_yaml()
    assert data["bundle"]["version"] == "1.0.0"


def test_behavior_yaml_bundle_description():
    data = _parse_behavior_yaml()
    assert "description" in data["bundle"]


def test_behavior_yaml_has_providers():
    data = _parse_behavior_yaml()
    assert "providers" in data, "behaviors YAML must have a 'providers' section"
    assert isinstance(data["providers"], list)
    assert len(data["providers"]) >= 1


def test_behavior_yaml_provider_module():
    data = _parse_behavior_yaml()
    provider = data["providers"][0]
    assert provider["module"] == "provider-chat-completions"


def test_behavior_yaml_provider_source_url():
    data = _parse_behavior_yaml()
    provider = data["providers"][0]
    assert "source" in provider, "provider must have a 'source' field"
    source = provider["source"]
    assert "github.com/microsoft/amplifier-bundle-chat-completions" in source, (
        "source must reference the correct GitHub repo"
    )
    assert "modules/provider-chat-completions" in source, (
        "source must reference subdirectory=modules/provider-chat-completions"
    )


# ─── context/provider-instructions.md tests ─────────────────────────────────


def _provider_instructions_content() -> str:
    return (BUNDLE_ROOT / "context" / "provider-instructions.md").read_text()


ALL_CONFIG_KEYS = [
    "base_url",
    "api_key",
    "model",
    "max_tokens",
    "temperature",
    "timeout",
    "max_retries",
    "min_retry_delay",
    "max_retry_delay",
    "use_streaming",
]


@pytest.mark.parametrize("config_key", ALL_CONFIG_KEYS)
def test_provider_instructions_has_config_key(config_key: str):
    content = _provider_instructions_content()
    assert config_key in content, (
        f"context/provider-instructions.md must reference config key '{config_key}'"
    )


def test_provider_instructions_has_config_table():
    content = _provider_instructions_content()
    # Should have a markdown table (pipe characters)
    assert "|" in content, "context/provider-instructions.md must contain a config reference table"


def test_provider_instructions_has_compatible_servers():
    content = _provider_instructions_content()
    # At least one of the known compatible servers should be mentioned
    servers = ["llama-server", "vLLM", "SGLang", "LocalAI", "LM Studio"]
    assert any(srv in content for srv in servers), (
        "context/provider-instructions.md must list compatible servers"
    )


def test_provider_instructions_has_example_config():
    content = _provider_instructions_content()
    assert "provider-chat-completions" in content, (
        "context/provider-instructions.md must include an example config with provider-chat-completions"
    )
