# amplifier-bundle-chat-completions

Amplifier provider bundle for any server implementing the OpenAI Chat Completions API (`/v1/chat/completions`).

## Overview

This bundle provides `provider-chat-completions`, a provider module that works with:

- **llama-server** (llama.cpp)
- **vLLM** (Chat Completions mode)
- **SGLang**
- **LocalAI**
- **LM Studio**
- **text-generation-inference**
- Any other server speaking the OpenAI Chat Completions wire format

## Installation (end users)

If you have an OpenAI-compatible server running (llama-server, vLLM, etc.) and
just want to point Amplifier at it, run the following from a fresh shell:

```bash
# 1. Install Amplifier itself (skip if already installed)
uv tool install git+https://github.com/microsoft/amplifier@main

# 2. Register this bundle
amplifier bundle add git+https://github.com/microsoft/amplifier-bundle-chat-completions@main

# 3. Install the provider module so it shows up in the provider picker.
#    This step is required -- `bundle add` alone does not install the module.
amplifier module add provider-chat-completions \
  --source git+https://github.com/microsoft/amplifier-bundle-chat-completions@main#subdirectory=modules/provider-chat-completions

# 4. Configure the provider (base_url, model, api_key) interactively.
#    "Chat Completions" will appear in the list after step 3.
amplifier provider manage

# 5. Activate a functional bundle. This bundle is provider-only -- it has no
#    tools, agents, or orchestrator -- so you need a base bundle on top.
#    Pick one:
#      foundation             -- the full default stack
#      exp-lean-foundation    -- a minimal, token-efficient stack
amplifier bundle add git+https://github.com/microsoft/amplifier-foundation@main#subdirectory=experiments/exp-lean/exp-lean-foundation.md
amplifier bundle use exp-lean-foundation

# 6. Start a session
amplifier
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `CHAT_COMPLETIONS_BASE_URL` | Server endpoint (overrides `base_url` config) |
| `CHAT_COMPLETIONS_API_KEY` | API key (overrides `api_key` config) |

## Composition (bundle authors)

If you are building your own bundle and want to include `chat-completions` as
a dependency, compose via YAML includes -- users of your bundle do not need to
run the `bundle add` / `module add` steps above.

Include the whole bundle:

```yaml
includes:
  - bundle: git+https://github.com/microsoft/amplifier-bundle-chat-completions@main
```

Or include just the behavior:

```yaml
includes:
  - bundle: git+https://github.com/microsoft/amplifier-bundle-chat-completions@main#behaviors/chat-completions
```

Then configure the provider in your own behavior YAML:

```yaml
providers:
  - module: provider-chat-completions
    source: git+https://github.com/microsoft/amplifier-bundle-chat-completions@main#subdirectory=modules/provider-chat-completions
    config:
      base_url: http://localhost:8080/v1
      model: gemma26-long
```

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
