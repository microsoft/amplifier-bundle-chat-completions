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

## Usage

Include this bundle in your Amplifier configuration:

```yaml
includes:
  - bundle: git+https://github.com/microsoft/amplifier-bundle-chat-completions@main
```

Or include the behavior directly:

```yaml
includes:
  - bundle: git+https://github.com/microsoft/amplifier-bundle-chat-completions@main#behaviors/chat-completions
```

### Configuration

In your behavior YAML:

```yaml
providers:
  - module: provider-chat-completions
    source: git+https://github.com/microsoft/amplifier-bundle-chat-completions@main#subdirectory=modules/provider-chat-completions
    config:
      base_url: http://localhost:8080/v1
      model: gemma26-long
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `CHAT_COMPLETIONS_BASE_URL` | Server endpoint (overrides `base_url` config) |
| `CHAT_COMPLETIONS_API_KEY` | API key (overrides `api_key` config) |

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
