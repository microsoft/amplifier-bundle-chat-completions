# Chat Completions Provider

This session uses `provider-chat-completions` to communicate with a local or remote server implementing the OpenAI Chat Completions API.

## Configuration Reference

| Config Key | Type | Default | Env Var | Purpose |
|-----------|------|---------|---------|---------|
| `base_url` | text | `http://localhost:8080/v1` | `CHAT_COMPLETIONS_BASE_URL` | Server endpoint |
| `api_key` | secret | `""` (empty) | `CHAT_COMPLETIONS_API_KEY` | API key. Empty = no auth header. |
| `model` | text | (required) | — | Model name to pass in requests |
| `max_tokens` | int | `4096` | — | Default max output tokens |
| `temperature` | float | `0.7` | — | Default temperature |
| `timeout` | float | `300.0` | — | Per-request timeout in seconds |
| `max_retries` | int | `3` | — | Retry attempts for transient failures |
| `min_retry_delay` | float | `1.0` | — | Base backoff delay |
| `max_retry_delay` | float | `30.0` | — | Backoff cap |
| `use_streaming` | bool | `true` | — | Use streaming API |

## Compatible Servers

Works with any server that implements `/v1/chat/completions`:

- llama-server (llama.cpp)
- vLLM (Chat Completions mode)
- SGLang
- LocalAI
- LM Studio
- text-generation-inference

## Example Behavior Config

```yaml
providers:
  - module: provider-chat-completions
    source: git+https://github.com/microsoft/amplifier-bundle-chat-completions@main#subdirectory=modules/provider-chat-completions
    config:
      base_url: http://my-server:8080/v1
      model: my-model-name
      max_tokens: 8192
      timeout: 600.0
```
