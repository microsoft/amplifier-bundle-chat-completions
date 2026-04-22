# amplifier-bundle-chat-completions

> [!WARNING]
> **DEPRECATED — This bundle is scheduled for retirement.**
>
> The `provider-chat-completions` module has been promoted to its own
> standalone repository:
> **[amplifier-module-provider-chat-completions](https://github.com/microsoft/amplifier-module-provider-chat-completions)**
>
> An upcoming Amplifier CLI release will pre-install it alongside the other
> well-known providers (anthropic, openai, azure-openai, gemini, ollama,
> vllm, github-copilot). Once that ships, you will no longer need this
> bundle — or any manual module registration — to use the provider.
>
> Until then, use the standalone module directly. See
> **[Migration](#migration)** below.
>
> This repository will remain in place for a short grace period so existing
> users are not broken immediately, then will be retired (archived on GitHub).

## What Changed?

Before: the `provider-chat-completions` Python module was hosted *inside* this
bundle at `modules/provider-chat-completions/`. Users had to `amplifier bundle
add` this repo AND `amplifier module add --source ...#subdirectory=...` to get
the provider available.

After: the module lives in its own top-level repository
(`amplifier-module-provider-chat-completions`) and is or will be pre-installed
by the Amplifier CLI, matching the shape of every other well-known provider.
No bundle required.

**Transparent behavior-level redirect during the grace period:** the
`behaviors/chat-completions` file in this bundle has been updated to pull the
provider from its new standalone location. If your bundle still `includes:`
this behavior, it will continue to work — you will silently pick up the
relocated module. Migrate to the new source URL at your convenience per the
[Bundle authors](#bundle-authors) section below.

## Migration

### End users

If you previously installed this bundle, remove it and switch to the
standalone module:

```bash
# 1. Remove the bundle (use whichever form matches how you added it):
amplifier bundle remove chat-completions
# or, if you added by URL:
amplifier bundle remove git+https://github.com/microsoft/amplifier-bundle-chat-completions@main

# 2. REQUIRED if you ever ran the old `module add` incantation:
#    this step clears the frozen `#subdirectory=...` URL from your
#    ~/.amplifier/settings.yaml. Skipping it means your CLI will keep
#    resolving to the (soon-to-be-retired) bundle subdirectory instead
#    of the new standalone module — even after this repo is archived.
amplifier module remove provider-chat-completions

# 3. Install the standalone module directly:
amplifier module add provider-chat-completions \
  --source git+https://github.com/microsoft/amplifier-module-provider-chat-completions@main

# 4. Configure as normal:
amplifier provider add
# Select "OpenAI-Compatible" (or "Chat Completions" on older CLI releases)
# and supply your base_url + model name.
```

Once an Amplifier CLI release that pre-installs `provider-chat-completions`
lands, step 3 becomes unnecessary — a plain `amplifier update` will be enough.
See [Amplifier release notes](https://github.com/microsoft/amplifier/releases)
for the version that adds it.

### Bundle authors

If your bundle currently composes this one, update your source URL:

```diff
  providers:
    - module: provider-chat-completions
-     source: git+https://github.com/microsoft/amplifier-bundle-chat-completions@main#subdirectory=modules/provider-chat-completions
+     source: git+https://github.com/microsoft/amplifier-module-provider-chat-completions@main
```

Or, once your consumers are on an Amplifier CLI release that pre-installs the
provider, drop the `source:` line entirely — the CLI's
`DEFAULT_PROVIDER_SOURCES` will resolve it.

If your bundle currently composes `behaviors/chat-completions` from this repo
via `includes:`, it will keep working during the grace period (thanks to the
transparent redirect described above) but will break when this repo is
archived. Update `includes:` to point at the new standalone module's source
URL directly, or compose the provider explicitly in your own `providers:`
block.

### If you do nothing

During the grace period: everything keeps working. The bundle still resolves,
the behavior YAML now points at the new standalone module, and existing user
installations continue to function.

After retirement (archival): `git clone` / git-based installs of this repo
will fail. Anyone who didn't migrate loses access to the provider until they
either (a) upgrade to an Amplifier CLI release that pre-installs it, or
(b) runs the manual `amplifier module add` command above pointing at the
standalone repo.

## Timeline

This repository will remain available during a short grace period while
consumers migrate. Specific retirement date will be announced on the
[new standalone repo](https://github.com/microsoft/amplifier-module-provider-chat-completions),
but plan on migrating within the next release cycle or two. **Do not treat
this as a stable long-term URL.**

## Configuration Reference

Configuration is unchanged from the bundled version — same module, same
config keys. See the standalone repo's
[README](https://github.com/microsoft/amplifier-module-provider-chat-completions#configuration)
for the full reference.

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
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.