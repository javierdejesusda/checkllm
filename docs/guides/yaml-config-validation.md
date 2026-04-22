# YAML / pyproject configuration validation

CheckLLM ships a JSON Schema (draft-07) describing every key it reads
from `[tool.checkllm]` in `pyproject.toml` and from the top-level
`checkllm.yaml` file. The schema lets IDEs surface inline tooltips and
red-underline typos before a single test runs.

## Why validate the config?

A surprising fraction of "why isn't my eval running?" issues trace back
to:

- A silent typo (`judgde_model` instead of `judge_model`).
- A value of the wrong type (`max_concurrency = "10"` as a string).
- A profile that shadows a top-level key nobody realised was there.

JSON Schema validation catches all three in under a millisecond and
without loading any code.

## Programmatic API

```python
from checkllm.config_schema import load_schema, validate_config

schema = load_schema()
errors = validate_config({
    "judge_model": "gpt-4o-mini",
    "default_threshold": 0.8,
    "max_concurrency": 16,
})
assert errors == []
```

`validate_config` returns a list of `ValidationError` records. Each has
a `path`, `message`, and `severity` (`"error"` or `"warning"`). An empty
list means the document is valid.

If you want to fail a CI job on any schema violation:

```python
from checkllm.config_schema import validate_config

problems = validate_config(cfg_dict)
if any(p.severity == "error" for p in problems):
    for p in problems:
        print(f"{p.severity.upper()} {p.path}: {p.message}")
    raise SystemExit(1)
```

The `jsonschema` dependency is optional. If it's missing,
`validate_config` returns a single installation-hint error instead of
crashing — so importing `config_schema` is always safe.

## IDE integration

The schema declares a stable `$id`:

```
https://checkllm.dev/schemas/checkllm.schema.json
```

To have VS Code / JetBrains / Neovim pick it up for `checkllm.yaml`,
point your `yaml.schemas` mapping at the bundled file:

```jsonc
// .vscode/settings.json
{
  "yaml.schemas": {
    "./node_modules/checkllm/schemas/checkllm.schema.json": "checkllm.yaml"
  }
}
```

For Python projects, you can extract the schema to the repo root so
editors find it without depending on a node install:

```bash
python -c "from checkllm.config_schema import generate_schema_to_file; \
  generate_schema_to_file('checkllm.schema.json')"
```

Then add the usual YAML language server comment to the top of your
file:

```yaml
# yaml-language-server: $schema=./checkllm.schema.json
judge_model: gpt-4o
default_threshold: 0.8
```

## Schema excerpt

The fields most users edit most often:

| Key | Type | Default | Notes |
|---|---|---|---|
| `judge_model` | string | `"gpt-4o"` | Model used by the default judge. |
| `judge_backend` | enum | `"auto"` | One of `auto`, `openai`, `anthropic`, `gemini`, `azure`, `ollama`, `litellm`, `deepseek`. |
| `default_threshold` | number (0..1) | `0.8` | Default pass/fail threshold for metric assertions. |
| `runs_per_test` | integer >= 1 | `1` | Repeats each test for variance estimation. |
| `max_concurrency` | integer >= 1 | `10` | Parallel judge-call cap. |
| `cache_enabled` | boolean | `true` | Turns the on-disk response cache on or off. |
| `cache_ttl_seconds` | integer >= 0 | `604800` | 0 means never expire. |
| `engine` | enum | `"auto"` | One of `auto`, `sync`, `async`, `thread`. |
| `budget` | number or null | `null` | Hard USD cap for judge calls. |

Open `src/checkllm/schemas/checkllm.schema.json` for the full schema source of truth.

## Regenerating the schema in CI

We keep the repo's bundled schema and the published copy in lockstep
via a tiny CI step:

```yaml
- name: Verify schema is in sync
  run: |
    python -c "from checkllm.config_schema import generate_schema_to_file; \
      generate_schema_to_file('build/checkllm.schema.json')"
    diff src/checkllm/schemas/checkllm.schema.json build/checkllm.schema.json
```

If the diff fails, regenerate the file locally and commit it.

## FAQ

**Q: Why draft-07 rather than 2020-12?**
A: Editor YAML plugins still default to draft-07. Upgrading requires
every consumer to opt in, and draft-07 covers everything our schema
needs.

**Q: Can I add my own keys under `[tool.checkllm]`?**
A: Yes — `additionalProperties` is `true` at the top level, so custom
keys are allowed. They'll show up as warnings if you explicitly run
with strict mode (planned) but never as errors today.

**Q: Do profiles get validated?**
A: The `profiles.*` sub-objects are free-form right now. We're tracking
stricter per-profile validation under issue #XYZ.
