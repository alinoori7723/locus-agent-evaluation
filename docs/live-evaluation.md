# Live model evaluation

The optional OpenAI adapter runs real model inference against the same synthetic environment used
by the offline controls. The model selects actions; the local evaluator validates and scores them.
The adapter does not contain a diagnostic policy or inspect the private fault.

## Status of published evidence

The repository includes executable provider integration and network-free SDK transport tests.
No live provider result is committed yet. `reports/sample` contains scripted controls, and mocked
SDK responses in the test suite are software tests. They are not evidence of model performance.

## Install and configure

From a checkout with an activated virtual environment:

```bash
python -m pip install -e ".[openai]"
```

The package is not published to PyPI. Use the checkout command above, or install a built wheel with
its `[openai]` extra. The base wheel has no runtime dependencies; the extra installs the OpenAI SDK.
Development requirements include the SDK to exercise its real serialization through a mock transport.

Supply `OPENAI_API_KEY` through your local secret manager or shell environment. Do not paste a key
into a command saved in shell history. One PowerShell option is a hidden prompt:

```powershell
$env:OPENAI_API_KEY = Read-Host -MaskInput 'OpenAI API key'
```

In Bash:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
```

The adapter pins the official `https://api.openai.com/v1` endpoint. It does not inherit an alternative
endpoint from `OPENAI_BASE_URL`. The API account must have access to the exact requested model;
there is no silent model substitution or provider fallback.

## Bounded smoke test

```bash
locus-eval run --agent openai --model gpt-5.6 --limit 2 --max-requests 16 --max-output-tokens 1024 --timeout 30 --output artifacts/live-eval
locus-eval replay artifacts/live-eval
```

The same entry point is available as `python examples/openai_agent.py --model gpt-5.6 ...`.
The output directory must be new or empty so a paid run cannot overwrite an earlier result.

| Option | Default | Scope |
| --- | ---: | --- |
| `--model` | Required | Exact provider model ID; requested and returned IDs are recorded |
| `--limit` | 24 | First N scenarios in the seed-defined order; recorded explicitly in the report |
| `--max-requests` | 192 | Shared across the whole run, including failed transport attempts |
| `--max-output-tokens` | 1024 | Output ceiling per model request, including reasoning where applicable |
| `--timeout` | 30 seconds | SDK transport timeout; not an end-to-end process deadline |
| `--seed` | 17 | Fixture IDs and ordering, not a model sampling seed |

Each episode still permits at most eight action attempts and two accepted probes. SDK retries are
disabled. A transport/authentication/rate-limit error opens the shared circuit breaker: remaining
episodes receive an explicit provider failure without further requests. Budget exhaustion also
stops additional requests. Incomplete output, refusal, malformed JSON, and multiple tool calls end
the affected episode; they never become a guessed or repaired action.

No dollar budget is inferred from token limits. Failed requests can incur cost and may lack usage
metadata. Start with a small selection and inspect the report before expanding the run. This runner
does not resume interrupted paid runs; a process interruption may leave no finalized report.

## The actual API contract

The adapter uses the [Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
with [function calling](https://developers.openai.com/api/docs/guides/function-calling):
`tool_choice="required"`, `parallel_tool_calls=False`, and strict schemas for the four tools.
It sends a fixed instruction plus the complete current public observation on every request, with
`store=False`. It does not use previous response IDs, server-side conversations, filesystem tools,
or private evaluator state. Function arguments are parsed with duplicate-key and non-finite-number
rejection, then passed to the ordinary environment validator.

## Read the result

- `summary.json` records selected case IDs, deterministic metrics, and provider usage totals.
- `episodes.jsonl` includes the environment trace plus provider configuration, SDK version, request
  payload digests, response/request IDs, returned model IDs, elapsed time, and token counts.
- `report.md` presents both execution metrics and provider usage.
- `manifest.json` binds the exact artifact bytes.

Unknown token usage remains unknown and is counted separately; totals cover only reported usage.
Raw provider exception bodies, API keys, arbitrary response text, and reasoning content are omitted.
The action receipt preserves the normalized function request used by the evaluator.

Replay reconstructs the public observation before each action and checks its provider-request hash.
It also reconciles call/action counts, failure placement, token-count arithmetic, selected-case
coverage, and all deterministic scores. It makes no API calls and does not need the optional SDK.
Provider IDs and telemetry are unsigned records, not authenticated proof of inference. A fresh model
run can differ even when the fixture seed and model alias are unchanged.

## Interpreting the sample size

A two-case run proves an integration path only for the observed calls. It does not establish model
capability or a reliable success rate. A prefix selected by `--limit` need not be balanced or contain
complete pairs. The built-in leakage and shortcut audit still runs on the full 24-case fixture bank
and is reported separately from the model's selected episodes. These public fixtures are not a
hidden benchmark or a held-out scientific evaluation.
