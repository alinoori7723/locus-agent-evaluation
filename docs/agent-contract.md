# Agent contract

An agent implements `act(observation: dict) -> JSON-compatible value`. A factory constructs a fresh
instance for each scenario. It receives only public observations; it must keep any model conversation
or adapter state within that instance.

```python
from locus_eval.agents import DiagnosticAgent
from locus_eval.runner import evaluate
from locus_eval.scenarios import build_suite

episodes = evaluate(build_suite(17), {"my-agent": DiagnosticAgent})
```

For command-line loading, expose a no-argument factory or class from an importable module:

```bash
locus-eval run --agent my_package.agent:MyAgent --seed 17 --output artifacts/my-agent
```

`examples/custom_agent.py` is a complete stateful adapter example. `DiagnosticAgent` demonstrates
reconstructing the next action solely from the public history.

For an adapter in the current checkout rather than an installed package, use
`python -m locus_eval run --agent examples.custom_agent:ExampleAgent` from the repository root.
The module invocation makes that directory importable.

## Observation

| Field | Content |
| --- | --- |
| `protocol_version` | Currently `1.0` |
| `task` | Incident ID, summary, two candidate components, initial failure status, untrusted ticket, objective |
| `tools` | Four tool names, descriptions, and JSON argument schemas |
| `budget` | Remaining action attempts and accepted probe calls |
| `state_revision` | Revision of the simulated system; increments after repair |
| `done` | Whether the episode is terminal |
| `history` | Prior normalized requests and responses, in execution order |

The incident ID is shared by the two matched worlds. A unique private case ID is used only in the
evaluator report. The task's `acceptance_status` describes the initial incident; `verify` supplies
the recovery result. Returned observations, responses, and histories are copies, so editing them
does not update the evaluator's state.

## Actions

Every action has exactly two fields, `tool` and `arguments`. Extra keys are rejected. Parameter
values must be strings of length 1–128; component values must name an available component.

| Tool | Required arguments | Preconditions | Output |
| --- | --- | --- | --- |
| `probe` | `component` | Probe budget remains | Input, expected/observed values, pass/fail, evidence ID, revision |
| `repair` | `component`, `evidence_id` | Existing current failed probe for this component | Repaired component and new revision |
| `verify` | None; use `{}` | A repair has occurred | Recovery result, verification ID, revision |
| `submit` | `component`, `verification_id` | Current successful verification and matching repair | `accepted: true`; episode closes |

Example request:

```json
{"tool": "probe", "arguments": {"component": "router"}}
```

Example response in the world with a router fault:

```json
{
  "ok": true,
  "result": {
    "component": "router",
    "evidence_id": "probe-1",
    "expected": "orders-primary",
    "input": "orders",
    "observed": "orders-missing",
    "passed": false,
    "state_revision": 0
  }
}
```

Example rejection:

```json
{"ok": false, "error": "VERIFICATION_REQUIRED"}
```

## Rejection codes

| Code | Meaning |
| --- | --- |
| `INVALID_ACTION` | Wrong envelope, non-JSON value, recursive value, non-finite number, or oversized request |
| `UNKNOWN_TOOL` | Tool is outside the four-tool allowlist |
| `INVALID_ARGUMENTS` | Missing/extra parameters or invalid string type/length |
| `UNKNOWN_COMPONENT` | Component is outside the scenario's public inventory |
| `PROBE_BUDGET_EXHAUSTED` | Two accepted probe calls already consumed |
| `UNKNOWN_EVIDENCE` | No probe with that ID exists in the current environment |
| `STALE_EVIDENCE` | Probe belongs to an earlier state revision |
| `EVIDENCE_TARGET_MISMATCH` | Probe was taken on another component |
| `EVIDENCE_NOT_DIAGNOSTIC` | Probe passed and cannot support repair |
| `REPAIR_REQUIRED` | Recovery verification requested before repair |
| `VERIFICATION_REQUIRED` | Submission attempted before a recovery check |
| `INVALID_VERIFICATION` | Receipt ID/revision does not match the current receipt |
| `RECOVERY_REQUIRED` | Verification failed; a defensive guard for future environments |
| `DIAGNOSIS_MISMATCH` | Submitted component differs from the repaired component |
| `EPISODE_CLOSED` | Submission accepted or action budget exhausted; no further event appended |

Evidence and verification IDs refer to entries held by this environment. They are deterministic
references, not secrets or cryptographic capabilities. A guessed ID is useful only if a matching
qualifying event has actually occurred.

## Errors, normalization, and exit status

The runner accepts dictionaries, not raw provider JSON text. A provider adapter must parse JSON
before returning. Requests are normalized through JSON serialization with a 4,096-byte encoded
limit. Nonserializable, recursive, non-finite, and oversized values become `null` in the trace and
receive `INVALID_ACTION`; their original payload is not retained. Other malformed JSON values
remain visible for replay. No rejected action receives system mutation privileges.

An exception in the factory or `act` ends the episode as `agent_error`. Raw exception text is not
copied into reports. Debug adapters separately when detailed provider diagnostics are needed.
Model/network timeouts and retries belong in the adapter; a blocking `act()` is not preempted by
the action budget. The runner does not retry provider calls on its own.

Exit `0` means the command completed and its fixture audits passed, not that every agent succeeded.
Exit `1` means an audit or agent execution failed. Exit `2` covers invalid input, a bad report, or
an import/file error. Custom-agent reports also include the separate bundled baseline audit;
passing that audit does not certify the custom agent.

Adapters are trusted local Python code. Do not use this interface as a hostile-code sandbox.

## Built-in OpenAI adapter

`locus-eval run --agent openai --model MODEL` selects the optional Responses API adapter. It returns
the model's function request to `Environment.step`; it has no separate success scorer. Configure the
SDK transport timeout with `--timeout`, the run-wide request budget with `--max-requests`, and each
generation's output limit with `--max-output-tokens`. SDK retries are disabled.

Provider failures retain the ordinary `agent_error` termination plus a sanitized `provider_error`
code. Codes distinguish authentication, rate limiting, timeout, connection/HTTP failure, incomplete
response, refusal, missing/multiple tool calls, invalid JSON, exhausted budget, and an open circuit.
The complete call configuration and minimal response telemetry are stored in `provider_trace`.
They are not included in the next public observation. See [the live example](live-evaluation.md).
