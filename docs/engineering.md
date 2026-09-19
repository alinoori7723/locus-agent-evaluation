# Engineering walkthrough

This repository is a compact example of building an evaluation system whose scores can be inspected
and challenged. The fastest review path is to run the demo, compare the diagnostic policies, and
follow one episode through its validation and replay code.

## A five-minute review

1. Read the [sample report](../reports/sample/report.md). `diagnosis-only` names every fault correctly
   but earns zero verified successes. This separates answer quality from execution validity.
2. Open [`Environment.step`](../src/locus_eval/environment.py). Each request is normalized,
   validated, executed through a fixed tool set, and recorded. Rejections are budgeted events.
3. Read [`test_leakage.py`](../tests/test_leakage.py). Deliberately leak the fault through an ID,
   hint, ordering, budget, or extra field and confirm that the guard fails.
4. Inspect [`test_reporting.py`](../tests/test_reporting.py). Recompute a file manifest after
   altering a response or score; replay still rejects the semantic inconsistency.
5. Run the [custom adapter](../examples/custom_agent.py), then replay its report. The interface is
   small enough to replace with a provider adapter without changing the environment or scorer.

## Capabilities demonstrated in code

| Engineering concern | Concrete artifact |
| --- | --- |
| Evaluation design | Separate answer match, completion, strict success, false credit, and agent errors |
| Tool protocol design | Exact argument schemas, explicit rejection codes, and evidence provenance |
| Adversarial testing | Contaminated observations, incorrect evidence, premature actions, forged reports |
| Reproducibility | Canonical JSON, stable fixture expansion, source/data digests, and replay |
| Regression control | Positive and negative policy controls with exact finite-suite expectations |
| Packaging | Installable zero-dependency runtime, CLI, wheel verification outside the checkout |
| Model integration | Optional OpenAI SDK, native tool calls, request limits, failure telemetry, and offline provider-trace replay |
| Delivery | Pinned development tools, lint/format checks, coverage gate, Linux/Windows CI |
| Technical communication | Executable quickstart, sample traces, architecture, contract, and threat model |

## Decisions worth discussing

**Why count a recovered episode as completed but not strictly successful after a rejection?**
Because these metrics answer different questions. Recovery matters operationally; path validity
matters when evaluating tool constraints. Reporting both avoids hiding the tradeoff in one number.

**Why include a policy that already knows how to succeed?**
The diagnostic control checks that the environment is solvable and the scorer can recognize the
expected path. It is a software control, not a competitive model baseline.

**Why test contaminated fixtures?**
A guard that only passes clean data could be vacuous. Negative controls show that the implemented
checks detect specific leaks and biased nuisance distributions.

**Why publish full traces?**
An aggregate can conceal illegal steps, missing cases, or scoring mistakes. A trace lets a reviewer
verify every transition and reproduce the aggregate without trusting the original agent process.

**How do real models enter the evaluation?**
The optional OpenAI adapter sends the public state and four function schemas to the Responses API.
It uses an explicit request budget, no retries, and a configured timeout. The response supplies an
action; the same deterministic environment validates and scores it. SDK transport tests exercise
serialization and failure paths without claiming live inference.

Production extensions still need process isolation, explicit monetary budgets, and new held-out
task families. Preserve the observation/ground-truth boundary and distinguish provider failures
from policy errors and evaluator bugs. Those extensions are not claimed as implemented.
