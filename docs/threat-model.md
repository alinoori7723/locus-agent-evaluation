# Threat model and evaluation scope

The protected boundary is the **agent-facing tool interface**. The tests ask whether public
observations leak the designated fault and whether invalid tool requests can receive execution
credit. They do not protect the host from malicious code running inside the Python process.

| Failure mode | Implemented control | Evidence |
| --- | --- | --- |
| Label leaked in IDs, metadata, order, text, or budgets | Matched worlds with exact public-view equality | Five contaminated-renderer negative controls |
| Fault inferred from rejection messages | Same invalid-action transcript in both worlds | Error-path equivalence audit |
| Position or ticket hint predicts the answer | Complete fault × order × hint cross | Exact baseline results plus intentionally biased-fixture rejection |
| Correct answer submitted without required work | Evidence-bound repair and receipt-bound submission | Diagnosis-only control and premature-submission tests |
| Healthy, fabricated, stale, or misbound evidence | Current-event lookup, target check, revision check, failed-probe requirement | Dedicated rejection tests |
| Agent changes returned data to alter scoring | Copies of observations, responses, and histories | Mutation tests |
| Infinite rejected-action retry loop | Attempt budget consumed by both accepted and rejected requests | Budget and terminal-state tests |
| Agent failure mistaken for completion | Explicit terminal `agent_error` status | Factory and policy exception tests |
| Incorrect scores or edited report content | File digests and semantic trace replay | Tampering tests even after manifest recomputation |
| Missing or duplicated evaluation cases | Full per-agent scenario coverage check | Replay coverage tests |

The misleading ticket is a deliberately simple untrusted instruction. It exercises a prompt-
injection-like decision surface. The ticket-follower is a scripted failure control, not evidence
of a model vulnerability; a live integration smoke test is not a general injection-resistance study.

The optional OpenAI adapter sends only the public observation and tool schemas to the official API.
Its configured timeout, shared request budget, output-token limit, and disabled retries bound specific
parts of the workload. An HTTP/connection/timeout error stops further attempts for the run. These are
not a process sandbox or a guaranteed dollar spending cap. Secrets and raw exception bodies are not
part of the provider trace. See [SECURITY.md](../SECURITY.md).

## Outside the boundary

- A local adapter can read source files, inspect Python objects, access the network, or retain global
  state. Fixture answers are shipped openly. There is no hidden test server or secret answer vault.
- An action-count limit does not bound wall time, CPU, memory, recursive serialization work, or
  provider spend. The JSON byte limit applies after serialization. Hostile resource exhaustion is
  not covered; `act()` must be isolated externally before accepting untrusted adapters.
- Matching public JSON says nothing about timing, memory, process inspection, external feedback,
  or training-data contamination. The equality guarantee covers declared observation bytes.
- Error-path tests exercise the listed paths, not every possible program or side channel.
- Hash manifests are unsigned consistency records. Someone who changes data, scores, code, and
  manifests coherently can produce a different internally consistent report.
- Replay checks recorded actions, not the identity or internal reasoning of the policy that chose
  them. An `agent_error` event can be represented but its external cause cannot be reconstructed.
- The simulator does not execute real repairs and does not model concurrent actors, persistent
  infrastructure, distributed failures, or rollback.

## Interpreting a pass

A regression pass means that the current implementation satisfies the declared finite fixture
checks and the four baseline expectations. A successful episode demonstrates compliance with the
specified observable workflow. Neither result proves that an agent used a particular internal
reasoning process, that every shortcut is excluded, or that behavior will transfer to real systems.
