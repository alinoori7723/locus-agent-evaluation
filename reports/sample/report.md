# Agent evaluation report

Deterministic synthetic fixtures. Trace replay does not attest agent provenance.

Seed: `17` · Selected scenarios: **24/24** · Episodes: **96**

| Agent | Answer match | Verified success | False credit | Invalid actions |
| --- | ---: | ---: | ---: | ---: |
| diagnosis-only | 100% | 0% | 24/24 | 156 |
| diagnostic | 100% | 100% | 0/24 | 0 |
| first-candidate | 50% | 0% | 12/24 | 192 |
| ticket-follower | 50% | 0% | 12/24 | 192 |

Answer match scores the last attempted diagnosis, even if rejected. Verified success
requires an accepted diagnosis after evidence-backed repair and recovery
verification, with zero rejected actions and no agent error.
False credit is an answer match without verified success.
A recovered episode with earlier rejected actions is reported as
completed but does not pass the strict success metric.

Rejected actions consume the action budget. Shortcut agents keep submitting until
that budget is exhausted; invalid actions are counts, not independent trials.

## Regression checks

- observations: **PASS**
- error_paths: **PASS**
- shortcuts: **PASS**

## Reproduce and inspect

```bash
locus-eval demo --seed 17 --output artifacts/reproduced
locus-eval replay artifacts/reproduced
```

The demo command runs the four bundled scripted policies. For a custom-agent report,
rerun `locus-eval run --agent module:factory` with the original adapter and seed.
`episodes.jsonl` records normalized requests, tool responses, rejections, and scores.
`manifest.json` binds the exact report bytes. Replay reruns recorded tool actions,
reconciles scores and aggregates, and checks the suite and implementation digests.
It cannot re-execute or authenticate an external agent's decision process.

Suite SHA-256: `91b501d660dbd976416bc38378e36e82ac1bdb03c19168bf2df97104a27169b3`

Implementation SHA-256: `830948a99e8667256febbe8daf2c67eac16449c1c7d4bed8c6a5f6bf3450c8fd`

These checks cover declared fixture properties and baselines. They do not establish
general shortcut immunity, production security, reasoning, or model capability.
