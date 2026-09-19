# Research motivation

LOCUS asks questions about causal abstraction and failure localization: which distinctions remain
available in an agent's observations, when a proposed explanation is justified, and when an
apparently successful answer can be produced by a simpler shortcut.

This repository carries those questions into a small software evaluation harness. Its public
story starts with an engineering failure: an answer-only evaluator can reward the correct component
even when the agent has not performed the required diagnostic and recovery procedure.

## Ideas translated into tests

| Research question | Engineering translation |
| --- | --- |
| Can the observation identify the failure? | Opposite-fault worlds share the same initial public payload; a probe supplies distinguishing evidence |
| Could an incidental feature predict the target? | Cross labels with candidate order and ticket hints; test exact public-view equality |
| Does a correct action justify a stronger success claim? | Report answer match separately from evidence-backed completion and strict path validity |
| Can a validator itself miss the failure? | Introduce known leaks, biased orderings, and corrupted traces and require rejection |
| Is a reported result reproducible? | Record actions and responses, fingerprint source and fixtures, then replay and reconcile scores |

These are design connections, not claims that the incident simulator implements the full LOCUS
research program. The simulator has a fixed two-component ontology and explicit observable checks.
It does not test causal abstraction theorems, ambiguity under ontology changes, or general causal
reasoning. The diagnostic policy follows a known procedure.

## Evidence boundary

All included outcomes are generated from the new synthetic fixtures and bundled scripted policies.
No model-study scores, human responses, research answer banks, private credentials, or publication
artifacts are part of this repository. No historical experiment is represented as a result of this
harness. The research workspace remains separate.

A pass supports the stated software properties. A study of real model behavior would need its own
model bindings, sampling design, held-out tasks, statistical analysis, and explicit treatment of
contamination and uncertainty. Publicly inspectable demo fixtures cannot serve as a hidden benchmark.
