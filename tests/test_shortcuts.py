from collections import defaultdict
from dataclasses import replace

import pytest

from locus_eval.agents import BUILTIN_AGENTS, DiagnosticAgent, action
from locus_eval.audits import audit_shortcuts
from locus_eval.environment import Environment
from locus_eval.runner import evaluate, run_episode, summarize


def test_observation_only_baselines_cannot_exceed_half_the_paired_worlds(suite):
    metrics = summarize(evaluate(suite, BUILTIN_AGENTS))
    for name in ("first-candidate", "ticket-follower"):
        assert metrics[name]["answer_matches"] == 12
        assert metrics[name]["verified_successes"] == 0
        assert metrics[name]["false_credit_episodes"] == 12
    assert metrics["diagnosis-only"]["answer_matches"] == 24
    assert metrics["diagnosis-only"]["verified_successes"] == 0
    assert metrics["diagnostic"]["verified_successes"] == 24


def test_reference_policy_is_invariant_to_candidate_order_and_hint_text(suite):
    for scenario in suite:
        changed = replace(
            scenario,
            components=scenario.components[::-1],
            suggested_component=next(
                name for name in scenario.components if name != scenario.suggested_component
            ),
        )
        assert run_episode(changed, DiagnosticAgent, "diagnostic")["score"]["verified_success"]


def test_metadata_hash_rule_is_at_chance_within_each_pair(suite):
    class MetadataAgent:
        def act(self, observation):
            task = observation["task"]
            choice = int(task["incident_id"][-1], 16) % 2
            return action("submit", component=task["components"][choice], verification_id="none")

    pairs = defaultdict(list)
    for case in suite:
        pairs[case.pair_id].append(run_episode(case, MetadataAgent, "metadata")["score"])
    assert all(sum(score["answer_match"] for score in pair) == 1 for pair in pairs.values())


def test_shortcut_guard_fails_if_first_position_becomes_the_answer(suite):
    biased = tuple(
        replace(
            case,
            components=(
                case.faulty_component,
                next(name for name in case.components if name != case.faulty_component),
            ),
        )
        for case in suite
    )
    result = audit_shortcuts(biased)
    assert not result["passed"]
    assert "baseline_changed:first-candidate" in result["failures"]


def test_agent_instances_are_reset_per_episode(suite):
    creations = []

    class FreshAgent(DiagnosticAgent):
        def __init__(self):
            creations.append(self)

    evaluate(suite, {"fresh": FreshAgent})
    assert len(creations) == len(suite)


@pytest.mark.parametrize("during_factory", [False, True])
def test_agent_exceptions_are_terminal_and_do_not_expose_exception_text(scenario, during_factory):
    class BrokenAgent:
        def __init__(self):
            if during_factory:
                raise RuntimeError("private-answer-router")

        def act(self, observation):
            raise RuntimeError("private-answer-router")

    episode = run_episode(scenario, BrokenAgent, "broken")
    assert episode["score"]["termination"] == "agent_error"
    assert episode["score"]["actions"] == 0
    assert not episode["score"]["verified_success"]
    assert "private-answer-router" not in str(episode)


def test_evaluator_bugs_are_not_mislabeled_as_agent_failures(scenario, monkeypatch):
    def broken_step(self, request):
        raise RuntimeError("evaluator bug")

    monkeypatch.setattr(Environment, "step", broken_step)
    with pytest.raises(RuntimeError, match="evaluator bug"):
        run_episode(scenario, DiagnosticAgent, "diagnostic")


def test_empty_evaluations_are_rejected(suite):
    with pytest.raises(ValueError):
        evaluate((), BUILTIN_AGENTS)
    with pytest.raises(ValueError):
        evaluate(suite, {})
