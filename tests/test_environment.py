import math

import pytest

from locus_eval.agents import DiagnosticAgent, action
from locus_eval.environment import Environment
from locus_eval.protocol import TOOL_PARAMETERS
from locus_eval.runner import run_episode
from locus_eval.serialization import canonical_json


def repair_fault(environment, component="router"):
    probe = environment.step(action("probe", component=component))["result"]
    environment.step(action("repair", component=component, evidence_id=probe["evidence_id"]))
    return probe


@pytest.mark.parametrize("index", range(24))
def test_reference_policy_completes_every_world(suite, index):
    episode = run_episode(suite[index], DiagnosticAgent, "diagnostic")
    assert episode["score"]["verified_success"]
    assert episode["score"]["termination"] == "submitted"
    assert episode["score"]["actions"] in (4, 5)
    assert episode["score"]["invalid_actions"] == 0
    requests = [event["request"]["tool"] for event in episode["events"]]
    assert requests[-3:] == ["repair", "verify", "submit"]


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (None, "INVALID_ACTION"),
        ([], "INVALID_ACTION"),
        ("probe", "INVALID_ACTION"),
        (True, "INVALID_ACTION"),
        (math.nan, "INVALID_ACTION"),
        ({}, "INVALID_ACTION"),
        ({"tool": [], "arguments": {}}, "INVALID_ACTION"),
        ({"tool": "probe", "arguments": []}, "INVALID_ACTION"),
        ({"tool": "probe", "arguments": {}, "override": True}, "INVALID_ACTION"),
        (action("shell", command="cat private.json"), "UNKNOWN_TOOL"),
        (action("probe"), "INVALID_ARGUMENTS"),
        (action("probe", component=""), "INVALID_ARGUMENTS"),
        (action("probe", component="x" * 129), "INVALID_ARGUMENTS"),
        (action("probe", component="router", answer="router"), "INVALID_ARGUMENTS"),
        ({"tool": "probe", "arguments": {"component": 1}}, "INVALID_ARGUMENTS"),
        ({"tool": "probe", "arguments": {"component": False}}, "INVALID_ARGUMENTS"),
        ({"tool": "probe", "arguments": {"component": ["router"]}}, "INVALID_ARGUMENTS"),
        (action("probe", component="../../private.json"), "UNKNOWN_COMPONENT"),
        (action("probe", component="x" * 5000), "INVALID_ACTION"),
        (action("verify", unnecessary="value"), "INVALID_ARGUMENTS"),
    ],
)
def test_invalid_requests_are_rejected_without_system_mutation(scenario, payload, code):
    environment = Environment(scenario)
    assert environment.step(payload) == {"ok": False, "error": code}
    public = environment.observe()
    assert public["state_revision"] == 0
    assert public["budget"] == {"actions_remaining": 7, "probes_remaining": 2}
    assert not environment.score()["completed"]


def test_unserializable_and_cyclic_requests_are_auditable(scenario):
    cyclic = {}
    cyclic["self"] = cyclic
    for request in (object(), {"value": object()}, cyclic):
        environment = Environment(scenario)
        assert environment.step(request)["error"] == "INVALID_ACTION"
        assert environment.events[0]["request"] is None
        canonical_json(environment.events)


def test_repair_requires_existing_evidence(scenario):
    environment = Environment(scenario)
    response = environment.step(action("repair", component="router", evidence_id="probe-1"))
    assert response["error"] == "UNKNOWN_EVIDENCE"


def test_evidence_cannot_be_rebound_to_another_component(scenario):
    environment = Environment(scenario)
    environment.step(action("probe", component="router"))
    response = environment.step(action("repair", component="queue", evidence_id="probe-1"))
    assert response["error"] == "EVIDENCE_TARGET_MISMATCH"
    assert environment.observe()["state_revision"] == 0


def test_healthy_probe_does_not_authorize_repair(scenario):
    environment = Environment(scenario)
    environment.step(action("probe", component="queue"))
    response = environment.step(action("repair", component="queue", evidence_id="probe-1"))
    assert response["error"] == "EVIDENCE_NOT_DIAGNOSTIC"


def test_repair_invalidates_prior_evidence(scenario):
    environment = Environment(scenario)
    probe = repair_fault(environment)
    response = environment.step(
        action("repair", component="router", evidence_id=probe["evidence_id"])
    )
    assert response["error"] == "STALE_EVIDENCE"
    assert environment.observe()["state_revision"] == 1


def test_probe_after_repair_observes_recovery(scenario):
    environment = Environment(scenario)
    repair_fault(environment)
    response = environment.step(action("probe", component="router"))["result"]
    assert response["passed"]
    assert response["observed"] == response["expected"]
    assert response["state_revision"] == 1


def test_verify_requires_a_repair(scenario):
    assert Environment(scenario).step(action("verify"))["error"] == "REPAIR_REQUIRED"


def test_correct_answer_before_verification_has_no_success_credit(scenario):
    environment = Environment(scenario)
    response = environment.step(action("submit", component="router", verification_id="guessed"))
    assert response["error"] == "VERIFICATION_REQUIRED"
    score = environment.score()
    assert score["answer_match"] and score["false_credit"]
    assert not score["completed"] and not score["verified_success"]


def test_repair_without_recovery_check_is_insufficient(scenario):
    environment = Environment(scenario)
    repair_fault(environment)
    response = environment.step(action("submit", component="router", verification_id="guessed"))
    assert response["error"] == "VERIFICATION_REQUIRED"


@pytest.mark.parametrize("token", ["verification-1", "unknown", "verification-999"])
def test_forged_verification_is_rejected(scenario, token):
    environment = Environment(scenario)
    repair_fault(environment)
    environment.step(action("verify"))
    response = environment.step(action("submit", component="router", verification_id=token))
    assert response["error"] == "INVALID_VERIFICATION"


def test_new_verification_supersedes_old_receipt(scenario):
    environment = Environment(scenario)
    repair_fault(environment)
    old = environment.step(action("verify"))["result"]["verification_id"]
    environment.step(action("verify"))
    response = environment.step(action("submit", component="router", verification_id=old))
    assert response["error"] == "INVALID_VERIFICATION"


def test_submission_must_match_repaired_component(scenario):
    environment = Environment(scenario)
    repair_fault(environment)
    token = environment.step(action("verify"))["result"]["verification_id"]
    response = environment.step(action("submit", component="queue", verification_id=token))
    assert response["error"] == "DIAGNOSIS_MISMATCH"


def test_recovery_after_rejection_is_distinct_from_strict_success(scenario):
    environment = Environment(scenario)
    environment.step(action("verify"))
    agent = DiagnosticAgent()
    while not environment.done:
        environment.step(agent.act(environment.observe()))
    score = environment.score()
    assert score["completed"] and score["answer_match"]
    assert not score["verified_success"]
    assert score["invalid_actions"] == 1


def test_exhausted_probe_budget_rejects_without_consuming_another_probe(scenario):
    environment = Environment(scenario)
    environment.step(action("probe", component="router"))
    environment.step(action("probe", component="queue"))
    assert (
        environment.step(action("probe", component="router"))["error"] == "PROBE_BUDGET_EXHAUSTED"
    )
    assert environment.score()["probes"] == 2


def test_rejections_exhaust_action_budget_and_closed_episodes_are_immutable(scenario):
    environment = Environment(scenario, max_actions=2)
    for _ in range(2):
        environment.step(action("verify"))
    before = environment.observe()
    assert environment.done
    assert environment.step(action("probe", component="router"))["error"] == "EPISODE_CLOSED"
    assert environment.observe() == before
    assert environment.score()["termination"] == "action_limit"


def test_agent_cannot_mutate_environment_through_observation_or_response(scenario):
    environment = Environment(scenario)
    public = environment.observe()
    public["task"]["components"].clear()
    public["tools"].clear()
    response = environment.step(action("probe", component="router"))
    response["result"]["passed"] = True
    public = environment.observe()
    public["history"][0]["response"]["result"]["component"] = "queue"
    assert len(environment.observe()["tools"]) == len(TOOL_PARAMETERS)
    assert len(environment.observe()["task"]["components"]) == 2
    assert environment.events[0]["response"]["result"]["passed"] is False
    assert environment.step(action("repair", component="router", evidence_id="probe-1"))["ok"]


@pytest.mark.parametrize("maximum", [0, -1, 101, True, 1.5, "8"])
def test_invalid_action_limits_fail_fast(scenario, maximum):
    with pytest.raises(ValueError):
        Environment(scenario, max_actions=maximum)


@pytest.mark.parametrize("maximum", [0, -1, 9, True, 1.5, "2"])
def test_invalid_probe_limits_fail_fast(scenario, maximum):
    with pytest.raises(ValueError):
        Environment(scenario, max_probes=maximum)
