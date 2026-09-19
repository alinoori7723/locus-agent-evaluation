from collections import defaultdict
from dataclasses import replace

import pytest

from locus_eval.agents import action
from locus_eval.audits import audit_error_paths, audit_observations, audit_suite
from locus_eval.environment import Environment
from locus_eval.scenarios import build_suite, suite_digest
from locus_eval.serialization import canonical_json


@pytest.mark.parametrize("seed", [0, 17, 982451653, 2**32 - 1])
def test_all_public_surfaces_match_between_opposite_faults(seed):
    suite = build_suite(seed)
    result = audit_suite(suite)
    assert result["passed"], result
    assert result["observations"]["pair_count"] == 12
    assert result["observations"]["scenario_count"] == 24
    assert result["observations"]["family_count"] == 3


@pytest.mark.parametrize("channel", ["metadata", "identifier", "ticket", "order", "budget"])
def test_leakage_audit_detects_deliberately_contaminated_views(suite, channel):
    def leaking_renderer(scenario):
        public = Environment(scenario).observe()
        fault = scenario.faulty_component
        if channel == "metadata":
            public["task"]["faulty_component"] = fault
        elif channel == "identifier":
            public["task"]["incident_id"] += fault
        elif channel == "ticket":
            public["task"]["ticket"]["text"] += fault
        elif channel == "order":
            public["task"]["components"].sort(key=lambda name: name != fault)
        else:
            public["budget"]["actions_remaining"] += scenario.components.index(fault)
        return public

    result = audit_observations(suite, leaking_renderer)
    assert not result["passed"]
    assert any("public_observation_leak" in failure for failure in result["failures"])


def test_error_paths_reveal_no_fault_before_an_informative_probe(suite):
    assert audit_error_paths(suite)["passed"]


def test_informative_probes_distinguish_every_paired_world(suite):
    pairs = defaultdict(list)
    for scenario in suite:
        pairs[scenario.pair_id].append(scenario)
    for left, right in pairs.values():
        first, second = Environment(left), Environment(right)
        assert canonical_json(first.observe()) == canonical_json(second.observe())
        probe = action("probe", component=left.components[0])
        observed = [first.step(probe), second.step(probe)]
        assert {response["result"]["passed"] for response in observed} == {True, False}
        assert observed[0]["result"]["evidence_id"] == observed[1]["result"]["evidence_id"]


def test_missing_pair_duplicate_case_and_empty_suite_fail_closed(suite):
    for broken in (suite[:-1], (*suite, suite[0]), ()):
        assert not audit_observations(broken)["passed"]
        assert not audit_suite(broken)["passed"]


def test_biased_hint_distribution_is_detected(suite):
    biased = tuple(replace(case, suggested_component=case.faulty_component) for case in suite)
    result = audit_observations(biased)
    assert not result["passed"]
    assert any("nuisance_cross_incomplete" in failure for failure in result["failures"])


def test_seed_changes_identifiers_and_order_but_not_label_counts():
    first, second = build_suite(17), build_suite(18)
    assert first == build_suite(17)
    assert suite_digest(first) != suite_digest(second)
    assert {case.pair_id for case in first}.isdisjoint({case.pair_id for case in second})
    assert sorted(case.faulty_component for case in first) == sorted(
        case.faulty_component for case in second
    )


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True, "17"])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError):
        build_suite(seed)


@pytest.mark.parametrize(
    "change",
    [
        {"faulty_component": "absent"},
        {"components": ("router", "router")},
        {"suggested_component": "absent"},
        {"checks": ()},
    ],
)
def test_invalid_scenario_contract_is_rejected(scenario, change):
    with pytest.raises(ValueError):
        replace(scenario, **change)


def test_invisible_fault_is_rejected(scenario):
    check = scenario.checks[0]
    with pytest.raises(ValueError):
        replace(scenario, checks=(replace(check, faulty=check.expected), scenario.checks[1]))
