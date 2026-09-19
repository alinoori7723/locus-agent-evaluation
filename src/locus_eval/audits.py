from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import product

from locus_eval.agents import BUILTIN_AGENTS, action
from locus_eval.environment import Environment
from locus_eval.runner import evaluate, summarize
from locus_eval.scenarios import Scenario
from locus_eval.serialization import canonical_json, digest


def audit_observations(
    scenarios: tuple[Scenario, ...],
    renderer: Callable[[Scenario], dict] = lambda scenario: Environment(scenario).observe(),
) -> dict:
    pairs: dict[str, list[Scenario]] = defaultdict(list)
    failures = []
    public_keys = {
        "protocol_version",
        "task",
        "tools",
        "budget",
        "state_revision",
        "done",
        "history",
    }
    task_keys = {"incident_id", "summary", "components", "acceptance_status", "ticket", "objective"}
    if not scenarios:
        failures.append("empty_suite")
    if len({scenario.case_id for scenario in scenarios}) != len(scenarios):
        failures.append("duplicate_case_id")
    for scenario in scenarios:
        pairs[scenario.pair_id].append(scenario)
        public = renderer(scenario)
        if set(public) != public_keys or set(public.get("task", {})) != task_keys:
            failures.append("public_schema_changed")
    for pair_id, pair in sorted(pairs.items()):
        if len(pair) != 2 or {row.faulty_component for row in pair} != set(pair[0].components):
            failures.append(f"unbalanced_pair:{pair_id}")
            continue
        if len({canonical_json(renderer(row)) for row in pair}) != 1:
            failures.append(f"public_observation_leak:{pair_id}")
    families = {scenario.family for scenario in scenarios}
    for family in sorted(families):
        rows = [scenario for scenario in scenarios if scenario.family == family]
        components = tuple(sorted(rows[0].components))
        actual = Counter(
            (row.components[0], row.suggested_component, row.faulty_component) for row in rows
        )
        expected = Counter(product(components, repeat=3))
        if actual != expected:
            failures.append(f"nuisance_cross_incomplete:{family}")
    return {
        "passed": not failures,
        "scenario_count": len(scenarios),
        "pair_count": len(pairs),
        "family_count": len(families),
        "failures": sorted(set(failures)),
    }


def audit_error_paths(scenarios: tuple[Scenario, ...]) -> dict:
    pairs: dict[str, list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        pairs[scenario.pair_id].append(scenario)
    failures = []
    for pair_id, pair in sorted(pairs.items()):
        views = []
        for scenario in pair:
            environment = Environment(scenario)
            component = scenario.components[0]
            for request in (
                action("read_private"),
                action("probe", component="unknown"),
                action("repair", component=component, evidence_id="unknown"),
                action("verify"),
                action("submit", component=component, verification_id="unknown"),
            ):
                environment.step(request)
            views.append(digest(environment.observe()))
        if len(pair) != 2 or len(set(views)) != 1:
            failures.append(f"error_path_leak:{pair_id}")
    if not scenarios:
        failures.append("empty_suite")
    return {"passed": not failures, "failures": failures}


def audit_shortcuts(scenarios: tuple[Scenario, ...]) -> dict:
    results = summarize(evaluate(scenarios, BUILTIN_AGENTS))
    expected = {
        "diagnostic": (1.0, 1.0),
        "diagnosis-only": (1.0, 0.0),
        "first-candidate": (0.5, 0.0),
        "ticket-follower": (0.5, 0.0),
    }
    failures = [
        f"baseline_changed:{name}"
        for name, (answer_rate, success_rate) in expected.items()
        if results[name]["answer_match_rate"] != answer_rate
        or results[name]["verified_success_rate"] != success_rate
    ]
    return {"passed": not failures, "failures": failures, "baselines": results}


def audit_suite(scenarios: tuple[Scenario, ...]) -> dict:
    observations = audit_observations(scenarios)
    errors = audit_error_paths(scenarios)
    shortcuts = (
        audit_shortcuts(scenarios)
        if observations["passed"]
        else {"passed": False, "failures": ["observation_audit_failed"], "baselines": {}}
    )
    return {
        "passed": all(result["passed"] for result in (observations, errors, shortcuts)),
        "observations": observations,
        "error_paths": errors,
        "shortcuts": shortcuts,
    }
