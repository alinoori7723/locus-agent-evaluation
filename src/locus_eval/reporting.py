from __future__ import annotations

import hashlib
import json
from pathlib import Path

from locus_eval.audits import audit_suite
from locus_eval.environment import Environment
from locus_eval.providers.telemetry import summarize_provider_calls, validate_trace
from locus_eval.runner import summarize
from locus_eval.scenarios import build_suite, suite_digest
from locus_eval.serialization import canonical_json, digest

REPORT_FILES = ("summary.json", "episodes.jsonl", "report.md")


def implementation_digest() -> str:
    root = Path(__file__).parent
    paths = sorted([*root.rglob("*.py"), *root.glob("fixtures/*.json")])
    return digest(
        {
            path.relative_to(root).as_posix(): path.read_text("utf-8").replace("\r\n", "\n")
            for path in paths
        }
    )


def make_summary(episodes: list[dict], seed: int, case_ids: list[str] | None = None) -> dict:
    scenarios = build_suite(seed)
    available = {scenario.case_id for scenario in scenarios}
    selected = [scenario.case_id for scenario in scenarios] if case_ids is None else case_ids
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(case, str) or case not in available for case in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("Invalid scenario selection")
    return {
        "schema_version": "2.0",
        "evidence_class": "synthetic-environment",
        "seed": seed,
        "scenario_count": len(scenarios),
        "selected_case_ids": selected,
        "selected_scenario_count": len(selected),
        "episode_count": len(episodes),
        "suite_sha256": suite_digest(scenarios),
        "implementation_sha256": implementation_digest(),
        "metrics": summarize(episodes),
        "provider_usage": summarize_provider_calls(episodes),
        "audits": audit_suite(scenarios),
    }


def markdown_report(summary: dict) -> str:
    lines = [
        "# Agent evaluation report",
        "",
        "Deterministic synthetic fixtures. Trace replay does not attest agent provenance.",
        "",
        f"Seed: `{summary['seed']}` · Selected scenarios: "
        f"**{summary['selected_scenario_count']}/{summary['scenario_count']}** · "
        f"Episodes: **{summary['episode_count']}**",
        "",
        "| Agent | Answer match | Verified success | False credit | Invalid actions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metric in summary["metrics"].items():
        lines.append(
            f"| {name} | {metric['answer_match_rate']:.0%} | "
            f"{metric['verified_success_rate']:.0%} | "
            f"{metric['false_credit_episodes']}/{metric['episodes']} | "
            f"{metric['invalid_actions']} |"
        )
    if summary["provider_usage"]:
        lines.extend(["", "## Provider calls", ""])
        for name, usage in summary["provider_usage"].items():
            lines.extend(
                [
                    f"- Agent: `{name}`; requested model: `{usage['model_requested']}`",
                    f"- Requests attempted: {usage['attempted_requests']}; "
                    f"call errors: {usage['call_errors']}",
                    f"- Known tokens: {usage['known_input_tokens']} input, "
                    f"{usage['known_output_tokens']} output; "
                    f"unknown-usage calls: {usage['unknown_usage_calls']}",
                    f"- Reported elapsed time: {usage['elapsed_ms']:.3f} ms",
                ]
            )
        lines.extend(
            [
                "",
                "Provider metadata is recorded evidence, not authenticated provenance.",
                "A fresh model run can differ. Offline replay verifies captured actions and scores",
                "without contacting a provider. Small public fixtures are integration smoke tests.",
            ]
        )
    lines.extend(
        [
            "",
            "Answer match scores the last attempted diagnosis, even if rejected. Verified success",
            "requires an accepted diagnosis after evidence-backed repair and recovery",
            "verification, with zero rejected actions and no agent error.",
            "False credit is an answer match without verified success.",
            "A recovered episode with earlier rejected actions is reported as",
            "completed but does not pass the strict success metric.",
            "",
            "Rejected actions consume the action budget. Shortcut agents keep submitting until",
            "that budget is exhausted; invalid actions are counts, not independent trials.",
            "",
            "## Regression checks",
            "",
        ]
    )
    for name in ("observations", "error_paths", "shortcuts"):
        result = summary["audits"][name]
        lines.append(f"- {name}: **{'PASS' if result['passed'] else 'FAIL'}**")
    lines.extend(
        [
            "",
            "## Reproduce and inspect",
            "",
            "```bash",
            f"locus-eval demo --seed {summary['seed']} --output artifacts/reproduced",
            "locus-eval replay artifacts/reproduced",
            "```",
            "",
            "The demo command runs the four bundled scripted policies. For a custom-agent report,",
            "rerun `locus-eval run --agent module:factory` with the original adapter and seed.",
            "`episodes.jsonl` records normalized requests, tool responses, rejections, and scores.",
            "`manifest.json` binds the exact report bytes. Replay reruns recorded tool actions,",
            "reconciles scores and aggregates, and checks the suite and implementation digests.",
            "It cannot re-execute or authenticate an external agent's decision process.",
            "",
            f"Suite SHA-256: `{summary['suite_sha256']}`",
            "",
            f"Implementation SHA-256: `{summary['implementation_sha256']}`",
            "",
            "These checks cover declared fixture properties and baselines. They do not establish",
            "general shortcut immunity, production security, reasoning, or model capability.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    directory: Path,
    episodes: list[dict],
    seed: int,
    case_ids: list[str] | None = None,
) -> dict:
    summary = make_summary(episodes, seed, case_ids)
    directory.mkdir(parents=True, exist_ok=True)
    contents = {
        "summary.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "episodes.jsonl": "".join(canonical_json(episode) + "\n" for episode in episodes),
        "report.md": markdown_report(summary),
    }
    for name, content in contents.items():
        (directory / name).write_bytes(content.encode("utf-8"))
    manifest = {
        "schema_version": "1.0",
        "files": {
            name: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for name, content in sorted(contents.items())
        },
    }
    (directory / "manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return summary


def replay_report(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    if manifest.get("schema_version") != "1.0" or set(manifest.get("files", {})) != set(
        REPORT_FILES
    ):
        raise ValueError("Invalid report manifest")
    for name in REPORT_FILES:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Report file digest mismatch: {name}")
    summary = json.loads((directory / "summary.json").read_text("utf-8"))
    scenarios = {scenario.case_id: scenario for scenario in build_suite(summary["seed"])}
    selected = summary["selected_case_ids"]
    make_summary([], summary["seed"], selected)
    episodes = [
        json.loads(line) for line in (directory / "episodes.jsonl").read_text("utf-8").splitlines()
    ]
    if not episodes:
        raise ValueError("A report must contain episodes")
    seen = set()
    for episode in episodes:
        key = episode["agent"], episode["case_id"]
        if key in seen or key[1] not in selected:
            raise ValueError("Duplicate or unknown episode")
        seen.add(key)
        scenario = scenarios[key[1]]
        environment = Environment(scenario)
        observations = [environment.observe()]
        if episode["initial_observation_sha256"] != digest(environment.observe()):
            raise ValueError("Initial observation mismatch")
        if episode["family"] != scenario.family:
            raise ValueError("Scenario family mismatch")
        for event in episode["events"]:
            if environment.done:
                raise ValueError("Events after episode termination")
            environment.step(event["request"])
            observations.append(environment.observe())
            if canonical_json(environment.events[-1]) != canonical_json(event):
                raise ValueError("Tool event mismatch")
        agent_error = episode["score"]["termination"] == "agent_error"
        if agent_error and environment.done:
            raise ValueError("Agent error after episode termination")
        if not agent_error and not environment.done:
            raise ValueError("Unfinished episode without an agent error")
        if canonical_json(environment.score(agent_error=agent_error)) != canonical_json(
            episode["score"]
        ):
            raise ValueError("Episode score mismatch")
        if "provider_trace" in episode:
            validate_trace(episode, observations)
        elif "provider_error" in episode:
            raise ValueError("Provider failure without provider trace")
    for name in {episode["agent"] for episode in episodes}:
        if {case for agent, case in seen if agent == name} != set(selected):
            raise ValueError("Incomplete scenario coverage")
    reconstructed = make_summary(episodes, summary["seed"], selected)
    if canonical_json(summary) != canonical_json(reconstructed):
        raise ValueError("Summary, suite, or implementation mismatch")
    if (directory / "report.md").read_bytes() != markdown_report(reconstructed).encode("utf-8"):
        raise ValueError("Markdown report mismatch")
    return {"passed": True, "episodes_replayed": len(episodes), "seed": summary["seed"]}
