import hashlib
import json
from pathlib import Path

import pytest

from locus_eval.agents import BUILTIN_AGENTS
from locus_eval.reporting import replay_report, write_report
from locus_eval.runner import evaluate
from locus_eval.serialization import canonical_json


@pytest.fixture
def report(tmp_path, suite):
    write_report(tmp_path, evaluate(suite, BUILTIN_AGENTS), 17)
    return tmp_path


def resign(directory):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    for name in manifest["files"]:
        manifest["files"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_generated_reports_replay_and_regenerate_byte_for_byte(report, suite):
    before = {path.name: path.read_bytes() for path in report.iterdir()}
    assert replay_report(report) == {"passed": True, "episodes_replayed": 96, "seed": 17}
    write_report(report, evaluate(suite, BUILTIN_AGENTS), 17)
    assert before == {path.name: path.read_bytes() for path in report.iterdir()}


def test_committed_sample_matches_current_implementation(report):
    committed = Path(__file__).resolve().parents[1] / "reports" / "sample"
    for name in ("summary.json", "episodes.jsonl", "report.md", "manifest.json"):
        assert (committed / name).read_bytes() == (report / name).read_bytes(), name


@pytest.mark.parametrize("name", ["summary.json", "episodes.jsonl", "report.md"])
def test_file_tampering_is_detected_before_replay(report, name):
    path = report / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        replay_report(report)


def test_manifest_cannot_select_files_outside_report_directory(report):
    manifest = {"schema_version": "1.0", "files": {"../../private.json": "0" * 64}}
    (report / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid report manifest"):
        replay_report(report)


@pytest.mark.parametrize(
    "mutation",
    ["response", "score", "family", "observation", "duplicate", "missing", "unknown", "extra-step"],
)
def test_replay_detects_semantic_tampering_even_after_manifest_is_recomputed(report, mutation):
    path = report / "episodes.jsonl"
    episodes = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    episode = next(item for item in episodes if item["agent"] == "diagnostic")
    if mutation == "response":
        episode["events"][0]["response"]["result"]["observed"] = "invented"
    elif mutation == "score":
        episode["score"]["actions"] += 1
    elif mutation == "family":
        episode["family"] = "unknown"
    elif mutation == "observation":
        episode["initial_observation_sha256"] = "0" * 64
    elif mutation == "duplicate":
        episodes.append(episodes[0])
    elif mutation == "missing":
        episodes.pop()
    elif mutation == "unknown":
        episode["case_id"] = "unknown"
    else:
        episode["events"].append(episode["events"][-1])
    path.write_bytes("".join(canonical_json(row) + "\n" for row in episodes).encode())
    resign(report)
    with pytest.raises(ValueError):
        replay_report(report)


@pytest.mark.parametrize("field", ["implementation_sha256", "suite_sha256", "episode_count"])
def test_summary_drift_cannot_be_hidden_by_resigning_manifest(report, field):
    path = report / "summary.json"
    summary = json.loads(path.read_text("utf-8"))
    summary[field] = "incorrect"
    path.write_text(json.dumps(summary), encoding="utf-8")
    resign(report)
    with pytest.raises(ValueError, match="Summary, suite, or implementation mismatch"):
        replay_report(report)


def test_readable_report_is_reconciled_too(report):
    path = report / "report.md"
    path.write_text("All agents succeeded", encoding="utf-8")
    resign(report)
    with pytest.raises(ValueError, match="Markdown report mismatch"):
        replay_report(report)


def test_empty_and_unfinished_traces_are_rejected(report):
    path = report / "episodes.jsonl"
    episodes = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    episodes[0]["events"] = []
    path.write_bytes("".join(canonical_json(row) + "\n" for row in episodes).encode())
    resign(report)
    with pytest.raises(ValueError, match="Unfinished episode"):
        replay_report(report)
    path.write_bytes(b"")
    resign(report)
    with pytest.raises(ValueError, match="must contain episodes"):
        replay_report(report)


def test_agent_error_reports_are_replayable(tmp_path, suite):
    class BrokenAgent:
        def act(self, observation):
            raise RuntimeError("failed provider")

    write_report(tmp_path, evaluate(suite, {"broken": BrokenAgent}), 17)
    assert replay_report(tmp_path)["passed"]


def test_agent_error_cannot_be_inserted_after_completion(report):
    path = report / "episodes.jsonl"
    episodes = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    episode = next(item for item in episodes if item["agent"] == "diagnostic")
    episode["score"]["termination"] = "agent_error"
    path.write_bytes("".join(canonical_json(row) + "\n" for row in episodes).encode())
    resign(report)
    with pytest.raises(ValueError, match="Agent error after episode termination"):
        replay_report(report)
