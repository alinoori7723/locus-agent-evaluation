import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from locus_eval.cli import load_agent, main


def test_demo_and_replay_commands(tmp_path, capsys):
    assert main(["demo", "--output", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"]
    assert main(["replay", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["episodes_replayed"] == 96


def test_builtin_single_agent_and_audit_commands(tmp_path, capsys):
    assert main(["run", "--agent", "diagnostic", "--output", str(tmp_path)]) == 0
    assert set(json.loads(capsys.readouterr().out)["metrics"]) == {"diagnostic"}
    assert main(["audit", "--seed", "42"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"]


def test_agent_factory_import_contract():
    assert load_agent("locus_eval.agents:DiagnosticAgent") is load_agent("diagnostic")
    for invalid in ("missing", "module:", ":factory", "module:too:many"):
        with pytest.raises(ValueError):
            load_agent(invalid)
    with pytest.raises(ValueError, match="callable"):
        load_agent("locus_eval.agents:BUILTIN_AGENTS")


def test_invalid_command_input_has_nonzero_exit(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["demo", "--seed", "-1", "--output", str(tmp_path)])
    assert exc.value.code == 2
    assert "Seed must" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["replay", str(tmp_path / "absent")])
    assert exc.value.code == 2


def test_module_entrypoint_works_outside_checkout(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "locus_eval", "audit"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert json.loads(completed.stdout)["passed"]


def test_reports_are_independent_of_python_hash_seed(tmp_path):
    outputs = []
    for hash_seed in ("1", "987654"):
        directory = tmp_path / hash_seed
        subprocess.run(
            [sys.executable, "-m", "locus_eval", "demo", "--output", str(directory)],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        outputs.append({path.name: path.read_bytes() for path in directory.iterdir()})
    assert outputs[0] == outputs[1]


def test_example_adapter_completes_the_suite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "locus_eval",
            "run",
            "--agent",
            "examples.custom_agent:ExampleAgent",
            "--output",
            str(tmp_path),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    metric = json.loads(completed.stdout)["metrics"]["examples.custom_agent:ExampleAgent"]
    assert metric["verified_successes"] == 24


def test_failing_agent_sets_nonzero_command_status(tmp_path, monkeypatch, capsys):
    class BrokenAgent:
        def act(self, observation):
            raise RuntimeError("provider failure")

    monkeypatch.setattr("locus_eval.cli.load_agent", lambda specification: BrokenAgent)
    assert main(["run", "--agent", "broken", "--output", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["metrics"]["broken"]["agent_errors"] == 24
