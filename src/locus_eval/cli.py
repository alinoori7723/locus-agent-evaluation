from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from locus_eval.agents import BUILTIN_AGENTS
from locus_eval.audits import audit_suite
from locus_eval.reporting import replay_report, write_report
from locus_eval.runner import evaluate
from locus_eval.scenarios import build_suite


def load_agent(specification: str):
    if specification in BUILTIN_AGENTS:
        return BUILTIN_AGENTS[specification]
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise ValueError("Agent must be a built-in name or module:factory")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise ValueError("Agent factory must be callable")
    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="locus-eval", description="Reproducible behavioral evaluation for tool-using agents"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "run", "audit"):
        command = commands.add_parser(name)
        command.add_argument("--seed", type=int, default=17)
        if name != "audit":
            command.add_argument("--output", type=Path, default=Path("artifacts") / name)
        if name == "run":
            command.add_argument("--agent", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("directory", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "replay":
            result = replay_report(arguments.directory)
        elif arguments.command == "audit":
            result = audit_suite(build_suite(arguments.seed))
        else:
            agents = (
                BUILTIN_AGENTS
                if arguments.command == "demo"
                else {arguments.agent: load_agent(arguments.agent)}
            )
            scenarios = build_suite(arguments.seed)
            episodes = evaluate(scenarios, agents)
            summary = write_report(arguments.output, episodes, arguments.seed)
            result = {
                "passed": summary["audits"]["passed"]
                and all(metric["agent_errors"] == 0 for metric in summary["metrics"].values()),
                "output": str(arguments.output),
                "metrics": summary["metrics"],
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    except (ValueError, OSError, KeyError, TypeError, AttributeError, ImportError) as exc:
        parser.exit(2, f"locus-eval: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
