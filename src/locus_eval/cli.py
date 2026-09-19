from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from locus_eval.agents import BUILTIN_AGENTS
from locus_eval.audits import audit_suite
from locus_eval.providers.base import ProviderConfig
from locus_eval.providers.openai import OpenAIFactory
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
            command.add_argument("--model")
            command.add_argument("--limit", type=int, default=24)
            command.add_argument("--timeout", type=float, default=30.0)
            command.add_argument("--max-output-tokens", type=int, default=1024)
            command.add_argument("--max-requests", type=int, default=192)
    replay = commands.add_parser("replay")
    replay.add_argument("directory", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "replay":
            result = replay_report(arguments.directory)
        elif arguments.command == "audit":
            result = audit_suite(build_suite(arguments.seed))
        else:
            scenarios = build_suite(arguments.seed)
            provider = None
            if arguments.command == "demo":
                agents = BUILTIN_AGENTS
            else:
                if not 1 <= arguments.limit <= len(scenarios):
                    raise ValueError("--limit must be between 1 and 24")
                scenarios = scenarios[: arguments.limit]
                if arguments.agent == "openai":
                    if arguments.output.exists() and any(arguments.output.iterdir()):
                        raise ValueError("Use a new empty output directory for a provider run")
                    provider = OpenAIFactory(
                        ProviderConfig(
                            model=arguments.model,
                            timeout_seconds=arguments.timeout,
                            max_output_tokens=arguments.max_output_tokens,
                            max_requests=arguments.max_requests,
                        )
                    )
                    agents = {"openai": provider}
                else:
                    if arguments.model is not None:
                        raise ValueError("--model is supported only with --agent openai")
                    agents = {arguments.agent: load_agent(arguments.agent)}
            try:
                episodes = evaluate(scenarios, agents, progress=_progress if provider else None)
            finally:
                if provider is not None:
                    provider.close()
            summary = write_report(
                arguments.output,
                episodes,
                arguments.seed,
                case_ids=[scenario.case_id for scenario in scenarios],
            )
            result = {
                "passed": summary["audits"]["passed"]
                and all(metric["agent_errors"] == 0 for metric in summary["metrics"].values()),
                "output": str(arguments.output),
                "metrics": summary["metrics"],
                "provider_usage": summary["provider_usage"],
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    except (ValueError, OSError, KeyError, TypeError, AttributeError, ImportError) as exc:
        parser.exit(2, f"locus-eval: {exc}\n")


def _progress(completed: int, total: int, episode: dict) -> None:
    score = episode["score"]
    print(
        f"Episode {completed}/{total}: {score['termination']}; "
        f"actions={score['actions']}; verified_success={score['verified_success']}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
