from __future__ import annotations

from collections.abc import Callable, Mapping

from locus_eval.environment import Environment
from locus_eval.protocol import Agent
from locus_eval.providers.base import ProviderAgent, ProviderFailure
from locus_eval.scenarios import Scenario
from locus_eval.serialization import digest


def run_episode(scenario: Scenario, factory: Callable[[], Agent], agent_name: str) -> dict:
    environment = Environment(scenario)
    agent_error = False
    provider_error = None
    agent = None
    try:
        agent = factory()
    except Exception:
        agent_error = True
    while not environment.done and not agent_error:
        try:
            request = agent.act(environment.observe())
        except ProviderFailure as exc:
            agent_error = True
            provider_error = exc.code
            break
        except Exception:
            agent_error = True
            break
        environment.step(request)
    episode = {
        "agent": agent_name,
        "case_id": scenario.case_id,
        "family": scenario.family,
        "initial_observation_sha256": digest(Environment(scenario).observe()),
        "events": environment.events,
        "score": environment.score(agent_error=agent_error),
    }
    if isinstance(agent, ProviderAgent):
        episode["provider_trace"] = agent.telemetry()
        episode["provider_error"] = provider_error
    return episode


def evaluate(
    scenarios: tuple[Scenario, ...],
    agents: Mapping[str, Callable[[], Agent]],
    progress: Callable[[int, int, dict], None] | None = None,
) -> list[dict]:
    if not scenarios or not agents:
        raise ValueError("Evaluation requires at least one scenario and one agent")
    episodes = []
    for name, factory in sorted(agents.items()):
        for scenario in scenarios:
            episode = run_episode(scenario, factory, name)
            episodes.append(episode)
            if progress is not None:
                progress(len(episodes), len(scenarios) * len(agents), episode)
    return episodes


def summarize(episodes: list[dict]) -> dict[str, dict]:
    summary = {}
    for name in sorted({episode["agent"] for episode in episodes}):
        scores = [episode["score"] for episode in episodes if episode["agent"] == name]
        count = len(scores)
        summary[name] = {
            "episodes": count,
            "answer_matches": sum(score["answer_match"] for score in scores),
            "verified_successes": sum(score["verified_success"] for score in scores),
            "completed_episodes": sum(score["completed"] for score in scores),
            "false_credit_episodes": sum(score["false_credit"] for score in scores),
            "invalid_actions": sum(score["invalid_actions"] for score in scores),
            "agent_errors": sum(score["termination"] == "agent_error" for score in scores),
            "answer_match_rate": sum(score["answer_match"] for score in scores) / count,
            "verified_success_rate": sum(score["verified_success"] for score in scores) / count,
            "mean_actions": sum(score["actions"] for score in scores) / count,
        }
    return summary
