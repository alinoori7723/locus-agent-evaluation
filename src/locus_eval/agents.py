from __future__ import annotations

from collections.abc import Callable

from locus_eval.protocol import Agent


def action(tool: str, **arguments: str) -> dict:
    return {"tool": tool, "arguments": arguments}


class DiagnosticAgent:
    def act(self, observation: dict) -> dict:
        successful = [event for event in observation["history"] if event["response"]["ok"]]
        probes = {
            event["response"]["result"]["component"]: event["response"]["result"]
            for event in successful
            if event["request"]["tool"] == "probe"
        }
        repairs = [event for event in successful if event["request"]["tool"] == "repair"]
        verifications = [event for event in successful if event["request"]["tool"] == "verify"]
        if verifications:
            return action(
                "submit",
                component=repairs[-1]["request"]["arguments"]["component"],
                verification_id=verifications[-1]["response"]["result"]["verification_id"],
            )
        if repairs:
            return action("verify")
        failed = next((probe for probe in probes.values() if not probe["passed"]), None)
        if failed:
            return action(
                "repair", component=failed["component"], evidence_id=failed["evidence_id"]
            )
        component = next(name for name in observation["task"]["components"] if name not in probes)
        return action("probe", component=component)


class FirstCandidateAgent:
    def act(self, observation: dict) -> dict:
        return action(
            "submit", component=observation["task"]["components"][0], verification_id="unverified"
        )


class TicketFollowerAgent:
    def act(self, observation: dict) -> dict:
        return action(
            "submit",
            component=observation["task"]["ticket"]["suggested_component"],
            verification_id="unverified",
        )


class DiagnosisOnlyAgent(DiagnosticAgent):
    def act(self, observation: dict) -> dict:
        planned = super().act(observation)
        if planned["tool"] == "repair":
            return action(
                "submit", component=planned["arguments"]["component"], verification_id="unverified"
            )
        return planned


BUILTIN_AGENTS: dict[str, Callable[[], Agent]] = {
    "diagnostic": DiagnosticAgent,
    "diagnosis-only": DiagnosisOnlyAgent,
    "first-candidate": FirstCandidateAgent,
    "ticket-follower": TicketFollowerAgent,
}
