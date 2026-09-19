from __future__ import annotations

from typing import Any

from locus_eval.protocol import Action, ActionError, normalize_request, tool_schema
from locus_eval.scenarios import Scenario
from locus_eval.serialization import json_copy


class Environment:
    def __init__(self, scenario: Scenario, *, max_actions: int = 8, max_probes: int = 2) -> None:
        if type(max_actions) is not int or not 1 <= max_actions <= 100:
            raise ValueError("max_actions must be an integer between 1 and 100")
        if type(max_probes) is not int or not 1 <= max_probes <= max_actions:
            raise ValueError("max_probes must be an integer between 1 and max_actions")
        self._scenario = scenario
        self.max_actions = max_actions
        self.max_probes = max_probes
        self._events: list[dict] = []
        self._evidence: dict[str, dict] = {}
        self._verification: dict | None = None
        self._repaired: str | None = None
        self._submitted: str | None = None
        self._revision = 0
        self._probes = 0

    @property
    def done(self) -> bool:
        return self._submitted is not None or len(self._events) >= self.max_actions

    @property
    def events(self) -> list[dict]:
        return json_copy(self._events)

    def observe(self) -> dict:
        return {
            "protocol_version": "1.0",
            "task": self._scenario.public_task(),
            "tools": tool_schema(),
            "budget": {
                "actions_remaining": self.max_actions - len(self._events),
                "probes_remaining": self.max_probes - self._probes,
            },
            "state_revision": self._revision,
            "done": self.done,
            "history": self.events,
        }

    def step(self, request: Any) -> dict:
        if self.done:
            return {"ok": False, "error": "EPISODE_CLOSED"}
        normalized = normalize_request(request)
        try:
            action = Action.parse(normalized)
            component = action.arguments.get("component")
            if component is not None and component not in self._scenario.components:
                raise ActionError("UNKNOWN_COMPONENT")
            result = self._execute(action)
            response = {"ok": True, "result": result}
        except ActionError as exc:
            response = {"ok": False, "error": str(exc)}
        self._events.append(
            {"step": len(self._events) + 1, "request": normalized, "response": response}
        )
        return json_copy(response)

    def _execute(self, action: Action) -> dict:
        arguments = action.arguments
        if action.tool == "probe":
            if self._probes >= self.max_probes:
                raise ActionError("PROBE_BUDGET_EXHAUSTED")
            component = arguments["component"]
            check = self._scenario.check(component)
            failed = component == self._scenario.faulty_component and self._repaired != component
            evidence_id = f"probe-{len(self._events) + 1}"
            evidence = {
                "evidence_id": evidence_id,
                "component": component,
                "input": check.input,
                "expected": check.expected,
                "observed": check.faulty if failed else check.expected,
                "passed": not failed,
                "state_revision": self._revision,
            }
            self._evidence[evidence_id] = evidence
            self._probes += 1
            return evidence
        if action.tool == "repair":
            evidence = self._evidence.get(arguments["evidence_id"])
            if evidence is None:
                raise ActionError("UNKNOWN_EVIDENCE")
            if evidence["state_revision"] != self._revision:
                raise ActionError("STALE_EVIDENCE")
            if evidence["component"] != arguments["component"]:
                raise ActionError("EVIDENCE_TARGET_MISMATCH")
            if evidence["passed"]:
                raise ActionError("EVIDENCE_NOT_DIAGNOSTIC")
            self._repaired = arguments["component"]
            self._revision += 1
            self._verification = None
            return {"component": self._repaired, "state_revision": self._revision}
        if action.tool == "verify":
            if self._repaired is None:
                raise ActionError("REPAIR_REQUIRED")
            self._verification = {
                "verification_id": f"verification-{len(self._events) + 1}",
                "passed": self._repaired == self._scenario.faulty_component,
                "state_revision": self._revision,
            }
            return self._verification
        if self._verification is None:
            raise ActionError("VERIFICATION_REQUIRED")
        if (
            arguments["verification_id"] != self._verification["verification_id"]
            or self._verification["state_revision"] != self._revision
        ):
            raise ActionError("INVALID_VERIFICATION")
        if not self._verification["passed"]:
            raise ActionError("RECOVERY_REQUIRED")
        if arguments["component"] != self._repaired:
            raise ActionError("DIAGNOSIS_MISMATCH")
        self._submitted = arguments["component"]
        return {"accepted": True}

    def score(self, *, agent_error: bool = False) -> dict:
        submissions = [
            event["request"]["arguments"].get("component")
            for event in self._events
            if isinstance(event["request"], dict)
            and event["request"].get("tool") == "submit"
            and isinstance(event["request"].get("arguments"), dict)
        ]
        answer_match = bool(submissions and submissions[-1] == self._scenario.faulty_component)
        invalid_actions = sum(not event["response"]["ok"] for event in self._events)
        completed = self._submitted == self._scenario.faulty_component
        verified_success = completed and invalid_actions == 0 and not agent_error
        if agent_error:
            termination = "agent_error"
        elif completed:
            termination = "submitted"
        elif self.done:
            termination = "action_limit"
        else:
            termination = "incomplete"
        return {
            "answer_match": answer_match,
            "completed": completed,
            "verified_success": verified_success,
            "false_credit": answer_match and not verified_success,
            "invalid_actions": invalid_actions,
            "actions": len(self._events),
            "probes": self._probes,
            "termination": termination,
        }
