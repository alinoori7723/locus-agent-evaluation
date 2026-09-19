from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from locus_eval.serialization import canonical_json, json_copy

TOOL_PARAMETERS = {
    "probe": ("component",),
    "repair": ("component", "evidence_id"),
    "verify": (),
    "submit": ("component", "verification_id"),
}
TOOL_DESCRIPTIONS = {
    "probe": "Run one isolated component check and return measured evidence.",
    "repair": "Repair a component using a current failed probe for that component.",
    "verify": "Run the acceptance check after a repair and issue a verification receipt.",
    "submit": "Submit the repaired component with its current successful verification receipt.",
}
MAX_REQUEST_BYTES = 4096


class Agent(Protocol):
    def act(self, observation: dict) -> Any: ...


class ActionError(ValueError):
    pass


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: dict[str, str]

    @classmethod
    def parse(cls, request: Any) -> Action:
        if not isinstance(request, dict) or set(request) != {"tool", "arguments"}:
            raise ActionError("INVALID_ACTION")
        tool, arguments = request["tool"], request["arguments"]
        if not isinstance(tool, str) or not isinstance(arguments, dict):
            raise ActionError("INVALID_ACTION")
        if tool not in TOOL_PARAMETERS:
            raise ActionError("UNKNOWN_TOOL")
        if set(arguments) != set(TOOL_PARAMETERS[tool]):
            raise ActionError("INVALID_ARGUMENTS")
        if any(
            type(value) is not str or not 1 <= len(value) <= 128 for value in arguments.values()
        ):
            raise ActionError("INVALID_ARGUMENTS")
        return cls(tool=tool, arguments=dict(arguments))


def normalize_request(value: Any) -> Any:
    try:
        encoded = canonical_json(value)
        if len(encoded.encode("utf-8")) > MAX_REQUEST_BYTES:
            return None
        return json.loads(encoded)
    except (TypeError, ValueError, RecursionError, OverflowError):
        return None


def tool_schema() -> list[dict]:
    return json_copy(
        [
            {
                "name": tool,
                "description": TOOL_DESCRIPTIONS[tool],
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: {"type": "string", "minLength": 1, "maxLength": 128}
                        for name in parameters
                    },
                    "required": list(parameters),
                    "additionalProperties": False,
                },
            }
            for tool, parameters in TOOL_PARAMETERS.items()
        ]
    )
