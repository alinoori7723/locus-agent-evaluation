from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from locus_eval.serialization import json_copy

ERROR_CODES = {
    "AUTHENTICATION_FAILED",
    "RATE_LIMITED",
    "PROVIDER_TIMEOUT",
    "PROVIDER_CONNECTION",
    "PROVIDER_HTTP_ERROR",
    "INCOMPLETE_RESPONSE",
    "MODEL_REFUSAL",
    "INVALID_PROVIDER_RESPONSE",
    "NO_TOOL_CALL",
    "MULTIPLE_TOOL_CALLS",
    "INVALID_TOOL_ARGUMENTS",
    "REQUEST_LIMIT",
    "PROVIDER_CIRCUIT_OPEN",
    "ADAPTER_ERROR",
}


class ProviderFailure(RuntimeError):
    def __init__(self, code: str):
        if code not in ERROR_CODES:
            raise ValueError("Unknown provider error code")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ProviderConfig:
    model: str
    timeout_seconds: float = 30.0
    max_output_tokens: int = 1024
    max_requests: int = 192

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 128:
            raise ValueError("An explicit model name of 1-128 characters is required")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (float, int))
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 120
        ):
            raise ValueError("timeout_seconds must be finite and between 0 and 120")
        if type(self.max_output_tokens) is not int or not 256 <= self.max_output_tokens <= 4096:
            raise ValueError("max_output_tokens must be an integer between 256 and 4096")
        if type(self.max_requests) is not int or not 1 <= self.max_requests <= 192:
            raise ValueError("max_requests must be an integer between 1 and 192")


class RequestBudget:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.used = 0
        self.circuit_open = False

    def reserve(self) -> None:
        if self.circuit_open:
            raise ProviderFailure("PROVIDER_CIRCUIT_OPEN")
        if self.used >= self.maximum:
            raise ProviderFailure("REQUEST_LIMIT")
        self.used += 1


class ProviderAgent:
    def __init__(self, config: ProviderConfig, sdk_version: str):
        self.config = config
        self.sdk_version = sdk_version
        self.calls: list[dict] = []

    def telemetry(self) -> dict:
        return json_copy(
            {
                "provider": "openai",
                "api": "responses",
                "sdk_version": self.sdk_version,
                "config": asdict(self.config),
                "calls": self.calls,
            }
        )
