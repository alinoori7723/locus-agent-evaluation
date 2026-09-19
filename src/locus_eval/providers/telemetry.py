from __future__ import annotations

import math

from locus_eval.providers.base import ERROR_CODES, ProviderConfig
from locus_eval.providers.openai import request_payload
from locus_eval.serialization import canonical_json, digest


def validate_trace(episode: dict, observations: list[dict]) -> None:
    trace = episode["provider_trace"]
    if set(trace) != {"provider", "api", "sdk_version", "config", "calls"}:
        raise ValueError("Invalid provider trace schema")
    if trace["provider"] != "openai" or trace["api"] != "responses":
        raise ValueError("Unknown provider trace")
    if not isinstance(trace["sdk_version"], str) or not trace["sdk_version"]:
        raise ValueError("Missing provider SDK version")
    config = ProviderConfig(**trace["config"])
    error = episode.get("provider_error")
    if error is not None and error not in ERROR_CODES:
        raise ValueError("Unknown provider failure")
    if (error is not None) != (episode["score"]["termination"] == "agent_error"):
        raise ValueError("Provider error and episode termination disagree")
    calls, events = trace["calls"], episode["events"]
    extra_failure = error is not None and error not in {"REQUEST_LIMIT", "PROVIDER_CIRCUIT_OPEN"}
    if len(calls) != len(events) + int(extra_failure):
        raise ValueError("Provider call and action counts disagree")
    for index, call in enumerate(calls):
        if set(call) != {
            "step",
            "request_sha256",
            "response_id",
            "request_id",
            "model_returned",
            "usage",
            "elapsed_ms",
            "action",
            "error",
        }:
            raise ValueError("Invalid provider call schema")
        if type(call["step"]) is not int or call["step"] != index + 1:
            raise ValueError("Provider call step mismatch")
        if call["request_sha256"] != digest(request_payload(observations[index], config)):
            raise ValueError("Provider request is not bound to the public observation")
        if index < len(events):
            if call["error"] is not None or canonical_json(call["action"]) != canonical_json(
                events[index]["request"]
            ):
                raise ValueError("Provider action and environment request disagree")
        elif call["error"] != error or call["action"] is not None:
            raise ValueError("Provider failure receipt mismatch")
        elapsed = call["elapsed_ms"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid provider elapsed time")
        for name in ("response_id", "request_id", "model_returned"):
            value = call[name]
            if value is not None and (not isinstance(value, str) or not 0 < len(value) <= 256):
                raise ValueError("Invalid provider identifier")
        usage = call["usage"]
        if usage is not None and (
            set(usage) != {"input_tokens", "output_tokens", "total_tokens"}
            or any(type(value) is not int or value < 0 for value in usage.values())
            or usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]
        ):
            raise ValueError("Invalid provider usage")


def summarize_provider_calls(episodes: list[dict]) -> dict:
    result = {}
    for name in sorted({row["agent"] for row in episodes if "provider_trace" in row}):
        rows = [row for row in episodes if row["agent"] == name]
        if any("provider_trace" not in row for row in rows):
            raise ValueError("Mixed provider and non-provider episodes")
        configurations = {canonical_json(row["provider_trace"]["config"]) for row in rows}
        if len(configurations) != 1:
            raise ValueError("Provider configuration changed within a run")
        calls = [call for row in rows for call in row["provider_trace"]["calls"]]
        config = rows[0]["provider_trace"]["config"]
        if len(calls) > config["max_requests"]:
            raise ValueError("Provider request budget exceeded")
        known_usage = [call["usage"] for call in calls if call["usage"] is not None]
        result[name] = {
            "provider": "openai",
            "model_requested": config["model"],
            "attempted_requests": len(calls),
            "returned_models": sorted(
                {call["model_returned"] for call in calls if call["model_returned"] is not None}
            ),
            "call_errors": sum(call["error"] is not None for call in calls),
            "known_input_tokens": sum(usage["input_tokens"] for usage in known_usage),
            "known_output_tokens": sum(usage["output_tokens"] for usage in known_usage),
            "known_total_tokens": sum(usage["total_tokens"] for usage in known_usage),
            "unknown_usage_calls": len(calls) - len(known_usage),
            "elapsed_ms": round(sum(call["elapsed_ms"] for call in calls), 3),
        }
    return result
