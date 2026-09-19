import hashlib
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import httpx2
import openai
import pytest

from locus_eval.agents import DiagnosticAgent, action
from locus_eval.cli import main
from locus_eval.environment import Environment
from locus_eval.providers.base import ProviderConfig, ProviderFailure, RequestBudget
from locus_eval.providers.openai import OpenAIAgent, OpenAIFactory, request_payload, response_action
from locus_eval.providers.telemetry import summarize_provider_calls, validate_trace
from locus_eval.reporting import replay_report, write_report
from locus_eval.runner import evaluate, run_episode
from locus_eval.serialization import canonical_json, digest


def response_body(request_action, **overrides):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.6-test-snapshot",
        "parallel_tool_calls": False,
        "output": [
            {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": request_action["tool"],
                "arguments": json.dumps(request_action["arguments"]),
                "status": "completed",
            }
        ],
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        **overrides,
    }


@pytest.fixture
def sdk_factory():
    clients = []

    def create(handler, config=None):
        config = config or ProviderConfig("gpt-5.6")
        client = openai.OpenAI(
            api_key="unit-test-placeholder",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
            max_retries=0,
            timeout=config.timeout_seconds,
        )
        clients.append(client)
        budget = RequestBudget(config.max_requests)
        return lambda: OpenAIAgent(client, config, budget, openai.__version__, openai.APIError)

    yield create
    for client in clients:
        client.close()


def diagnostic_transport(captured):
    def handle(request):
        payload = json.loads(request.content)
        captured.append(payload)
        observation = json.loads(payload["input"][0]["content"])
        selected = DiagnosticAgent().act(observation)
        return httpx2.Response(
            200, json=response_body(selected), headers={"x-request-id": "request_test"}
        )

    return handle


def test_real_sdk_serialization_executes_full_workflow_and_replays_offline(
    sdk_factory, suite, tmp_path
):
    captured = []
    factory = sdk_factory(diagnostic_transport(captured))
    episodes = evaluate(suite, {"openai": factory})
    summary = write_report(tmp_path, episodes, 17)
    assert summary["metrics"]["openai"]["verified_successes"] == 24
    assert summary["provider_usage"]["openai"]["attempted_requests"] == 108
    assert summary["provider_usage"]["openai"]["known_total_tokens"] == 108 * 120
    assert summary["provider_usage"]["openai"]["returned_models"] == ["gpt-5.6-test-snapshot"]
    assert replay_report(tmp_path)["episodes_replayed"] == 24
    assert len(captured) == 108
    for payload in captured:
        assert payload["model"] == "gpt-5.6"
        assert payload["store"] is False and payload["parallel_tool_calls"] is False
        assert payload["tool_choice"] == "required"
        assert payload["max_output_tokens"] == 1024
        assert len(payload["tools"]) == 4
        assert all(tool["strict"] is True for tool in payload["tools"])
        text = canonical_json(payload)
        for forbidden in ("faulty_component", "case_id", "unit-test-placeholder"):
            assert forbidden not in text
    receipt = episodes[0]["provider_trace"]["calls"][0]
    assert receipt["request_id"] == "request_test"
    assert receipt["response_id"] == "resp_test"
    assert "unit-test-placeholder" not in canonical_json(episodes)


def test_provider_cannot_bypass_deterministic_validation(sdk_factory, scenario):
    factory = sdk_factory(
        lambda request: httpx2.Response(
            200,
            json=response_body(action("submit", component="router", verification_id="invented")),
        )
    )
    episode = run_episode(scenario, factory, "openai")
    assert episode["score"]["answer_match"]
    assert not episode["score"]["verified_success"]
    assert episode["score"]["invalid_actions"] == 8


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "AUTHENTICATION_FAILED"),
        (403, "AUTHENTICATION_FAILED"),
        (429, "RATE_LIMITED"),
        (500, "PROVIDER_HTTP_ERROR"),
        (400, "PROVIDER_HTTP_ERROR"),
    ],
)
def test_http_errors_are_redacted_not_retried_and_open_the_circuit(
    sdk_factory,
    suite,
    tmp_path,
    status,
    code,
):
    requests = []

    def fail(request):
        requests.append(request)
        return httpx2.Response(status, json={"error": {"message": "secret-sentinel"}})

    episodes = evaluate(suite, {"openai": sdk_factory(fail)})
    assert len(requests) == 1
    assert episodes[0]["provider_error"] == code
    assert all(row["provider_error"] == "PROVIDER_CIRCUIT_OPEN" for row in episodes[1:])
    assert "secret-sentinel" not in canonical_json(episodes)
    summary = write_report(tmp_path, episodes, 17)
    assert summary["provider_usage"]["openai"]["unknown_usage_calls"] == 1
    assert replay_report(tmp_path)["passed"]


@pytest.mark.parametrize(
    ("exception", "code"),
    [
        (httpx2.ReadTimeout, "PROVIDER_TIMEOUT"),
        (httpx2.ConnectError, "PROVIDER_CONNECTION"),
    ],
)
def test_network_failures_are_terminal_and_bounded(sdk_factory, scenario, exception, code):
    requests = []

    def fail(request):
        requests.append(request)
        raise exception("secret-sentinel", request=request)

    episode = run_episode(scenario, sdk_factory(fail), "openai")
    assert len(requests) == 1
    assert episode["provider_error"] == code
    assert episode["score"]["termination"] == "agent_error"
    assert "secret-sentinel" not in canonical_json(episode)


def test_shared_request_budget_survives_fresh_agent_instances(sdk_factory, suite, tmp_path):
    requests = []
    factory = sdk_factory(diagnostic_transport(requests), ProviderConfig("gpt-5.6", max_requests=2))
    episodes = evaluate(suite, {"openai": factory})
    assert len(requests) == 2
    assert all(row["provider_error"] == "REQUEST_LIMIT" for row in episodes)
    write_report(tmp_path, episodes, 17)
    assert replay_report(tmp_path)["passed"]


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"status": "incomplete"}, "INCOMPLETE_RESPONSE"),
        ({"status": "failed"}, "INCOMPLETE_RESPONSE"),
        ({"output": []}, "NO_TOOL_CALL"),
        ({"output": None}, "INVALID_PROVIDER_RESPONSE"),
        (
            {
                "output": [
                    {"type": "message", "content": [{"type": "refusal", "refusal": "secret"}]}
                ]
            },
            "MODEL_REFUSAL",
        ),
        (
            {"output": [{"type": "function_call", "name": "probe", "arguments": "{"}]},
            "INVALID_TOOL_ARGUMENTS",
        ),
        (
            {"output": [{"type": "function_call", "name": "probe", "arguments": None}]},
            "INVALID_TOOL_ARGUMENTS",
        ),
        (
            {
                "output": [
                    {
                        "type": "function_call",
                        "name": "probe",
                        "arguments": '{"component":"router","component":"queue"}',
                    }
                ]
            },
            "INVALID_TOOL_ARGUMENTS",
        ),
        (
            {
                "output": [
                    {"type": "function_call", "name": "probe", "arguments": '{"component":NaN}'}
                ]
            },
            "INVALID_TOOL_ARGUMENTS",
        ),
        (
            {
                "output": [
                    {"type": "function_call", "name": "probe", "arguments": "{}"},
                    {"type": "function_call", "name": "verify", "arguments": "{}"},
                ]
            },
            "MULTIPLE_TOOL_CALLS",
        ),
    ],
)
def test_invalid_model_outputs_receive_no_credit(sdk_factory, suite, tmp_path, overrides, code):
    factory = sdk_factory(
        lambda request: httpx2.Response(
            200, json=response_body(action("probe", component="router"), **overrides)
        )
    )
    episodes = evaluate(suite[:1], {"openai": factory})
    assert episodes[0]["provider_error"] == code
    assert not episodes[0]["score"]["verified_success"]
    write_report(tmp_path, episodes, 17, [suite[0].case_id])
    assert replay_report(tmp_path)["passed"]


def test_incomplete_response_preserves_reported_usage(sdk_factory, scenario):
    factory = sdk_factory(
        lambda request: httpx2.Response(
            200, json=response_body(action("probe", component="router"), status="incomplete")
        )
    )
    episode = run_episode(scenario, factory, "openai")
    assert episode["provider_trace"]["calls"][0]["usage"]["total_tokens"] == 120


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"input_tokens": -1},
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 999,
        },
    ],
)
def test_missing_or_inconsistent_usage_stays_unknown(sdk_factory, scenario, usage):
    factory = sdk_factory(
        lambda request: httpx2.Response(200, json=response_body(action("verify"), usage=usage))
    )
    episode = run_episode(scenario, factory, "openai")
    assert all(call["usage"] is None for call in episode["provider_trace"]["calls"])


@pytest.mark.parametrize(
    "changes",
    [
        {"model": None},
        {"model": ""},
        {"model": "  "},
        {"model": "x" * 129},
        {"timeout_seconds": math.nan},
        {"timeout_seconds": math.inf},
        {"timeout_seconds": 0},
        {"timeout_seconds": True},
        {"timeout_seconds": 121},
        {"timeout_seconds": "30"},
        {"max_output_tokens": 255},
        {"max_output_tokens": 4097},
        {"max_output_tokens": True},
        {"max_requests": 0},
        {"max_requests": 193},
        {"max_requests": True},
    ],
)
def test_invalid_provider_configuration_fails_before_network(changes):
    with pytest.raises(ValueError):
        replace(ProviderConfig("gpt-5.6"), **changes)


def test_optional_sdk_and_key_errors_are_actionable(monkeypatch):
    def missing(name):
        raise ImportError("absent")

    with monkeypatch.context() as patch:
        patch.setattr("locus_eval.providers.openai.importlib.import_module", missing)
        with pytest.raises(ValueError, match="provider extra"):
            OpenAIFactory(ProviderConfig("gpt-5.6"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIFactory(ProviderConfig("gpt-5.6"))


def test_factory_pins_endpoint_disables_retries_and_closes_client(monkeypatch):
    seen, closed = {}, []

    def client(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(close=lambda: closed.append(True))

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untrusted.invalid/v1")
    monkeypatch.setattr(openai, "OpenAI", client)
    factory = OpenAIFactory(ProviderConfig("gpt-5.6", timeout_seconds=7.0))
    assert seen["base_url"] == "https://api.openai.com/v1"
    assert seen["max_retries"] == 0 and seen["timeout"] == 7.0
    assert factory().budget is factory().budget
    factory.close()
    assert closed == [True]


def test_paired_worlds_produce_identical_initial_api_payloads(suite):
    first = suite[0]
    other = next(case for case in suite if case.pair_id == first.pair_id and case != first)
    config = ProviderConfig("gpt-5.6")
    assert digest(request_payload(Environment(first).observe(), config)) == digest(
        request_payload(Environment(other).observe(), config)
    )


def test_function_name_and_arguments_are_passed_to_the_environment_unchanged():
    response = SimpleNamespace(
        status="completed",
        output=[SimpleNamespace(type="function_call", name="unauthorized", arguments='{"x":1}')],
    )
    assert response_action(response) == {"tool": "unauthorized", "arguments": {"x": 1}}


def test_unexpected_adapter_errors_are_redacted(scenario):
    def explode(**kwargs):
        raise RuntimeError("secret-sentinel")

    client = SimpleNamespace(responses=SimpleNamespace(create=explode))

    def factory():
        return OpenAIAgent(
            client, ProviderConfig("gpt-5.6"), RequestBudget(10), "test", openai.APIError
        )

    episode = run_episode(scenario, factory, "openai")
    assert episode["provider_error"] == "ADAPTER_ERROR"
    assert "secret-sentinel" not in canonical_json(episode)


def test_error_codes_are_allowlisted():
    with pytest.raises(ValueError, match="Unknown provider"):
        ProviderFailure("secret-sentinel")


def test_provider_cli_runs_selected_cases_closes_client_and_replays(
    sdk_factory,
    tmp_path,
    monkeypatch,
    capsys,
):
    captured, closed = [], []

    class Factory:
        def __init__(self, config):
            self.factory = sdk_factory(diagnostic_transport(captured), config)

        def __call__(self):
            return self.factory()

        def close(self):
            closed.append(True)

    monkeypatch.setattr("locus_eval.cli.OpenAIFactory", Factory)
    assert (
        main(
            [
                "run",
                "--agent",
                "openai",
                "--model",
                "gpt-5.6",
                "--limit",
                "2",
                "--max-requests",
                "16",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert "Episode 2/2" in output.err
    assert json.loads(output.out)["metrics"]["openai"]["episodes"] == 2
    assert closed == [True]
    before = len(captured)
    assert replay_report(tmp_path)["episodes_replayed"] == 2
    assert len(captured) == before
    with pytest.raises(SystemExit) as error:
        main(["run", "--agent", "openai", "--model", "gpt-5.6", "--output", str(tmp_path)])
    assert error.value.code == 2
    assert "empty output directory" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--agent", "diagnostic", "--limit", "0"],
        ["--agent", "diagnostic", "--limit", "25"],
        ["--agent", "diagnostic", "--model", "gpt-5.6"],
        ["--agent", "openai"],
    ],
)
def test_invalid_live_cli_options_exit_before_provider_calls(tmp_path, arguments):
    with pytest.raises(SystemExit) as error:
        main(["run", *arguments, "--output", str(tmp_path)])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "trace_schema",
        "provider",
        "sdk",
        "error",
        "termination",
        "count",
        "call_schema",
        "step",
        "request",
        "action",
        "elapsed",
        "identifier",
        "usage",
        "selection",
    ],
)
def test_provider_metadata_tampering_is_rejected_after_resigning(
    sdk_factory,
    suite,
    tmp_path,
    mutation,
):
    episodes = evaluate(suite[:1], {"openai": sdk_factory(diagnostic_transport([]))})
    write_report(tmp_path, episodes, 17, [suite[0].case_id])
    episode, trace = episodes[0], episodes[0]["provider_trace"]
    call = trace["calls"][0]
    if mutation == "trace_schema":
        trace["extra"] = True
    elif mutation == "provider":
        trace["provider"] = "unknown"
    elif mutation == "sdk":
        trace["sdk_version"] = ""
    elif mutation == "error":
        episode["provider_error"] = "unknown"
    elif mutation == "termination":
        episode["provider_error"] = "REQUEST_LIMIT"
    elif mutation == "count":
        trace["calls"].pop()
    elif mutation == "call_schema":
        call["extra"] = True
    elif mutation == "step":
        call["step"] = 2
    elif mutation == "request":
        call["request_sha256"] = "0" * 64
    elif mutation == "action":
        call["action"] = action("verify")
    elif mutation == "elapsed":
        call["elapsed_ms"] = -1
    elif mutation == "identifier":
        call["response_id"] = 7
    elif mutation == "usage":
        call["usage"]["total_tokens"] = 121
    else:
        path = tmp_path / "summary.json"
        summary = json.loads(path.read_text("utf-8"))
        summary["selected_case_ids"] = []
        path.write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "episodes.jsonl").write_bytes(
        "".join(canonical_json(row) + "\n" for row in episodes).encode()
    )
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    for name in manifest["files"]:
        manifest["files"][name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        replay_report(tmp_path)


def test_provider_failure_receipt_must_match_episode_error(sdk_factory, scenario):
    factory = sdk_factory(
        lambda request: httpx2.Response(
            200, json=response_body(action("verify"), status="incomplete")
        )
    )
    episode = run_episode(scenario, factory, "openai")
    episode["provider_trace"]["calls"][0]["error"] = "NO_TOOL_CALL"
    with pytest.raises(ValueError, match="failure receipt mismatch"):
        validate_trace(episode, [Environment(scenario).observe()])


@pytest.mark.parametrize("mutation", ["mixed", "config", "budget"])
def test_provider_aggregates_reject_inconsistent_runs(sdk_factory, suite, mutation):
    episodes = evaluate(suite[:2], {"openai": sdk_factory(diagnostic_transport([]))})
    if mutation == "mixed":
        del episodes[1]["provider_trace"]
    elif mutation == "config":
        episodes[1]["provider_trace"]["config"]["model"] = "different"
    else:
        for episode in episodes:
            episode["provider_trace"]["config"]["max_requests"] = 1
    with pytest.raises(ValueError):
        summarize_provider_calls(episodes)
