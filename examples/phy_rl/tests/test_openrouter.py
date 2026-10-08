"""Budget and response-contract checks without paid API calls."""

import asyncio
import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "data_pipeline"))

from physics_rlvr_data.openrouter import (
    BudgetExceededError,
    IncompleteResponseError,
    OpenRouterClient,
    StructuredOutputError,
    UnconfirmedRequestError,
    decode_result,
    input_token_ceiling,
    provider_schema,
)

SCHEMA = {"type": "object", "properties": {"result": {"type": "number"}},
          "required": ["result"], "additionalProperties": False}
MODEL = {"id": "qwen/test", "pricing": {"prompt": "0.000001", "completion": "0.00001"}}


def test_provider_schema_keeps_length_checks_local():
    schema = {"type": "object", "properties": {"label": {"type": "string", "maxLength": 3}},
              "required": ["label"], "additionalProperties": False}
    portable = provider_schema("google/gemini-3-flash-preview", schema)
    assert portable["properties"]["label"] == {"type": "string"}
    assert schema["properties"]["label"]["maxLength"] == 3
    named = {"properties": {"maxLength": {"type": "string", "maxLength": 3}}}
    assert provider_schema("google/gemini-test", named) == {"properties": {"maxLength": {"type": "string"}}}
    assert provider_schema("qwen/test", schema) == schema
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": '{"label":"long"}'}}]}
    with pytest.raises(StructuredOutputError, match="Schema violation"):
        decode_result(payload, schema, None)
    payload["choices"][0]["message"]["content"] = '```json\n{"label":"ok"}\n```'
    assert decode_result(payload, schema, None) == {"label": "ok"}
    payload["choices"][0]["message"]["content"] = 'Preamble\n```json\n{"label":"ok"}\n```'
    with pytest.raises(StructuredOutputError):
        decode_result(payload, schema, None)


def test_confirmed_truncation_gets_one_budgeted_repair(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    from phy_rl_curate import complete_with_output_repair

    async def run():
        posts = 0

        async def handle(request):
            nonlocal posts
            if request.method == "GET":
                return httpx.Response(200, json={"data": [MODEL]})
            posts += 1
            body = json.loads(request.content)
            payload = response()
            if body["max_tokens"] == 64:
                payload["choices"][0]["finish_reason"] = "length"
            return httpx.Response(200, json=payload)

        async with OpenRouterClient("fake", tmp_path / "usage.json", budget=1,
                                    concurrency=1, requests_per_minute=600000,
                                    transport=httpx.MockTransport(handle)) as client:
            args = SimpleNamespace(max_output_repairs=1)
            call = dict(stage="audit", problem_id="one", max_tokens=64, effort="none", schema=SCHEMA)
            assert await complete_with_output_repair(client, args, MODEL["id"], "solve", **call) == {"result": 42}
            assert posts == 2 and client.spent == 0.002
            assert client.ledger[0]["truncated_output"]
            assert client.ledger[1]["stage"] == "audit_output_repair"
            assert await complete_with_output_repair(client, args, MODEL["id"], "solve", **call) == {"result": 42}
            assert posts == 2
            args.max_output_repairs = 0
            with pytest.raises(IncompleteResponseError):
                await complete_with_output_repair(client, args, MODEL["id"], "solve", **call)
            assert posts == 2

    asyncio.run(run())


def test_image_reservation_uses_pixels_instead_of_encoded_bytes():
    picture = Image.new("RGB", (1200, 1700))
    raw = io.BytesIO()
    picture.save(raw, format="PNG")
    encoded = base64.b64encode(raw.getvalue()).decode()
    url = "data:image/png;base64," + encoded
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Transcribe this page"},
                                                          {"type": "image_url", "image_url": {"url": url}}]}]}
    value = input_token_ceiling("google/gemini-3-flash-preview", body, [url])
    assert 6 * 4096 + 2048 < value < 6 * 4096 + 2300
    assert input_token_ceiling("qwen/test", body, [url]) == value


def response(content: str = '{"result":42}', *, cost: float = 0.001) -> dict:
    return {"id": "fake", "model": MODEL["id"], "usage": {"cost": cost},
            "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
                {"function": {"name": "submit_result", "arguments": content}}
            ]}}]}


def test_tool_result_requires_complete_strict_schema() -> None:
    assert decode_result(response(), SCHEMA, "submit_result") == {"result": 42}
    for content in ['{"result":1,"result":2}', '{"result":NaN}', '{"result":"42"}',
                    '{"result":42,"extra":1}', 'Result: {"result":42}']:
        with pytest.raises(StructuredOutputError):
            decode_result(response(content), SCHEMA, "submit_result")
    truncated = response()
    truncated["choices"][0]["finish_reason"] = "length"
    with pytest.raises(IncompleteResponseError):
        decode_result(truncated, SCHEMA, "submit_result")
    ceiling = response('{"result":')
    ceiling["usage"]["completion_tokens"] = 32
    with pytest.raises(IncompleteResponseError):
        decode_result(ceiling, SCHEMA, "submit_result", max_tokens=32)
    assert decode_result(response(), SCHEMA, "submit_result", max_tokens=32) == {"result": 42}
    for message in [None, {"tool_calls": None}, {"tool_calls": [None]}, {"tool_calls": [{"function": None}]}]:
        malformed = response()
        malformed["choices"][0]["message"] = message
        with pytest.raises(StructuredOutputError):
            decode_result(malformed, SCHEMA, "submit_result")


def test_concurrent_requests_reserve_budget_before_submission(tmp_path: Path) -> None:
    async def run():
        active = peak = posts = 0

        async def handle(request):
            nonlocal active, peak, posts
            if request.method == "GET":
                return httpx.Response(200, json={"data": [MODEL]})
            active += 1
            posts += 1
            peak = max(peak, active)
            body = json.loads(request.content)
            assert body["tool_choice"]["function"]["name"] == "submit_result"
            assert body["provider"]["max_price"] == {"prompt": 1, "completion": 10}
            await asyncio.sleep(0.02)
            active -= 1
            return httpx.Response(200, json=response())

        async with OpenRouterClient("fake", tmp_path / "usage.json", budget=0.006,
                                    concurrency=8, requests_per_minute=600000,
                                    transport=httpx.MockTransport(handle)) as client:
            results = await asyncio.gather(*(client.complete(MODEL["id"], str(i), stage="audit",
                problem_id=str(i), schema=SCHEMA, max_tokens=32) for i in range(8)), return_exceptions=True)
            assert any(isinstance(r, BudgetExceededError) for r in results)
            assert all(isinstance(r, (dict, BudgetExceededError)) for r in results)
            assert client.spent <= 0.006
            assert not client.unresolved
            assert posts == len(client.ledger)
            assert 1 <= peak <= 2
        assert len(json.loads((tmp_path / "usage.json").read_text())) == posts
    asyncio.run(run())


def test_invalid_billed_result_is_cached_without_paid_retry(tmp_path: Path) -> None:
    async def run():
        posts = 0

        async def handle(request):
            nonlocal posts
            if request.method == "GET":
                return httpx.Response(200, json={"data": [MODEL]})
            posts += 1
            return httpx.Response(200, json=response('{"result":"wrong type"}'))

        async with OpenRouterClient("fake", tmp_path / "usage.json",
                                    transport=httpx.MockTransport(handle)) as client:
            for _ in range(2):
                with pytest.raises(StructuredOutputError):
                    await client.complete(MODEL["id"], "same", stage="audit", problem_id="p", schema=SCHEMA)
            assert posts == 1
            assert client.spent == 0.001
            assert client.ledger[0]["structured_output_valid"] is False
            assert not client.unresolved
    asyncio.run(run())


def test_official_qwen_flash_uses_native_json_without_forced_tools(tmp_path: Path) -> None:
    async def run():
        model = {"id": "qwen/qwen3.5-flash-02-23", "pricing": MODEL["pricing"]}

        async def handle(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [model]})
            body = json.loads(request.content)
            assert "tools" not in body and "tool_choice" not in body
            assert body["response_format"]["json_schema"]["schema"] == SCHEMA
            assert body["provider"]["only"] == ["alibaba"]
            return httpx.Response(200, json={"id": "fake-json", "model": model["id"], "usage": {"cost": 0.001},
                "choices": [{"finish_reason": "stop", "message": {"content": '{"result":42}'}}]})

        async with OpenRouterClient("fake", tmp_path / "usage.json", transport=httpx.MockTransport(handle),
                                    qwen_providers=("alibaba",)) as client:
            assert await client.complete(model["id"], "audit", stage="audit", problem_id="p", schema=SCHEMA) == {"result": 42}
            assert client.spent == 0.001 and not client.unresolved
            raw = io.BytesIO()
            Image.new("RGB", (8, 8)).save(raw, format="PNG")
            image = "data:image/png;base64," + base64.b64encode(raw.getvalue()).decode()
            assert await client.complete(model["id"], "audit", stage="audit", problem_id="p", schema=SCHEMA,
                                         image_urls=[image]) == {"result": 42}
            assert len(client.ledger) == 2
            assert await client.complete(model["id"], "audit", stage="audit", problem_id="p", schema=SCHEMA,
                                         image_urls=[image]) == {"result": 42}
            assert len(client.ledger) == 2
            assert client.ledger[0]["request_fingerprint"] != client.ledger[1]["request_fingerprint"]
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["timeout", "server"])
def test_uncertain_request_is_reserved_and_never_repeated(tmp_path: Path, failure: str) -> None:
    async def run():
        posts = 0

        async def handle(request):
            nonlocal posts
            if request.method == "GET":
                return httpx.Response(200, json={"data": [MODEL]})
            posts += 1
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            return httpx.Response(503)

        async with OpenRouterClient("fake", tmp_path / "usage.json",
                                    transport=httpx.MockTransport(handle)) as client:
            for _ in range(2):
                with pytest.raises(UnconfirmedRequestError):
                    await client.complete(MODEL["id"], "same", stage="audit", problem_id="p", schema=SCHEMA)
            assert posts == 1
            assert len(client.unresolved) == 1
            assert client.unresolved[0]["reserved_cost_usd"] > 0
            assert not client.active_reservations
    asyncio.run(run())


def test_rate_limit_backoff_retries_rejected_request_only(tmp_path: Path) -> None:
    async def run():
        posts = 0

        async def handle(request):
            nonlocal posts
            if request.method == "GET":
                return httpx.Response(200, json={"data": [MODEL]})
            posts += 1
            if posts == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json=response())

        async with OpenRouterClient("fake", tmp_path / "usage.json", requests_per_minute=600000,
                                    transport=httpx.MockTransport(handle)) as client:
            assert await client.complete(MODEL["id"], "p", stage="audit", problem_id="p", schema=SCHEMA) == {"result": 42}
            assert posts == 2
            assert client.gates[MODEL["id"]].rate_limits == 1
            assert client.gates[MODEL["id"]].limit == 2
            assert len(client.ledger) == 1
            assert client.spent == 0.001
    asyncio.run(run())


def test_cancellation_propagates_and_keeps_charge_reserved(tmp_path: Path) -> None:
    async def run():
        submitted = asyncio.Event()

        async def handle(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [MODEL]})
            submitted.set()
            await asyncio.sleep(10)
            return httpx.Response(200, json=response())

        async with OpenRouterClient("fake", tmp_path / "usage.json",
                                    transport=httpx.MockTransport(handle)) as client:
            task = asyncio.create_task(client.complete(MODEL["id"], "p", stage="audit", problem_id="p", schema=SCHEMA))
            await submitted.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert client.unresolved[0]["reason"] == "cancelled_request"
            assert not client.active_reservations
    asyncio.run(run())
