from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math
import random
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
from jsonschema import Draft202012Validator
from PIL import Image


def provider_schema(model: str, schema: dict | None) -> dict | None:
    """Keep bounded collection and string limits in local validation for Gemini."""
    if not schema or not model.startswith("google/gemini-"):
        return schema

    def convert(value, property_map=False):
        if isinstance(value, dict):
            if property_map:
                return {key: convert(item) for key, item in value.items()}
            return {key: convert(item, key in {"properties", "$defs", "definitions", "patternProperties"})
                    for key, item in value.items() if key not in {"minLength", "maxLength", "minItems", "maxItems"}}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(schema)


def input_token_ceiling(model: str, body: dict, images: list[str] | None) -> int:
    if not images:
        return len(json.dumps(body, ensure_ascii=False).encode()) + 2048
    text_body = dict(body, messages=[{"role": "user", "content": body["messages"][-1]["content"][0]["text"]}])
    ceiling = len(json.dumps(text_body, ensure_ascii=False).encode()) + 2048
    for image in images:
        raw = base64.b64decode(image.split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as decoded:
            width, height = decoded.size
        # Reserve well above documented image-tile budgets; base64 is transport, not text tokens.
        ceiling += 4096 * math.ceil(width / 768) * math.ceil(height / 768)
    return ceiling


class IncompleteResponseError(ValueError):
    """A billed response ended before its structured result was complete."""


class StructuredOutputError(ValueError):
    """A completed response violates the requested output contract."""


class BudgetExceededError(RuntimeError):
    pass


class UnconfirmedRequestError(RuntimeError):
    pass


class OpenRouterHTTPError(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")
        file.flush()


def strict_json(text: str) -> dict:
    def unique(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise StructuredOutputError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise StructuredOutputError(f"Non-finite JSON constant: {value}")

    try:
        result = json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"Invalid JSON: {exc.msg}") from None
    if not isinstance(result, dict):
        raise StructuredOutputError("Response must be a JSON object")
    return result


def decode_result(payload: dict, schema: dict | None, tool_name: str | None, max_tokens: int | None = None,
                  text: bool = False) -> dict:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise StructuredOutputError("Expected one completion choice")
    choice = choices[0]
    reason = choice.get("finish_reason")
    if reason not in ({"tool_calls", "stop"} if tool_name else {"stop"}):
        raise IncompleteResponseError(f"Incomplete response: {reason}")
    message = choice.get("message", {})
    if not isinstance(message, dict):
        raise StructuredOutputError("Completion message must be an object")
    if tool_name:
        calls = message.get("tool_calls", [])
        if (not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict)
                or not isinstance(calls[0].get("function"), dict)
                or calls[0]["function"].get("name") != tool_name):
            raise StructuredOutputError("Expected exactly one required result tool call")
        content = calls[0]["function"].get("arguments")
    else:
        content = message.get("content")
    if not isinstance(content, str):
        raise StructuredOutputError("Structured response content must be a string")
    if text:
        return {"text": content}
    fence = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", content, re.DOTALL)
    if fence and tool_name is None:
        content = fence[1]
    try:
        result = strict_json(content)
    except StructuredOutputError as exc:
        if max_tokens is not None and payload.get("usage", {}).get("completion_tokens", 0) >= max_tokens:
            raise IncompleteResponseError("Malformed structured output at the token ceiling") from exc
        raise
    if schema:
        errors = sorted(Draft202012Validator(schema).iter_errors(result), key=lambda e: str(e.path))
        if errors:
            error = errors[0]
            location = ".".join(str(part) for part in error.path) or "root"
            raise StructuredOutputError(f"Schema violation at {location}: {error.message[:160]}")
    return result


def retry_delay(header: str, attempt: int) -> float:
    if header.replace(".", "", 1).isdigit():
        return max(0, float(header))
    if header:
        try:
            return max(0, (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            pass
    return 2 ** (attempt + 1)


class RequestGate:
    """Pace request starts and reduce concurrency when a provider rejects traffic."""

    def __init__(self, maximum: int, requests_per_minute: float, initial: int = 4):
        if maximum < 1 or requests_per_minute <= 0:
            raise ValueError("Concurrency and request rate must be positive")
        self.maximum = maximum
        self.limit = min(initial, maximum)
        self.interval = 60 / requests_per_minute
        self.active = 0
        self.peak = 0
        self.successes = 0
        self.rate_limits = 0
        self.next_start = 0.0
        self.cooldown_until = 0.0
        self.condition = asyncio.Condition()

    async def __aenter__(self) -> None:
        while True:
            async with self.condition:
                if self.active >= self.limit:
                    await self.condition.wait()
                    continue
                delay = max(self.next_start, self.cooldown_until) - time.monotonic()
                if delay <= 0:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                    self.next_start = time.monotonic() + self.interval
                    return
            await asyncio.sleep(delay)

    async def __aexit__(self, *_: Any) -> None:
        async with self.condition:
            self.active -= 1
            self.condition.notify_all()

    async def success(self) -> None:
        async with self.condition:
            self.successes += 1
            if self.successes >= 8:
                self.limit = min(self.maximum, self.limit + 1)
                self.successes = 0
            self.condition.notify_all()

    async def throttled(self, retry_after: float) -> None:
        async with self.condition:
            self.rate_limits += 1
            self.limit = max(1, self.limit // 2)
            self.successes = 0
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + retry_after)
            self.condition.notify_all()


class OpenRouterClient:
    def __init__(
        self, api_key: str, ledger_path: Path, *, budget: float = 0.50,
        concurrency: int = 4, requests_per_minute: float = 60,
        transport: httpx.AsyncBaseTransport | None = None,
        qwen_providers: tuple[str, ...] = ("Parasail", "Darkbloom"), request_timeout: float = 240,
        initial_concurrency: int = 4,
    ):
        self.api_key = api_key
        self.ledger_path = ledger_path
        self.journal_path = ledger_path.with_suffix(".jsonl")
        if self.journal_path.exists():
            self.ledger = [json.loads(line) for line in self.journal_path.read_text().splitlines() if line.strip()]
        else:
            self.ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
            for entry in self.ledger:
                append_jsonl(self.journal_path, entry)
        self.unresolved_path = ledger_path.with_name("unresolved_requests.json")
        self.unresolved = json.loads(self.unresolved_path.read_text()) if self.unresolved_path.exists() else []
        self.budget = Decimal(str(budget))
        self.active_reservations: set[str] = set()
        self.accounting = asyncio.Condition()
        # One gate per model, so a rate limit from one provider does not slow the others.
        self.gates = defaultdict(lambda: RequestGate(concurrency, requests_per_minute, initial_concurrency))
        self.request_timeout = request_timeout
        self.http = httpx.AsyncClient(
            transport=transport, timeout=httpx.Timeout(request_timeout, connect=15, write=120, pool=15),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=concurrency, keepalive_expiry=60),
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        )
        self.models: dict[str, dict] = {}
        self.qwen_providers = qwen_providers
        self.stop_reason: str | None = None

    async def __aenter__(self) -> OpenRouterClient:
        if self.api_key:
            for attempt in range(3):
                try:
                    response = await self.http.get("https://openrouter.ai/api/v1/models")
                    response.raise_for_status()
                    break
                except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                    retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in {429, 500, 502, 503, 504}
                    if not retryable or attempt == 2:
                        await self.http.aclose()
                        raise
                    await asyncio.sleep(2 ** attempt)
            self.models = {model["id"]: model for model in response.json()["data"]}
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.flush()
        await self.http.aclose()

    @property
    def spent(self) -> float:
        return float(sum((Decimal(str(entry["charged_cost_usd"])) for entry in self.ledger), Decimal(0)))

    def flush(self) -> None:
        atomic_json(self.ledger_path, self.ledger)

    async def reserve(self, request_id: str, metadata: dict, amount: Decimal) -> None:
        async with self.accounting:
            while True:
                if self.stop_reason:
                    raise UnconfirmedRequestError(self.stop_reason)
                if any(e.get("request_fingerprint") == metadata["request_fingerprint"] for e in self.unresolved):
                    raise UnconfirmedRequestError("Matching request has an unconfirmed charge; not repeated")
                pending = sum((Decimal(str(e["reserved_cost_usd"])) for e in self.unresolved), Decimal(0))
                charged = sum((Decimal(str(e["charged_cost_usd"])) for e in self.ledger), Decimal(0))
                if charged + pending + amount <= self.budget:
                    self.unresolved.append(dict(metadata, local_request_id=request_id, reserved_cost_usd=float(amount), reason="in_flight"))
                    self.active_reservations.add(request_id)
                    atomic_json(self.unresolved_path, self.unresolved)
                    return
                if not self.active_reservations:
                    raise BudgetExceededError(f"Budget reached: ${charged} charged and ${pending} unconfirmed")
                await self.accounting.wait()

    async def settle(self, request_id: str, entry: dict | None = None, *, unknown: str | None = None) -> None:
        async with self.accounting:
            reservation = next(e for e in self.unresolved if e.get("local_request_id") == request_id)
            if entry:
                append_jsonl(self.journal_path, entry)
                self.ledger.append(entry)
                if Decimal(str(entry["charged_cost_usd"])) > Decimal(str(reservation["reserved_cost_usd"])):
                    self.stop_reason = "Returned charge exceeds the reserved ceiling; accounting review required"
            if unknown:
                reservation["reason"] = unknown
            else:
                self.unresolved.remove(reservation)
            self.active_reservations.discard(request_id)
            atomic_json(self.unresolved_path, self.unresolved)
            self.accounting.notify_all()

    async def complete(
        self, model: str, prompt: str, *, stage: str, problem_id: str,
        max_tokens: int = 4096, effort: str = "low", schema: dict | None = None,
        image_urls: list[str] | None = None, system: str | None = None,
        temperature: float = 0, text: bool = False,
    ) -> dict[str, Any]:
        tool_name = ("submit_result" if model.startswith("qwen/") and schema
                     and model != "qwen/qwen3.5-flash-02-23" else None)
        fingerprint_data = {
            "model": model, "prompt": prompt, "stage": stage, "max_tokens": max_tokens,
            "effort": effort, "schema": schema, "tool": tool_name,
        }
        if system or temperature or text:
            fingerprint_data.update(system=system, temperature=temperature, text=text)
        wire_schema = provider_schema(model, schema)
        if wire_schema != schema:
            fingerprint_data["provider_schema"] = wire_schema
        if image_urls:
            if any(not url.startswith("data:image/") for url in image_urls):
                raise ValueError("Source images must be inline image data URLs")
            fingerprint_data["image_sha256"] = [hashlib.sha256(url.encode()).hexdigest() for url in image_urls]
        fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
        # A provider-side error is transient, so only completed responses are replayed.
        cached = next((e for e in self.ledger if e.get("request_fingerprint") == fingerprint
                       and e.get("finish_reason") != "error"), None)
        if cached:
            payload = json.loads((self.ledger_path.parent / cached["raw_response_path"]).read_text())
            return decode_result(payload, schema, tool_name, max_tokens, text)
        if not self.api_key:
            raise RuntimeError(f"Offline run has no cached response for {problem_id}/{stage}")
        if schema:
            Draft202012Validator.check_schema(schema)
        pricing = self.models[model]["pricing"]
        input_rate = Decimal(pricing["prompt"])
        output_rate = Decimal(pricing["completion"])
        provider = {"require_parameters": True, "sort": "throughput", "max_price": {
            "prompt": float(input_rate * 1_000_000), "completion": float(output_rate * 1_000_000),
        }}
        content = ([{"type": "text", "text": prompt},
                    *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls]]
                   if image_urls else prompt)
        messages = [{"role": "system", "content": system}] if system else []
        body = {"model": model, "messages": [*messages, {"role": "user", "content": content}],
                "max_tokens": max_tokens, "temperature": temperature,
                "reasoning": {"enabled": False} if effort == "none" else {"effort": effort, "exclude": True},
                "usage": {"include": True}, "provider": provider}
        if model.startswith("openai/"):
            body.pop("temperature")  # OpenAI reasoning endpoints reject sampling parameters.
        if model.startswith("qwen/") and schema:
            provider["only"] = list(self.qwen_providers)
        if tool_name:
            body["tools"] = [{"type": "function", "function": {
                "name": tool_name, "description": "Return the completed physics result. No prose or other tool calls.",
                "strict": True, "parameters": wire_schema,
            }}]
            body["tool_choice"] = {"type": "function", "function": {"name": tool_name}}
        elif not text:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": stage, "strict": True, "schema": wire_schema,
            }} if schema else {"type": "json_object"}
        input_ceiling = input_token_ceiling(model, body, image_urls)
        reserve = Decimal(input_ceiling) * input_rate + Decimal(max_tokens) * output_rate
        request_id = uuid.uuid4().hex
        metadata = {"problem_id": problem_id, "stage": stage, "model": model, "request_fingerprint": fingerprint}
        await self.reserve(request_id, metadata, reserve)
        started = time.monotonic()
        gate = self.gates[model]
        for attempt in range(3):
            try:
                async with gate:
                    async with asyncio.timeout(self.request_timeout + 60):
                        response = await self.http.post("https://openrouter.ai/api/v1/chat/completions", json=body)
            except asyncio.CancelledError:
                await asyncio.shield(self.settle(request_id, unknown="cancelled_request"))
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteTimeout) as exc:
                # The request never fully reached OpenRouter, so it cannot have been charged.
                if attempt == 2:
                    await self.settle(request_id)
                    raise OpenRouterHTTPError(f"Could not reach OpenRouter: {type(exc).__name__}") from exc
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            except (httpx.RequestError, TimeoutError) as exc:
                await self.settle(request_id, unknown=type(exc).__name__)
                raise UnconfirmedRequestError("Request interrupted; charge reserved and request not repeated") from exc
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After", "")
            delay = retry_delay(retry_after, attempt)
            delay += random.uniform(0, 0.5)
            await gate.throttled(delay)
            append_jsonl(self.ledger_path.with_name("request_events.jsonl"), dict(metadata, event="rate_limit", retry=attempt, delay_seconds=delay))
            if attempt == 2:
                await self.settle(request_id)
                raise OpenRouterHTTPError("OpenRouter rate limit persisted after bounded backoff")
        if response.is_error:
            if response.status_code >= 500:
                await self.settle(request_id, unknown=f"http_{response.status_code}")
                raise UnconfirmedRequestError(f"OpenRouter HTTP {response.status_code}; charge reserved and request not repeated")
            await self.settle(request_id)
            detail = ""
            try:
                error = response.json().get("error", {})
                detail = str(error.get("message", ""))[:400]
                metadata = error.get("metadata", {})
                raw_error = str(metadata.get("raw", "")) if isinstance(metadata, dict) else ""
                if raw_error:
                    detail += "; " + raw_error[:1600]
            except json.JSONDecodeError:
                detail = "Unreadable provider error body"
            detail = detail.replace(self.api_key, "[redacted]")
            detail = re.sub(r"(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,})", "[redacted]", detail)
            raise OpenRouterHTTPError(f"OpenRouter HTTP {response.status_code}; {detail}; no automatic inference retry")
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            await self.settle(request_id, unknown="invalid_response_envelope")
            raise UnconfirmedRequestError("Unreadable response envelope; charge remains reserved") from exc
        if not isinstance(payload, dict):
            await self.settle(request_id, unknown="invalid_response_envelope")
            raise UnconfirmedRequestError("Invalid response envelope; charge remains reserved")
        raw_relative = f"raw_responses/{request_id}.json"
        atomic_json(self.ledger_path.parent / raw_relative, payload)
        usage = payload.get("usage", {})
        cost = usage.get("cost")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            await self.settle(request_id, unknown="missing_billed_cost")
            self.stop_reason = "Response omitted its billed cost; accounting review required"
            raise UnconfirmedRequestError(self.stop_reason)
        output_error = None
        decoded = None
        try:
            decoded = decode_result(payload, schema, tool_name, max_tokens, text)
        except (IncompleteResponseError, StructuredOutputError) as exc:
            output_error = exc
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        entry = {"request_id": payload.get("id"), "problem_id": problem_id, "stage": stage,
                 "requested_model": model, "served_model": payload.get("model"), "provider": payload.get("provider"),
                 "usage": usage, "charged_cost_usd": cost, "seconds": round(time.monotonic() - started, 3),
                 "finish_reason": choice.get("finish_reason"), "pricing_at_request": pricing,
                 "token_ceiling_reached": usage.get("completion_tokens", 0) >= max_tokens,
                 "truncated_output": isinstance(output_error, IncompleteResponseError),
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "request_fingerprint": fingerprint,
                 "reasoning_effort": effort, "max_tokens": max_tokens, "raw_response_path": raw_relative,
                 "structured_output_valid": output_error is None, "output_error": str(output_error) if output_error else None,
                 "rate_limit_retries": attempt}
        message = choice.get("message")
        if not tool_name and isinstance(message, dict) and isinstance(message.get("content"), str):
            entry["whole_json_fence_removed"] = bool(re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*",
                message["content"], re.DOTALL))
        await self.settle(request_id, entry)
        if output_error:
            raise output_error
        await gate.success()
        return decoded
