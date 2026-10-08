"""Check the Qwen result-tool contract on cached training-source statements."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import getpass
import json
import os
from pathlib import Path

from phy_rl_pilot import AUDIT_PROMPT, AUDIT_SCHEMA, OpenRouterClient, normalize_model_fields, statement_from_tex
from physics_rlvr_common.verifier import answer_from_dict, verify_answer
from physics_rlvr_data.openrouter import IncompleteResponseError, StructuredOutputError, atomic_json


async def run(args, key: str) -> None:
    artifact = json.loads((args.dataset / "pilot.json").read_text())
    ids = args.source_ids.split(",")
    records = {row["source_id"]: row for row in artifact["records"]}
    async with OpenRouterClient(key, args.dataset / "usage.json", budget=args.budget,
                                concurrency=3, requests_per_minute=60) as client:
        starting_cost = client.spent

        async def check(source_id: str) -> dict:
            row = records[source_id]
            prompt = AUDIT_PROMPT + "\nUse these output labels: " + json.dumps([a["label"] for a in row["answers"]])
            prompt += "\nORIGINAL STATEMENT:\n" + statement_from_tex(row["original_source"])
            prompt += "\nENGLISH STATEMENT:\n" + row["question"]
            try:
                result = await client.complete("qwen/qwen3.5-35b-a3b", prompt, stage="tool_smoke_audit",
                                               problem_id=row["problem_id"], max_tokens=4096, effort="none", schema=AUDIT_SCHEMA)
            except (IncompleteResponseError, StructuredOutputError) as exc:
                return {"source_id": source_id, "tool_contract_passed": False, "error": str(exc)}
            audit = normalize_model_fields(result)
            by_label = {a["label"]: a for a in audit["answers"]}
            matches = {a["label"]: a["label"] in by_label and verify_answer(by_label[a["label"]], answer_from_dict(a))
                       for a in row["answers"]}
            return {"source_id": source_id, "tool_contract_passed": True,
                    "output_coverage": len(by_label) == len(audit["answers"]) and set(by_label) == set(matches),
                    "answer_agreement": matches, "audit": audit}

        results = await asyncio.gather(*(check(source_id) for source_id in ids))
        report = {"results": results, "tool_contract_passed": all(r["tool_contract_passed"] for r in results),
                  "cost_usd": client.spent - starting_cost, "cumulative_reported_cost_usd": client.spent,
                  "unconfirmed_reserve_usd": sum(r["reserved_cost_usd"] for r in client.unresolved),
                  "peak_in_flight": max((g.peak for g in client.gates.values()), default=0), "rate_limit_events": sum(g.rate_limits for g in client.gates.values()),
                  "budget_usd": args.budget}
        atomic_json(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2), flush=True)
        for result in results:
            print(json.dumps({k: v for k, v in result.items() if k != "audit"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-ids", default="2007-v3g-03,2016-lahg-06,2016-v3g-03")
    parser.add_argument("--budget", type=float, default=0.50)
    args = parser.parse_args()
    with (args.dataset / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter key (hidden): ")
        asyncio.run(run(args, key))


if __name__ == "__main__":
    main()
