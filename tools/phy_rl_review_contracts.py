"""Review existing competition releases against their requested output scope."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from phy_rl_curate import CONTRACT_PROMPT, CONTRACT_SCHEMA, contract_checks, read_rows, run_checks, write_rows
from physics_rlvr_data.openrouter import OpenRouterClient, atomic_json


async def run(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)
    accepted, held = [], []
    async with OpenRouterClient(os.environ["OPENROUTER_API_KEY"], args.output / "usage.json",
                                budget=args.budget, concurrency=2, requests_per_minute=30) as client:
        for original in rows:
            row = dict(original)
            prompt_input = {"question": row["question"],
                            "proposed_output_labels": [answer["label"] for answer in row["answers"]]}
            contract = await client.complete("google/gemini-2.5-flash", CONTRACT_PROMPT + json.dumps(prompt_input),
                stage="release_contract_review", problem_id=row["problem_id"], max_tokens=2048,
                effort="low", schema=CONTRACT_SCHEMA)
            row["contract_review"] = contract
            row["checks"] = dict(row["checks"], required_output_contract=contract_checks(row["question"], row["answers"], contract))
            row["contract_review_provenance"] = {"original_release_file": str(args.input),
                "original_release_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "checker_model": "google/gemini-2.5-flash"}
            row.update(training_ready=False, release_status="held", status="review", required_outputs=[])
            result = await run_checks(row, args.output / "records" / (row["problem_id"] + ".json"))
            row["checks"].update(result["checks"])
            row["output_review"] = result.get("output_review", [])
            if all(row["checks"].values()):
                row.update(training_ready=True, release_status="ready", status="model_checked", required_outputs=row["answers"])
                accepted.append(row)
            else:
                held.append(row)
            row["failed_checks"] = [key for key, passed in row["checks"].items() if not passed]
            atomic_json(args.output / "records" / (row["problem_id"] + ".json"), row)
            write_rows(args.output / "accepted.jsonl", accepted)
            write_rows(args.output / "review.jsonl", held)
            atomic_json(args.output / "status.json", {"state": "running", "input_count": len(rows),
                "completed": len(accepted) + len(held), "accepted": len(accepted), "held": len(held),
                "charged_cost_usd": client.spent, "hard_cap_usd": args.budget})
            print(row["problem_id"], row["failed_checks"], flush=True)
        atomic_json(args.output / "status.json", {"state": "complete", "input_count": len(rows),
            "completed": len(rows), "accepted": len(accepted), "held": len(held),
            "charged_cost_usd": client.spent, "hard_cap_usd": args.budget})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=float, default=0.10)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
