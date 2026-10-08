"""Recheck notation-only holds from cached answers without model calls."""

from __future__ import annotations

import argparse
import asyncio
import collections
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from phy_rl_curate import (
    COMPACT_AUDIT_SCHEMA,
    atomic_json,
    family_key,
    read_rows,
    run_checks,
    screen_batch,
    sha,
    write_rows,
)
from physics_rlvr_data.openrouter import StructuredOutputError, decode_result


async def recover(args) -> None:
    accepted = read_rows(args.run / "accepted.jsonl")
    held = read_rows(args.run / "review.jsonl")
    candidates = (accepted + held if args.all_records else
                  [row for row in held if row.get("failed_checks") == ["equivalent_forms_match_primary"]])
    selected_ids = {row["problem_id"] for row in candidates}
    preserved = [row for row in accepted if row["problem_id"] not in selected_ids]
    patches = json.loads(args.domain_reviews.read_text()) if args.domain_reviews else {}
    binding_reviews = json.loads(args.binding_reviews.read_text()) if args.binding_reviews else {}
    args.output.mkdir(parents=True, exist_ok=True)
    records = args.output / "records"
    records.mkdir(exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(source):
        async with semaphore:
            row = copy.deepcopy(source)
            row.update(training_ready=False, release_status="held", status="review", required_outputs=[])
            original = args.run / "records" / f"{row['problem_id']}.json"
            row["recovery"] = {"method": "notation_normalization_and_full_recheck",
                               "original_checkpoint": str(original),
                               "original_checkpoint_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                               "additional_api_cost_usd": 0}
            if row["problem_id"] in patches:
                patch = patches[row["problem_id"]]
                if patch["question_sha256"] != sha(row["question"]) or not patch["source_evidence"]:
                    raise ValueError("Domain review is not bound to this source question")
                for answer in row["answers"]:
                    answer["assumptions"] = answer.get("assumptions", []) + patch["assumptions_by_label"].get(answer["label"], [])
                row["domain_review"] = patch
            if row["problem_id"] in binding_reviews:
                review = binding_reviews[row["problem_id"]]
                if review["question_sha256"] != sha(row["question"]) or not review["reason"]:
                    raise ValueError("Binding review is not bound to this source question")
                labels = {answer["label"] for answer in row["answers"]}
                if not set(review["answers_by_label"]).issubset(labels):
                    raise ValueError("Binding review refers to an unknown output")
                for answer in row["answers"]:
                    if answer["label"] in review["answers_by_label"]:
                        adjustment = review["answers_by_label"][answer["label"]]
                        if set(adjustment) != {"bindings", "rtol"}:
                            raise ValueError("Binding reviews can change only bindings and relative tolerance")
                        row["recovery"].setdefault("answer_adjustments", {})[answer["label"]] = {
                            "before": {key: answer.get(key) for key in adjustment}, "after": adjustment}
                        answer.update(adjustment)
                row["binding_review"] = review
            if not row.get("audit") and row.get("api_error") == "StructuredOutputError":
                curation = args.run / "responses" / row["problem_id"] / "curation.json"
                audits = [entry for entry in row.get("usage", []) if entry["stage"] == "blind_audit"
                          and not entry.get("truncated_output")]
                if curation.exists() and len(audits) == 1:
                    raw_path = args.run / audits[0]["raw_response_path"]
                    try:
                        audit = decode_result(json.loads(raw_path.read_text()), COMPACT_AUDIT_SCHEMA, None,
                                              audits[0]["max_tokens"])
                    except StructuredOutputError:
                        audit = None
                    if audit is not None:
                        row.update(json.loads(curation.read_text()))
                        row.update(audit=audit, problem_text=row["question"], requires_diagram=not row["self_contained"])
                        row["recovery"]["raw_audit_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                        row["recovery"]["original_api_error"] = row.pop("api_error")
            if not row.get("audit") or not row.get("answers"):
                return row
            result = await run_checks(row, records / f"{row['problem_id']}.json")
            row.update(result.get("curated", {}))
            row["audit"] = result.get("audit", row["audit"])
            row["checks"] = result["checks"]
            row["output_review"] = result.get("output_review", [])
            if result.get("verification_error"):
                row["verification_error"] = result["verification_error"]
            print(f"Rechecked {row['problem_id']}: {all(row['checks'].values())}", flush=True)
            return row

    checked = await asyncio.gather(*(one(row) for row in candidates))
    passing = [row for row in checked if all(row["checks"].values())]
    screen_args = SimpleNamespace(output=args.output, eval_cache=args.eval_cache)
    screens = await screen_batch(screen_args, passing, 0) if passing else {}
    seen = {family_key(row["question"]) for row in preserved}
    recovered = []
    for row in checked:
        if row["problem_id"] in screens:
            screen = screens[row["problem_id"]]
            if screen["question_sha256"] != sha(row["question"]):
                raise ValueError("Recovery screening does not match the question")
            row["benchmark_screening"] = screen
            row["checks"]["final_question_decontamination"] = screen["status"] == "clear"
            family = family_key(row["question"])
            row["checks"]["distinct_question_family"] = family not in seen
            if all(row["checks"].values()):
                row.update(training_ready=True, release_status="ready", status="model_checked",
                           required_outputs=row["answers"])
                recovered.append(row)
                seen.add(family)
        row["failed_checks"] = [key for key, passed in row["checks"].items() if not passed]
        atomic_json(records / f"{row['problem_id']}.json", row)
    recovered_ids = {row["problem_id"] for row in recovered}
    write_rows(args.output / "recovered.jsonl", recovered)
    write_rows(args.output / "accepted.jsonl", preserved + recovered)
    write_rows(args.output / "review.jsonl", [row for row in held if row["problem_id"] not in selected_ids]
               + [row for row in checked if row["problem_id"] not in recovered_ids])
    summary = {"original_run": str(args.run), "rechecked": len(checked), "recovered": len(recovered),
               "accepted_total": len(preserved) + len(recovered),
               "held_total": len(held) + len(accepted) - len(preserved) - len(recovered),
               "newly_recovered": len(recovered_ids - {row["problem_id"] for row in accepted}),
               "additional_api_cost_usd": 0,
               "recovered_by_source": dict(collections.Counter(row["source"] for row in recovered))}
    atomic_json(args.output / "status.json", summary)
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--eval-cache", type=Path, default=Path("/tmp/phy-rl-training-eval-cache"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--all-records", action="store_true", help="Recheck accepted and held rows; preserve original artifacts")
    parser.add_argument("--domain-reviews", type=Path, help="Source-bound reviews of explicit symbol domains")
    parser.add_argument("--binding-reviews", type=Path, help="Source-bound fixed parameters and reviewed approximation tolerances")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 16:
        parser.error("Concurrency must be 1..16")
    asyncio.run(recover(args))


if __name__ == "__main__":
    main()
