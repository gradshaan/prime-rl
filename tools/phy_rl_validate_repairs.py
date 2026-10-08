"""Validate repaired competition extraction and curation within the existing cap."""

from __future__ import annotations

import argparse
import collections
import fcntl
import json
import signal
from pathlib import Path

from phy_rl_competition_run import CompetitionRun, select_sources
from phy_rl_curate import DATA, read_rows, write_rows
from physics_rlvr_data.openrouter import atomic_json

PARENTS = {
    "IPhO_1967_problem_4", "WoPhO_2013_problem_1", "NBPhO_2022_problem_2",
    "APhO_2003_problem_1", "NBPhO_2022_problem_3", "NBPhO_2023_problem_5",
    "NBPhO_2023_problem_7", "APhO_2012_problem_2",
}


def run(args) -> None:
    workflow = CompetitionRun(args)
    for shutdown_signal in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(shutdown_signal, workflow.request_stop)
    root = args.output / ("forward_pilot" if args.fresh else "repair_validation")
    extraction, audit = root / "extraction", root / "audit"
    root.mkdir(parents=True, exist_ok=True)
    spent, reserved = workflow.accounting()
    allocated_charge = sum(row["charged_cost_usd"] for path in root.rglob("usage.jsonl") for row in read_rows(path))
    extraction_cap, curation_cap = (0.30, 0.60) if args.fresh else (0.45, 0.65)
    allocation = extraction_cap + curation_cap
    if args.budget - spent - reserved < allocation - allocated_charge:
        raise RuntimeError(f"Validation allocations require ${allocation:.2f} of remaining workflow headroom")
    inventory = read_rows(DATA / "competition_5k/source_queue.jsonl")
    queue = root / "source_input.jsonl"
    if args.fresh and queue.exists():
        sources = read_rows(queue)
        if len(sources) != args.parents:
            raise ValueError("Fresh pilot size differs from the saved source manifest")
    elif args.fresh:
        excluded = {json.loads(path.read_text())["parent_problem_id"]
                    for path in args.output.rglob("transcriptions/*.json")}
        excluded.update(json.loads(path.read_text())["parent_problem_id"]
                        for path in (args.repaired_sources / "transcriptions").glob("*.json"))
        sources = select_sources([row for row in inventory if row["year"] >= 2000], excluded, args.parents)
        if len(sources) != args.parents:
            raise ValueError("Not enough unused competition parents for this pilot")
        write_rows(queue, sources)
    else:
        sources = [row for row in inventory if row["problem_id"] in PARENTS]
        if len(sources) != len(PARENTS):
            raise ValueError("Validation identities are absent from the pinned inventory")
        write_rows(queue, sources)
    atomic_json(root / "status.json", {"state": "running", "phase": "checking_question_and_solution_pages",
                "original_parents": len(sources), "maximum_tasks": 10,
                "extraction_cap_usd": extraction_cap, "curation_cap_usd": curation_cap,
                "workflow_cap_usd": args.budget, "starting_charged_usd": spent,
                "starting_unknown_reserve_usd": reserved})
    try:
        workflow.phase = "validating_source_solution_transcription"
        reuse = [args.repaired_sources, *sorted((args.output / "batches").glob("*/extraction"))]
        arguments = [str(extraction), "--input-queue", str(queue), "--limit", str(len(sources)),
                     "--budget", str(extraction_cap), "--pilot-year-min", "1967", "--concurrency", "3"]
        for directory in reuse:
            arguments.extend(["--reuse-transcriptions", str(directory)])
        workflow.command("phy_rl_extract_archives.py", arguments, extraction)
        result = json.loads((extraction / "status.json").read_text())
        if result["unconfirmed_reserve_usd"] or result["state"] == "stopped":
            raise RuntimeError("Source validation stopped; inspect its holds and cost ledger")
        candidates = read_rows(extraction / "curation_queue.jsonl")
        groups = collections.defaultdict(collections.deque)
        for row in candidates:
            groups[row["parent_problem_id"]].append(row)
        selected = []
        while len(selected) < 10 and any(groups.values()):
            for parent in sorted(groups):
                if groups[parent] and len(selected) < 10:
                    selected.append(groups[parent].popleft())
        if not selected:
            raise RuntimeError("No source-verified candidates; review transcription before further spending")
        physics_queue = root / "curation_input.jsonl"
        write_rows(physics_queue, selected)
        workflow.command("phy_rl_curate.py", ["prepare", str(audit), "--count", str(len(selected)),
            "--pool", "competition", "--candidate-file", str(physics_queue), "--only-candidate-files",
            "--minimum-olympiad-fraction", "1", "--minimum-ipho-count", "0"], audit)
        workflow.phase = "validating_complete_physics_outputs"
        workflow.command("phy_rl_curate.py", ["run", str(audit), "--budget", str(curation_cap),
            "--curator", "google/gemini-3-flash-preview", "--auditor",
            "google/gemini-3.1-pro-preview" if args.fresh else "deepseek/deepseek-v3.2",
            "--audit-effort", "low" if args.fresh else "none", "--audit-max-tokens", "6144", "--pilot-size", str(len(selected)),
            "--batch-size", "10", "--concurrency", "3", "--requests-per-minute", "30"], audit)
        physics = json.loads((audit / "status.json").read_text())
        atomic_json(root / "status.json", {"state": physics["state"], "reason": physics["reason"],
            "original_parents": len(sources), "source_verified_candidates": len(candidates),
            "audited_tasks": physics["completed"], "accepted_tasks": physics["accepted"],
            "extraction_cost_usd": result["charged_cost_usd"], "curation_cost_usd": physics["charged_cost_usd"],
            "additional_cost_usd": result["charged_cost_usd"] + physics["charged_cost_usd"],
            "scope": "Repair validation only; full expansion requires a passing pilot"})
        workflow.phase = "repair_validation_finished"
        workflow.reason = "Repair pilot finished; inspect its checks before resuming competition expansion"
        workflow.report("stopped")
    except RuntimeError as exc:
        atomic_json(root / "status.json", {"state": "stopped", "reason": str(exc)})
        workflow.reason = str(exc)
        workflow.report("stopped")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=float, default=2.6)
    parser.add_argument("--fresh", action="store_true", help="Validate unused source parents with a stronger blind solver")
    parser.add_argument("--parents", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.parents <= 20:
        parser.error("Pilot parents must be 1..20")
    args.limit = 100
    args.repaired_sources = DATA / "competition_5k/repaired_pilot"
    with (args.output / ".workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
