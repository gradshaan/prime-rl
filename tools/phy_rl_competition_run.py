"""Run bounded competition extraction and physics audits in resumable batches."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path

from phy_rl_curate import DATA, PROJECT, REPO, read_rows, write_rows
from phy_rl_gather_training import family_key
from physics_rlvr_data.openrouter import atomic_json

SOURCES = {"ipho_olimpicos", "apho_archive", "nbpho_olimpicos", "wopho_archive"}


def admitted_union(paths: list[Path], review_holds: dict | None = None) -> tuple[list[dict], list[dict]]:
    accepted, duplicates, identities, families, covered = [], [], set(), set(), set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_rows(path):
            if row.get("training_ready") is not True or not row["checks"] or not all(row["checks"].values()):
                raise ValueError(f"Non-admitted record in {path}")
            if row["source"] not in SOURCES or type(row["year"]) is not int or row["year"] > 2023:
                raise ValueError("Competition release contains an out-of-scope source")
            identity = row["problem_id"]
            parent = row["parent_problem_id"]
            hold = (review_holds or {}).get(identity)
            if hold:
                if hold["question_sha256"] != hashlib.sha256(row["question"].encode()).hexdigest():
                    raise ValueError("Release review no longer matches the question")
                duplicates.append({"problem_id": identity, "parent_problem_id": parent,
                                   "reason": hold["reason"], "artifact": str(path)})
                continue
            labels = row["source_evidence"]["provenance"]["subpart_labels"]
            parts = {(parent, label) for label in labels}
            family = family_key(row["question"])
            if identity in identities or family in families or parts & covered:
                duplicates.append({"problem_id": identity, "parent_problem_id": parent,
                                   "reason": "duplicate_identity_family_or_source_subpart", "artifact": str(path)})
                continue
            accepted.append(row)
            identities.add(identity)
            families.add(family)
            covered.update(parts)
    return accepted, duplicates


def select_sources(rows: list[dict], excluded: set[str], limit: int) -> list[dict]:
    groups = collections.defaultdict(collections.deque)
    eligible = [row for row in rows if row["source"] in SOURCES and row["problem_id"] not in excluded
                and row["year"] <= 2023 and sum(meta["page_count"] for meta in row["documents"].values()) <= 12]
    eligible.sort(key=lambda row: (sum(meta["page_count"] for meta in row["documents"].values()), -row["year"], row["problem_id"]))
    for row in eligible:
        groups[row["source"]].append(row)
    selected, counts = [], collections.Counter()
    while len(selected) < limit and any(groups.values()):
        source = min((key for key, values in groups.items() if values),
                     key=lambda key: (counts[key], key != "ipho_olimpicos", key))
        selected.append(groups[source].popleft())
        counts[source] += 1
    return selected


class CompetitionRun:
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.phase = "initializing"
        self.active = None
        self.reason = ""
        self.stop_requested = False

    def request_stop(self, *_):
        self.stop_requested = True
        (self.output / "STOP").touch()
        if self.active:
            (self.active / "STOP").touch()

    def accounting(self) -> tuple[float, float]:
        spent, reserved = 0.0, 0.0
        for path in self.output.rglob("usage.jsonl"):
            spent += sum(row["charged_cost_usd"] for row in read_rows(path))
        for path in self.output.rglob("unresolved_requests.json"):
            reserved += sum(row["reserved_cost_usd"] for row in json.loads(path.read_text()))
        return spent, reserved

    def report(self, state: str) -> dict:
        inputs = sorted((self.output / "batches").glob("*/audit*/accepted.jsonl"))
        recovered = self.output / "repaired_audit/accepted.jsonl"
        reviews = self.output / "release_review_holds.json"
        holds = json.loads(reviews.read_text()) if reviews.exists() else {}
        recovered_inputs = sorted((self.output / "offline_recheck").glob("*/accepted.jsonl"))
        replaced = {json.loads(path.read_text())["original_run"] for path in
                    (self.output / "offline_recheck").glob("*/status.json")}
        originals = [path for path in [recovered, *inputs] if str(path.parent) not in replaced]
        validation = self.output / "repair_validation/audit/accepted.jsonl"
        strengthened = [self.output / "repair_validation/scope_fixed_audit/accepted.jsonl",
                        self.output / "repair_validation/resolution_audit/accepted.jsonl",
                        self.output / "contract_review/accepted.jsonl",
                        self.output / "forward_pilot/audit/accepted.jsonl"]
        accepted, duplicates = admitted_union([*strengthened, validation, *recovered_inputs, *originals], holds)
        write_rows(self.output / "accepted.jsonl", accepted)
        write_rows(self.output / "merge_review.jsonl", duplicates)
        extraction_roots = sorted((self.output / "batches").glob("*/extraction"))
        extraction_roots.append(self.args.repaired_sources)
        extraction_roots.append(self.output / "forward_pilot/extraction")
        transcripts = [json.loads(path.read_text()) for root in extraction_roots
                       for path in (root / "transcriptions").glob("*.json")]
        plan_path = self.output / "source_plan.jsonl"
        planned = {row["problem_id"] for row in read_rows(plan_path)} if plan_path.exists() else set()
        processed = planned & {row["parent_problem_id"] for row in transcripts}
        audit_roots = [self.output / "repaired_audit", *sorted((self.output / "batches").glob("*/audit*"))]
        audit_roots.append(self.output / "repair_validation/audit")
        audit_roots.extend([self.output / "repair_validation/resolution_audit",
                            self.output / "repair_validation/scope_fixed_audit",
                            self.output / "forward_pilot/audit"])
        audits = [json.loads(path.read_text()) for root in audit_roots for path in (root / "records").glob("*.json")
                  if not path.name.endswith(".checks.json")]
        completed = [row for row in audits if row.get("curation_completed")]
        finalized = [row for row in completed if row.get("batch_finalized")]
        spent, reserved = self.accounting()
        status = {"updated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "state": state,
                  "phase": self.phase, "active_directory": str(self.active) if self.active else None,
                  "reason": self.reason, "planned_new_originals": self.args.limit,
                  "new_original_parents_processed": len(processed),
                  "new_original_parents_remaining": len(planned - processed),
                  "original_parents_transcribed": len({r["parent_problem_id"] for r in transcripts}),
                  "candidate_tasks": sum(len(r["tasks"]) for r in transcripts),
                  "physics_checks_completed": len(completed), "finalized_rows": len(finalized),
                  "accepted_tasks": len(accepted),
                  "accepted_original_parents": len({r["parent_problem_id"] for r in accepted}),
                  "accepted_by_source": dict(collections.Counter(r["source"] for r in accepted)),
                  "accepted_by_topic": dict(collections.Counter(r["topic"] for r in accepted)),
                  "charged_cost_usd": round(spent, 8), "unconfirmed_reserve_usd": round(reserved, 8),
                  "workflow_hard_cap_usd": self.args.budget,
                  "prior_source_repair_cost_is_separate": True,
                  "raw_pdfs_retained": False, "release_file": str(self.output / "accepted.jsonl")}
        atomic_json(self.output / "status.json", status)
        atomic_json(self.output / "viewer.json", {"summary": status, "records": accepted})
        return status

    def remaining(self) -> float:
        spent, reserved = self.accounting()
        if reserved and not self.args.allow_reserved_unconfirmed:
            raise RuntimeError("Unconfirmed API charge: stop before further model calls")
        return round(self.args.budget - spent - reserved, 8)

    def command(self, script: str, arguments: list[str], directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.active = directory
        with (directory / "run.log").open("a") as log:
            process = subprocess.Popen(["uv", "--no-config", "run", "--project", str(PROJECT),
                                        str(REPO / "tools" / script), *arguments], cwd=REPO, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                self.report("running")
                time.sleep(5)
        self.report("running")
        if self.stop_requested:
            raise RuntimeError("Shutdown requested; child work drained and checkpointed")
        if process.returncode:
            raise RuntimeError(f"{script} exited {process.returncode}; inspect {directory / 'run.log'}")

    def wait_for_lock(self, directory: Path, name: str) -> None:
        if not directory.exists():
            return
        with (directory / name).open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.active = directory
                    self.report("running")
                    time.sleep(5)
            fcntl.flock(lock, fcntl.LOCK_UN)

    def audit(self, queue: Path, directory: Path) -> None:
        rows = read_rows(queue)
        if not rows:
            return
        self.wait_for_lock(directory, ".run.lock")
        status = directory / "status.json"
        finished = False
        if status.exists():
            result = json.loads(status.read_text())
            if result["state"] == "complete":
                finished = True
            if result["state"] == "stopped":
                raise RuntimeError("Physics audit stopped: " + result["reason"])
        if not (directory / "input.jsonl").exists():
            self.command("phy_rl_curate.py", ["prepare", str(directory), "--count", str(len(rows)),
                         "--pool", "competition", "--candidate-file", str(queue), "--only-candidate-files",
                         "--minimum-olympiad-fraction", "1", "--minimum-ipho-count", "0"], directory)
        prior_inputs = {row["problem_id"]: row for row in read_rows(directory / "input.jsonl")}
        for row in rows:
            if row["problem_id"] in prior_inputs and row["question_sha256"] != prior_inputs[row["problem_id"]]["question_sha256"]:
                raise ValueError("Source question changed after physics audit")
        extra = [row for row in rows if row["problem_id"] not in prior_inputs]
        if finished:
            if extra:
                additional = directory.with_name(directory.name + "_additional")
                additional.mkdir(exist_ok=True)
                queue = additional / "source_input.jsonl"
                write_rows(queue, extra)
                self.audit(queue, additional)
            return
        existing = sum(row["charged_cost_usd"] for row in read_rows(directory / "usage.jsonl")) if (directory / "usage.jsonl").exists() else 0
        budget = existing + min(self.remaining(), max(0.15, len(rows) * 0.035))
        if budget <= existing:
            raise RuntimeError("Workflow spending cap reached")
        prior = json.loads((directory / "run_config.json").read_text()) if (directory / "run_config.json").exists() else {}
        arguments = ["run", str(directory), "--budget", str(budget),
                     "--curator", "google/gemini-3-flash-preview", "--auditor", prior.get("auditor", self.args.auditor),
                     "--audit-effort", prior.get("audit_effort", "low"),
                     "--audit-max-tokens", str(prior.get("audit_max_tokens", 8192)), "--pilot-size", "10",
                     "--batch-size", "30", "--concurrency", str(self.args.concurrency), "--requests-per-minute", "40"]
        arguments.extend(["--resolver", prior.get("resolver", "google/gemini-3.1-pro-preview")])
        if self.args.allow_reserved_unconfirmed:
            arguments.append("--allow-reserved-unconfirmed")
        self.command("phy_rl_curate.py", arguments, directory)
        result = json.loads((directory / "status.json").read_text())
        if result["state"] != "complete":
            raise RuntimeError("Physics audit stopped: " + result["reason"])
        if extra:
            additional = directory.with_name(directory.name + "_additional")
            additional.mkdir(exist_ok=True)
            queue = additional / "source_input.jsonl"
            write_rows(queue, extra)
            self.audit(queue, additional)

    def run(self) -> None:
        for shutdown_signal in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(shutdown_signal, self.request_stop)
        inventory = read_rows(DATA / "competition_5k/source_queue.jsonl")
        old = read_rows(DATA / "competition_5k/repair_input.jsonl")
        plan = select_sources(inventory, {r["problem_id"] for r in old}, self.args.limit)
        if len(plan) != self.args.limit:
            raise ValueError(f"Only {len(plan)} eligible new source parents available")
        path = self.output / "source_plan.jsonl"
        serialized = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in plan)
        if path.exists() and path.read_text() != serialized:
            raise ValueError("Source plan changed; refusing to resume")
        path.write_text(serialized)
        atomic_json(self.output / "config.json", {"new_originals": len(plan), "budget_usd": self.args.budget,
                    "source_plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "by_source": dict(collections.Counter(r["source"] for r in plan)),
                    "concurrency": self.args.concurrency, "source_batch_size": self.args.batch_size,
                    "new_batch_auditor": self.args.auditor})
        try:
            self.phase = "waiting_for_source_repair"
            deadline = time.monotonic() + 1800
            while not (self.args.repaired_sources / "status.json").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Source repair did not finish within 30 minutes")
                self.report("running")
                time.sleep(5)
            repair = json.loads((self.args.repaired_sources / "status.json").read_text())
            if repair["unconfirmed_reserve_usd"]:
                raise RuntimeError("Source repair has an unconfirmed charge")
            self.phase = "auditing_repaired_pilot"
            self.audit(self.args.repaired_sources / "curation_queue.jsonl", self.output / "repaired_audit")
            for index in range(0, len(plan), self.args.batch_size):
                if (self.output / "STOP").exists():
                    raise RuntimeError("STOP file requested a stop at the batch boundary")
                batch = self.output / "batches" / f"{index // self.args.batch_size:04d}"
                batch.mkdir(parents=True, exist_ok=True)
                queue = batch / "source_input.jsonl"
                write_rows(queue, plan[index:index + self.args.batch_size])
                extraction = batch / "extraction"
                self.phase = "extracting_new_competition_sources"
                self.wait_for_lock(extraction, "run.lock")
                previous = json.loads((extraction / "status.json").read_text()) if (extraction / "status.json").exists() else None
                needs_resume = previous is None or (previous["state"] == "stopped"
                                                    and previous["original_problems_completed"] < len(read_rows(queue)))
                if needs_resume:
                    allowance = min(self.remaining(), 1.25)
                    if allowance < 0.08:
                        raise RuntimeError("Workflow spending cap reached")
                    budget = (previous["charged_cost_usd"] if previous else 0) + allowance
                    self.command("phy_rl_extract_archives.py", [str(extraction), "--input-queue", str(queue),
                                 "--limit", str(self.args.batch_size), "--budget", str(budget), "--pilot-year-min", "1967",
                                 "--concurrency", str(self.args.concurrency), "--requests-per-minute", "40"], extraction)
                result = json.loads((extraction / "status.json").read_text())
                if result["unconfirmed_reserve_usd"]:
                    raise RuntimeError("Source extraction has an unconfirmed charge")
                if result["candidate_parent_count"] / max(1, result["original_problems_completed"]) < 0.25:
                    raise RuntimeError("Fewer than 25% of source parents yielded screened candidates; inspect source holds")
                self.phase = "auditing_new_competition_tasks"
                self.audit(extraction / "curation_queue.jsonl", batch / "audit")
            status = self.report("running")
            self.phase = "finished"
            if status["new_original_parents_remaining"]:
                self.reason = f"{status['new_original_parents_remaining']} planned parents remain; resume source extraction before calling the batch complete"
                self.report("stopped")
            else:
                self.report("complete")
        except RuntimeError as exc:
            self.reason = str(exc)
            self.report("stopped")
            print(self.reason, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--budget", type=float, default=2.6)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--auditor", default="google/gemini-2.5-flash")
    parser.add_argument("--allow-reserved-unconfirmed", action="store_true",
                        help="Keep unknown charges reserved in full; never repeat a matching request")
    parser.add_argument("--repaired-sources", type=Path, default=DATA / "competition_5k/repaired_pilot")
    args = parser.parse_args()
    if args.limit < 1 or args.budget <= 0 or args.batch_size < 1 or not 1 <= args.concurrency <= 8:
        parser.error("Use positive limits and concurrency 1..8")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        CompetitionRun(args).run()


if __name__ == "__main__":
    main()
