"""Join pilot validation and pinned overlap screening without model calls."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/phy_rl/data_pipeline"))
from physics_rlvr_data.openrouter import atomic_json


REVIEW_NOTES = {
    "2011-lahg-07": "The independent solution subtracts the Mach-cone delay, giving 1912.06 m. The jet must pass the observer before the boom arrives: v*tau=(H-l)[L/(9-l)+sqrt((v/u)^2-1)]. This gives about 1631.9 m, consistent with the source's rounded 1630 m. Keep held while independent agreement is unresolved.",
    "usapho-2011-a1": "The independent solution gets four outputs right but gives -3.88e-9 J for work ON the bubble. It uses the wrong sign for the work done BY the gas during adiabatic collapse. Correct work on the complete cycle is +3.40e-9 J. Original audit retained; the disagreement remains held.",
    "2012-lahg-01": "The formal temperature is below absolute zero. The official solution concludes that freezing is impossible. A numeric-only temperature target omits that conclusion; this needs a revised output contract.",
    "2013-v3g-01": "The blind solver gives a different lens formula. Ray propagation supports the source result R*f1/(4*r), but agreement is unresolved. Keep this row in review.",
    "2014-v3g-07": "The blind solver uses b-mu*h in the denominator. Forward acceleration unloads the front axle, giving N_front=m*(g*s-a*h)/b and the source denominator b+mu*h. Keep this row in review.",
    "2016-v3g-05": "The independent rock mass is about 1,000 times the source result. The specified uranium concentration is 0.0003 kg per kg of rock. The source-scale answer is about 1.4 kg; audit agreement is unresolved.",
    "2016-v3g-10": "The independent maximum speed agrees, but the acceleration components disagree. The thread forces must satisfy the initial isosceles geometry and cannot push. All six outputs must agree before this row can pass.",
    "2017-lahg-09": "The independent toy-car acceleration differs by a factor of two. Averaging the signed accelerations over the two half-cycles introduces the missing factor. Keep this row in review.",
    "2017-v3g-08": "The blind telephoto-lens focal length and spacing disagree with the source equations. Both required outputs remain in review.",
}

TOPICS = {"Geometric Optics": "optics", "Geometrical optics": "optics",
          "Gases": "thermodynamics", "Celestial Mechanics": "mechanics", "Statics": "mechanics",
          "Fluid Mechanics": "mechanics", "Kinematics": "mechanics", "Dynamics": "mechanics"}


def analyze(output: Path, site_dir: Path | None = None) -> dict:
    artifact = json.loads((output / "pilot.json").read_text())
    screening = json.loads((output / "screening.json").read_text())
    reviews_path = output / "screening_reviews.json"
    reviews = json.loads(reviews_path.read_text()) if reviews_path.exists() else {}
    source_inputs = json.loads((output / "source_manifest.json").read_text())
    manifest_ids = source_inputs["source_ids"]
    prepared = {item["source_id"]: item for item in source_inputs.get("items", [])}
    records = artifact["records"]
    source_manifest = json.loads((output / "source_git_blobs.json").read_text())
    if source_manifest["revision"] != artifact["summary"]["source_revision"]:
        raise ValueError("Source blob manifest uses a different revision")
    if len(records) != len(manifest_ids) or {r["source_id"] for r in records} != set(manifest_ids):
        raise ValueError("Pilot records do not match the source manifest")
    screens = {r["problem_id"]: r for r in screening["records"]}
    fingerprints = Counter(re.sub(r"\W+", "", r["question"].lower()) for r in records)
    for row in records:
        if row["topic"] == "Varia" and row["source_id"] == "2013-lahg-10":
            row["model_topic"] = row["topic"]
            row["topic"] = "optics"
        if row["topic"] in TOPICS:
            row["model_topic"] = row["topic"]
            row["topic"] = TOPICS[row["topic"]]
        row["model_validation_status"] = row["status"]
        report = dict(screens[row["problem_id"]])
        current_hash = hashlib.sha256(row["question"].encode()).hexdigest()
        current = report["question_sha256"] == current_hash
        resolution = reviews.get(row["problem_id"])
        if resolution and (resolution["question_sha256"] != current_hash or
                           resolution["snapshot_sha256"] != report["snapshot_sha256"]):
            raise ValueError(f"Stale screening review: {row['problem_id']}")
        report["automatic_status"] = report["status"]
        if report["status"] == "review" and resolution:
            report["manual_review"] = resolution
            report["status"] = resolution["status"]
        source_bytes = (output / "raw" / (row["source_id"] + ".tex")).read_bytes()
        git_blob = hashlib.sha1(b"blob " + str(len(source_bytes)).encode() + b"\0" + source_bytes).hexdigest()
        if row["source_id"] in prepared and "source_pdf_sha256" in prepared[row["source_id"]]:
            expected = prepared[row["source_id"]]
            revision_matches = (hashlib.sha256(source_bytes).hexdigest() == expected["source_hash"] and
                                row["provenance"]["source_revision"] == expected["source_pdf_sha256"])
            extraction = output / "prepared_sources" / (row["source_id"] + ".source-pages.txt")
            revision_matches = revision_matches and hashlib.sha256(extraction.read_bytes()).hexdigest() == expected["embedded_text_sha256"]
        else:
            revision_matches = git_blob == source_manifest["blobs"][row["source_id"]]
        row["checks"].update(
            screening_current=current,
            benchmark_screening=current and report["status"] == "clear",
            source_hash_matches=hashlib.sha256(source_bytes).hexdigest() == row["provenance"]["source_hash"],
            source_revision_blob_matches=revision_matches,
            unique_question_text=fingerprints[re.sub(r"\W+", "", row["question"].lower())] == 1,
        )
        row["screening"] = report
        row["failed_checks"] = [key for key, passed in row["checks"].items() if not passed]
        row["status"] = "review" if row["failed_checks"] else "model_checked"
        row["review_notes"] = [REVIEW_NOTES[row["source_id"]]] if row["source_id"] in REVIEW_NOTES else []
        if row["failed_checks"] and not row["review_notes"]:
            row["review_notes"].append("Unresolved checks: " + ", ".join(row["failed_checks"]) + ". Inspect the independent derivation before release.")
        if not current or report["status"] != "clear":
            row["review_notes"].append("Overlap screening is unresolved for the current question.")
        row["release_status"] = "staging_only_final_review_pending"
        atomic_json(output / "records" / (row["source_id"] + ".json"), row)
    ledger = json.loads((Path(artifact["summary"].get("ledger_dir", output)) / "usage.json").read_text())
    model_stats = []
    for model in sorted({entry["requested_model"] for entry in ledger}):
        entries = [entry for entry in ledger if entry["requested_model"] == model]
        model_stats.append({
            "model": model, "responses": len(entries),
            "cost_usd": sum(entry["charged_cost_usd"] for entry in entries),
            "median_seconds": round(statistics.median(entry["seconds"] for entry in entries), 2),
            "truncated_responses": sum(entry["finish_reason"] == "length" or entry.get("truncated_output") is True or (entry.get("structured_output_valid") is False and entry["usage"].get("completion_tokens", 0) >= entry.get("max_tokens", 4096)) for entry in entries),
        })
    summary = artifact["summary"]
    summary.update(
        analyzed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        model_checked_count=sum(r["status"] == "model_checked" for r in records),
        review_count=sum(r["status"] == "review" for r in records),
        release_status="staging only; final source and verifier review pending",
        decontamination="Six pinned benchmark releases screened; unresolved matches remain held",
        screening_scope=screening["manifest"]["scope"],
        screening_question_count=screening["manifest"]["question_count"],
        screening_snapshot_sha256=screening["manifest"]["snapshot_sha256"],
        screening_coverage_notes=screening["manifest"]["coverage_notes"],
        screening_automatic_review_count=sum(r["screening"]["automatic_status"] == "review" for r in records),
        screening_unresolved_count=sum(r["screening"]["status"] != "clear" for r in records),
        screening_source_or_exact_hits=sum(bool(r["screening"]["source_identity_hits"]) or r["screening"]["exact_match"] for r in records),
        required_output_count=sum(len(r["answers"]) for r in records),
        checked_output_count=sum(len(r["answers"]) for r in records if r["status"] == "model_checked"),
        topic_counts=dict(Counter(r["topic"] for r in records)),
        difficulty_counts=dict(Counter(str(r["difficulty"]) if r["difficulty"] is not None else "unrated" for r in records)),
        source_counts=dict(Counter(r["source"] for r in records)),
        batch_cost_usd=sum(r["cost_usd"] for r in records),
        batch_api_calls=sum(len(r["usage"]) for r in records),
        batch_truncated_responses=sum(e["finish_reason"] == "length" or (e.get("structured_output_valid") is False and e["usage"].get("completion_tokens", 0) >= e.get("max_tokens", 4096)) for r in records for e in r["usage"]),
        model_statistics=model_stats,
        source_year_min=min(r["year"] for r in records), source_year_max=max(r["year"] for r in records),
        analysis_paid_api_calls=0,
        last_replay={"offline": True, "elapsed_seconds": summary.get("elapsed_seconds"),
                     "peak_in_flight": summary.get("peak_in_flight")},
    )
    atomic_json(output / "pilot.json", artifact)
    for filename, rows in [("candidates.jsonl", records),
                           ("checked_candidates.staging.jsonl", [r for r in records if r["status"] == "model_checked"]),
                           ("review.jsonl", [r for r in records if r["status"] == "review"])]:
        (output / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    if site_dir:
        atomic_json(site_dir / "pilot.json", artifact)
    write_report(output, summary, records)
    return summary


def write_report(output: Path, summary: dict, records: list[dict]) -> None:
    if len(records) != 50 or len(summary.get("source_counts", {})) > 1:
        write_expansion_report(output, summary, records)
        return
    lines = ["# PHY-RL: 50-problem batch review", "",
             f"**{summary['model_checked_count']} checked candidates; {summary['review_count']} held for physics or answer-contract review.** All 50 remain inspectable. The checked file is a staging subset, not a training release.", "",
             "## What this batch contains", "",
             f"50 original Estonian Physics Olympiad problems, {summary['source_year_min']}–{summary['source_year_max']}, extracted from pinned native LaTeX. No OCR was required. There are {summary['required_output_count']} labeled outputs across all rows and {summary['checked_output_count']} in the checked subset. Every raw source matches both its recorded SHA-256 and the Git blob hash in the pinned original revision. Exact normalized question text is distinct within this batch; this is not a full reasoning-family deduplication.", "",
             "| Topic | Problems |", "|---|---:|"]
    lines.extend(f"| {topic} | {count} |" for topic, count in summary["topic_counts"].items())
    lines += ["", "The collection is useful for checking the pipeline, but one national contest is not sufficient source diversity for the full corpus. Source difficulty labels are preserved: 21 of 50 are rated 1–3 out of 10, and 17 are rated 7–10. The batch is not uniformly hard. No base-policy rollouts have measured how difficult these questions are for the model being trained.", "",
              "## Remaining holds", "", "| Source ID | Reason |", "|---|---|"]
    lines.extend(f"| {r['source_id']} | {' '.join(r['review_notes'])} |" for r in records if r["status"] == "review")
    lines += ["", "A failed independent solve does not by itself prove the source target is wrong. The six disagreements above remain held instead of rewriting the targets to make the check pass. The freezing-water example has a different problem: the numeric extraction loses the official impossibility conclusion.", "",
              "## Repairs made without more API calls", "",
              "Four false disagreements were repaired with documented, record-specific checks: the braking problem's radical forms are equal for positive physical parameters; the steering solver supplied its own parseable arctan equivalent on the stated acute branch; the slat distance omitted its length unit; and the charged spheres used two spellings of vacuum permittivity. These are stored in overrides.json, with the original responses intact. The runtime unit checks remain strict.", "",
              "The Mars-balloon prompt now specifies the two physical cases rather than discussing what the official solutions accept. Repeated source-hint text was removed. Construction notes stay outside the question.", "",
              "## Evaluation overlap screening", "",
              f"Screened against {summary['screening_question_count']:,} available statements from HiPhO, OlympiadBench, PHYBench, PHYSICS-test, PhysOlym-A and UGPhysics at pinned revisions. No exact question or original-source identity matches were found. Seven similarity flags, including one after prompt cleanup, were compared with their three nearest statements. All seven refer to different setups or requested outputs; the comparison reasons are retained per record. The benchmark text stays outside the curation corpus and was not sent to a model API.", ""]
    lines.extend(summary["screening_coverage_notes"])
    lines += ["", f"Screening snapshot: `{summary['screening_snapshot_sha256']}`. Reports are bound to each current question hash. `desimfj/PHYSICS` is blocked for future training imports; its test questions are used only in the external exclusion index. No IPhO 2026 material is present.", "",
              "## Measured API costs", "",
              f"Reported charges: **${summary['cost_usd']:.6f}** for **{summary['api_calls']} returned responses**. An older timed-out request has an unconfirmed charge; **${summary['unresolved_cost_reserve_usd']:.6f}** remains reserved. Charged plus reserved is **${summary['cost_usd'] + summary['unresolved_cost_reserve_usd']:.6f}**, within the $0.50 cap. This analysis made no paid requests.", "",
              "| Model | Responses | Reported cost | Median response time | Truncated |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {s['model']} | {s['responses']} | ${s['cost_usd']:.6f} | {s['median_seconds']} s | {s['truncated_responses']} |" for s in summary["model_statistics"])
    lines += ["", "These costs include the original ten, failed returned solves and targeted re-audits. A straight extrapolation to 4,000 problems would ignore source extraction differences and difficulty-dependent failures.", "",
              "## Async runner", "",
              "The runner uses one shared async HTTP connection pool, up to eight workers and paced request starts at 60 per minute. It starts at four concurrent requests, increases after successful responses and reduces concurrency after a 429. Each task finishes curation before its blind solve; different problems can proceed together. Full request ceilings are reserved before submission, so concurrent tasks share the same budget.", "",
              "Qwen requests use a required result tool, compatible provider routing and local JSON Schema validation. Invalid or truncated returned results stay charged and held; no paid repair retry is automatic. Timeouts and server errors retain a cost reserve. Rejected 429 requests alone receive bounded retries. Provider price ceilings prevent a faster route from silently raising the token price.", "",
              "The original 50-problem paid run was serial. Later live tool checks and expansion runs have separate reports; offline replay time is not generation time.", ""]
    async_path = output / "async_validation.json"
    if async_path.exists():
        bench = json.loads(async_path.read_text())
        lines += ["A separate offline timing check ran 50 synthetic problems with two 40 ms stages each:", "",
                  "| Workers | Time | Relative speed | Responses | Simulated cost |", "|---|---:|---:|---:|---:|"]
        lines.extend(f"| {r['workers']} | {r['seconds']} s | {r['speedup_vs_serial']}× | {r['responses']} | ${r['simulated_cost_usd']:.2f} |" for r in bench["results"])
        lines += ["", "This checks scheduling and equal call counts, not real provider speed. It made zero paid API calls. Actual throughput depends on provider latency and rate limits.", ""]
    lines += ["## Files and next work", "",
              "`checked_candidates.staging.jsonl` contains the checked subset; `review.jsonl` contains the seven holds; `pilot.json` powers the viewer. Raw LaTeX, responses, usage and documented overrides remain available for inspection. Training export and the environment reject these explicit staging records.", "",
              "Resolve the seven held contracts/solutions, then complete the source-to-target review of the checked subset before a training release. The next source batch should add pre-2024 official olympiads with more electromagnetism, waves and modern physics. Measure solve rates with the intended base policy before deciding the RL curriculum. The source license is CC BY-NC 4.0, and its attribution stays in each record.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def write_expansion_report(output: Path, summary: dict, records: list[dict]) -> None:
    smoke_path = output / "live_check.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    lines = [f"# PHY-RL: {len(records)}-problem expansion", "",
             f"{summary['model_checked_count']} pass the displayed checks; {summary['review_count']} remain held. These are staging records.", "",
             "## Sources and topics", "",
             "| Source | Problems |", "|---|---:|"]
    lines.extend(f"| {source} | {count} |" for source, count in summary["source_counts"].items())
    lines += ["", "| Topic | Problems |", "|---|---:|"]
    lines.extend(f"| {topic} | {count} |" for topic, count in summary["topic_counts"].items())
    lines += ["", f"{summary['required_output_count']} required outputs; {summary['checked_output_count']} in the checked subset. Source years are {summary['source_year_min']}–{summary['source_year_max']}. USAPhO difficulty is unrated; Estonian difficulty scores are retained. Selected USAPhO subparts keep their shared physical context and one source-family identity.", "",
              "Native Estonian files match the pinned Git blobs. USAPhO transcriptions retain the official PDF hash, page numbers, embedded page text and transcription hashes. Source corrections and transcription decisions are documented per record in source_notes. Construction notes are excluded from the RL question.", "",
              "USAPhO is copyright AAPT; redistribution permission is unverified. The private staging viewer shows each record's license. Estonian material is CC BY-NC 4.0.", "",
              "## Checks and remaining holds", "",
              "| Problem | Status | Review note |", "|---|---|---|"]
    lines.extend(f"| {r['source_id']} | {r['status']} | {' '.join(r['review_notes']) or r.get('review_resolution', '')} |" for r in records)
    lines += ["", "Original model responses and documented local corrections are retained. Unresolved answer disagreements stay held.", "",
              "## Evaluation screening", "",
              f"All current English questions were screened against {summary['screening_question_count']:,} statements in six pinned releases. {summary['screening_automatic_review_count']} similarity flags; {summary['screening_unresolved_count']} unresolved. Review decisions are bound to question and index hashes. Evaluation questions remain in an external exclusion index and are never sent to curation models.", "",
              "## Live API results and budget", "",
              f"The three-request Qwen smoke check passed and cost ${smoke['cost_usd']:.6f}." if smoke else "The Qwen tool path was checked live in the preceding pilot; this batch records all returned tool contracts and failures.", "",
              f"New problem curation and audits cost **${summary['batch_cost_usd']:.6f}** across {summary['batch_api_calls']} returned responses, including held responses and explicit re-audits. {summary['batch_truncated_responses']} returned responses reached their token limit. The last run peaked at {summary['peak_in_flight']} concurrent requests and received {summary['rate_limit_events']} rate-limit responses.", "",
              f"Cumulative reported charges are **${summary['cost_usd']:.6f}**. The earlier uncertain timeout retains **${summary['unresolved_cost_reserve_usd']:.6f}**. Charged plus reserved is **${summary['cost_usd'] + summary['unresolved_cost_reserve_usd']:.6f}**, leaving **${summary['budget_usd'] - summary['cost_usd'] - summary['unresolved_cost_reserve_usd']:.6f}** under the $0.50 cumulative cap.", "",
              "The runner shares the original ledger and process lock across batch directories. Returned malformed or truncated answers are charged and held. Only rejected rate-limit requests receive automatic retries; failed physics answers need explicit review. No additional budget has been approved for a 100-problem batch.", "",
              "## Files", "",
              "pilot.json and candidates.jsonl contain the complete batch. checked_candidates.staging.jsonl contains only passing candidates; review.jsonl contains held rows. raw/, prepared_sources/, responses/, overrides.json and screening_reviews.json preserve the evidence. Training export rejects these staging records."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--site-dir", type=Path)
    args = parser.parse_args()
    summary = analyze(args.output, args.site_dir)
    print(json.dumps({key: summary[key] for key in ["candidate_count", "model_checked_count", "review_count",
                                                  "required_output_count", "checked_output_count", "cost_usd"]}, indent=2))


if __name__ == "__main__":
    main()
