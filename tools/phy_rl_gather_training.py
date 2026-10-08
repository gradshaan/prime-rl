"""Stage existing physics training releases without changing their reference answers."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pyarrow.parquet as pq

TEXTBOOK = "MegaScience/TextbookReasoning"
NEMOTRON = "nvidia/Nemotron-RL-Science-v1"
PINNED = {
    TEXTBOOK: ("ca7ecbec76d01bff2e99f3dc17735b02f87d4e96", "data/train-00000-of-00001.parquet"),
    NEMOTRON: ("cc485d0ec5b53d430a43c756db0a4d6f00dc4cee", "so_openq.jsonl"),
}
CHECKSUMS = {
    TEXTBOOK: "f27c71763b21db22cb58c75a8495decb2fe5e2a83efc16f4b6c004fda606e543",
    NEMOTRON: "22e5240d0602d1d1207f5b9c1e56e5826caae5a81d373b71581ea44521476b45",
}
CONTEXT = re.compile(
    r"\b(?:figure|fig\.|diagram|pictured|illustrated|shown|previous problem|preceding problem|"
    r"above problem|the graph|the chart|accompanying|attached|given table|table\s+\d|"
    r"problem\s+\d+|exercise\s+\d+|equation\s+\(?\d+\.\d+|see\s+section)\b|"
    r"<image|!\[.*?\]\(|https?://", re.I,
)
UNCERTAIN = re.compile(
    r"\b(?:not enough information|insufficient (?:information|data)|not specified|"
    r"seems (?:incorrect|unlikely|unrealistic)|misinterpretation|likely incorrect|"
    r"reference (?:document|answer) is incorrect|assum(?:e|ing).{0,100}not given|"
    r"unusually high|graph.reading|consult.{0,30}tables)\b", re.I,
)
TASK = re.compile(r"\b(?:calculate|determine|find|derive|obtain|evaluate|compute|solve)\b", re.I)
MATH = re.compile(r"\\(?:frac|dfrac|sqrt|int|sum|partial|hbar|omega|lambda|epsilon|gamma|psi)\b|[=^]")
BENCHMARK = re.compile(r"\b(?:OlympiadBench|PHYBench|UGPhysics|PhysReason|HiPhO|PhysOlym|MMMU|CoPBench|IPhO\s*2026)\b", re.I)
FUTURE_CONTEST = re.compile(r"\b(?:IPhO|APhO|EuPhO|NBPhO|USAPhO|INPhO|WoPhO)\b.{0,30}\b20(?:2[4-9]|[3-9]\d)\b|"
                            r"\b20(?:2[4-9]|[3-9]\d)\b.{0,30}\b(?:IPhO|APhO|EuPhO|NBPhO|USAPhO|INPhO|WoPhO)\b", re.I)
ENGINEERING = re.compile(r"\b(?:heat exchanger|refrigerant|air.conditioning|HVAC|brine|permeability of|"
                         r"psychrometric|Btu|diesel engine|car.plane|wing area|Reynolds number|"
                         r"reinforced concrete|distillation|pump efficiency|notched member|"
                         r"nominal stress|pipe network|five.pipe)\b", re.I)
OFF_TOPIC = re.compile(r"\b(?:financial data|stock prices|Apple stock|neural network|write a program|"
                       r"write a function|provide references|find references|Ward identities|"
                       r"supersymmetry|Grassmann|BRST|bosonisation|bosonization|Yang.Mills)\b", re.I)
TOPICS = [
    ("modern_physics", r"\b(?:quantum|wavefunction|wave function|Schr|Hamiltonian|eigenstate|eigenvalue|"
                       r"fermion|boson|electron|photon|nuclear|neutron|radioactive|relativistic|"
                       r"relativity|Lorentz|Planck|positron|hydrogen atom|Pauli|spin|density operator|"
                       r"state vector|Lifshitz|bosons|fermions)\b|\\hbar"),
    ("electromagnetism", r"\b(?:electric|magnetic|charge|capacit|induct|current|circuit|resistor|"
                       r"voltage|dipole|Maxwell|dielectric|solenoid|coil|conductor)"),
    ("thermodynamics", r"\b(?:entropy|temperature|thermal|thermodynamic|heat|Carnot|ideal gas|"
                       r"Boltzmann|partition function|isothermal|adiabatic|specific heat)\b"),
    ("waves_optics", r"\b(?:wave|optical|optics|lens|interference|diffraction|refraction|"
                     r"oscillation|oscillator|frequency|wavelength|sound|light ray|mirror)\b"),
    ("fluids", r"\b(?:fluid|liquid|buoyancy|buoyant|viscosity|surface tension|Bernoulli|hydrostatic)\b"),
    ("mechanics", r"\b(?:mass|force|velocity|acceleration|momentum|gravity|gravitational|orbit|"
                   r"friction|pulley|spring|torque|inertia|pendulum|collision|rotation|rotating)\b"),
]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text: str) -> str:
    text = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", text)
    return re.sub(r"\s+", " ", re.sub(r"[^\w]", " ", text.casefold())).strip()


def family_key(question: str) -> str:
    return digest(re.sub(r"\b\d+(?:[.,]\d+)*\b", "NUMBER", normalize(question)))


def assess(question: str, answer: str, solution: str) -> tuple[list[str], int]:
    reasons = []
    if not 180 <= len(question) <= 6500:
        reasons.append("statement_length")
    if not answer.strip() or len(answer) > 3500 or answer.strip().upper() in {"XX", "XXX", "N/A", "TBD", "UNKNOWN"}:
        reasons.append("missing_or_long_reference_answer")
    if not TASK.search(question):
        reasons.append("no_explicit_calculation_or_derivation")
    if not MATH.search(question + answer):
        reasons.append("no_numeric_or_symbolic_structure")
    if CONTEXT.search(question):
        reasons.append("external_context_or_image")
    if UNCERTAIN.search(solution):
        reasons.append("solution_has_explicit_uncertainty_or_missing_data")
    if BENCHMARK.search(question) or FUTURE_CONTEST.search(question):
        reasons.append("named_benchmark_or_forbidden_year")
    if ENGINEERING.search(question):
        reasons.append("engineering_application_outside_priority_distribution")
    if OFF_TOPIC.search(question) or infer_topic(question) == "other_physics":
        reasons.append("outside_core_physics_distribution")
    if re.search(r"\b(?:not provided|not specified|not given|not available|can be calculated|refined answer provides)\b", answer, re.I):
        reasons.append("reference_answer_incomplete")
    if re.search(r"\b(?:prove|show|demonstrate)\b", question, re.I) and not re.search(
        r"\b(?:calculate|determine|find|evaluate|compute|obtain)\b", question, re.I
    ):
        reasons.append("proof_only_request")
    if target_is_given(question, answer):
        reasons.append("target_expression_already_in_statement")
    if "\ufffd" in question + answer + solution:
        reasons.append("replacement_character")
    if re.search(r"(?:^|\n)\s*\(?[A-D]\)?[.:)]\s+", question) and "correct" in question.lower():
        reasons.append("multiple_choice")
    if solution and not 650 <= len(solution) <= 22000:
        reasons.append("solution_length")
    if not solution and len(answer) < 70:
        reasons.append("short_answer_without_worked_solution")
    score = min(12, len(MATH.findall(question))) + min(8, len(question) // 250)
    score += 6 * bool(re.search(r"\b(?:derive|obtain|as a function|in terms of|limiting|maximum|minimum)\b", question, re.I))
    score += 3 * bool(re.search(r"(?:\([abc]\)|\n[123]\.)", question))
    score += 4 * bool(solution) + min(4, len(answer) // 150)
    return reasons, score


def target_is_given(question: str, answer: str) -> bool:
    spans = re.findall(r"\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$|(?<!\$)\$(?!\$)(.*?)(?<!\$)\$", answer, re.S)
    expressions = [next(value for value in span if value) for span in spans if any(span)]
    expressions = [value for value in expressions if len(value) >= 30 and "=" in value]
    compact_question = re.sub(r"\s+", "", question)
    return bool(expressions) and all(re.sub(r"\s+", "", value) in compact_question for value in expressions)


def infer_topic(question: str) -> str:
    for topic, pattern in TOPICS:
        if re.search(pattern, question, re.I):
            return topic
    return "other_physics"


def stage_record(repo: str, index: int, question: str, answer: str, solution: str, metadata: dict) -> dict:
    revision, filename = PINNED[repo]
    source = "textbookreasoning_physics" if repo == TEXTBOOK else "nemotron_rl_science_physics"
    reasons, score = assess(question, answer, solution)
    missing = ["independent_physics_audit", "complete_per_output_contract", "deterministic_verifier_checks",
               "underlying_contest_source_and_year_if_any"]
    if repo == TEXTBOOK:
        missing.append("original_book_and_page_provenance")
    else:
        missing.append("full_original_answer_and_post_date_review")
    return {
        "problem_id": source + "-" + (metadata.get("uuid") or str(index)),
        "source": source, "source_id": str(index), "source_split": "train" if repo == TEXTBOOK else "so_openq",
        "question": question, "reference_answer": answer, "reference_solution": solution,
        "question_sha256": digest(question), "answer_sha256": digest(answer), "solution_sha256": digest(solution),
        "problem_family_id": family_key(question), "topic": infer_topic(question), "topic_method": "keyword_inference",
        "selection_score": score, "difficulty": None, "difficulty_status": "not_measured",
        "status": "source_candidate", "release_status": "held", "training_ready": False,
        "required_outputs": [], "pending_checks": missing, "preflight_rejections": reasons,
        "provenance": {"repository": repo, "revision": revision, "file": filename, "row_index": index,
                       "dataset_url": f"https://huggingface.co/datasets/{repo}/tree/{revision}",
                       "original_source_url": metadata.get("QuestionLink"),
                       "original_book": None, "original_page": None,
                       "license": "CC-BY-NC-SA-4.0" if repo == TEXTBOOK else "CC-BY-SA-4.0",
                       "reference_solution_origin": "publisher_model_refinement" if solution else "not_provided",
                       "metadata": metadata},
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def fetch_sources(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)

    def fetch(repo: str) -> None:
        revision, filename = PINNED[repo]
        path = raw / (repo.replace("/", "--") + (".parquet" if repo == TEXTBOOK else ".jsonl"))
        if path.exists():
            with path.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() == CHECKSUMS[repo]:
                    return
            raise ValueError(f"Existing source cache has the wrong checksum: {path}")
        partial = path.with_suffix(path.suffix + ".part")
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
        for attempt in range(3):
            try:
                with httpx.Client(timeout=180, follow_redirects=True) as client, client.stream("GET", url) as response:
                    response.raise_for_status()
                    checksum = hashlib.sha256()
                    with partial.open("wb") as file:
                        for chunk in response.iter_bytes():
                            checksum.update(chunk)
                            file.write(chunk)
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        if checksum.hexdigest() != CHECKSUMS[repo]:
            raise ValueError(f"Pinned source download checksum differs: {repo}")
        partial.replace(path)
        print(f"Downloaded {repo}: {path.stat().st_size} bytes", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, PINNED))


def balanced(records: list[dict], limit: int) -> list[dict]:
    groups = collections.defaultdict(list)
    for row in records:
        groups[(row["source"], row["topic"])].append(row)
    for group in groups.values():
        group.sort(key=lambda r: (-r["selection_score"], r["problem_id"]))
    result = []
    keys = sorted(groups)
    positions = {key: 0 for key in keys}
    while any(positions[key] < len(groups[key]) for key in keys):
        for key in keys:
            weight = 4 if key[0] == "textbookreasoning_physics" else 1
            for _ in range(weight):
                if positions[key] == len(groups[key]):
                    break
                result.append(groups[key][positions[key]])
                positions[key] += 1
                if len(result) == limit:
                    return result
    return result


def prepare(raw: Path, output: Path, pool_size: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    candidates, counts, rejected = [], collections.Counter(), collections.Counter()
    physics_rows, identities, families = [], set(), set()
    downloads = []
    for repo in PINNED:
        path = raw / (repo.replace("/", "--") + (".parquet" if repo == TEXTBOOK else ".jsonl"))
        with path.open("rb") as file:
            checksum = hashlib.file_digest(file, "sha256").hexdigest()
        if checksum != CHECKSUMS[repo]:
            raise ValueError(f"Source does not match the pinned release: {repo}")
        downloads.append({"repository": repo, "revision": PINNED[repo][0], "file": PINNED[repo][1],
                          "sha256": checksum, "download_bytes": path.stat().st_size})
        if repo == TEXTBOOK:
            def iter_rows():
                for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
                    yield from batch.to_pylist()
            rows = iter_rows()
        else:
            rows = (json.loads(line) for line in path.open() if line.strip())
        for index, original in enumerate(rows):
            counts[repo + ":all"] += 1
            physics = original.get("subject") == "physics" if repo == TEXTBOOK else original["metadata"]["topic"] == "Physics"
            if not physics:
                continue
            counts[repo + ":physics"] += 1
            if repo == TEXTBOOK:
                physics_rows.append({**original, "source_row_index": index})
                row = stage_record(repo, index, original["question"], original["reference_answer"], original["answer"], {})
            else:
                row = stage_record(repo, index, original["problem"], original["expected_answer"], "",
                                   {**original["metadata"], "uuid": original["uuid"],
                                    "verifier_type": original.get("verifier_type"), "agent_ref": original["agent_ref"]})
            if row["preflight_rejections"]:
                rejected.update(row["preflight_rejections"])
                continue
            identity = row["provenance"]["original_source_url"] or normalize(row["question"])
            if identity in identities or row["problem_family_id"] in families:
                rejected["duplicate_source_post_or_numeric_family"] += 1
                continue
            identities.add(identity)
            families.add(row["problem_family_id"])
            candidates.append(row)
        print(f"Read {repo}: {counts[repo + ':physics']} physics rows", flush=True)
    # Preserve just the physics partition with stable original row indices; raw scientific data is disposable.
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(physics_rows), output / "textbook_physics.parquet", compression="zstd")
    write_jsonl(output / "eligible_pool.jsonl.gz", candidates)
    selected = balanced(candidates, pool_size)
    (output / "preflight.json").write_text(json.dumps({"records": selected}, ensure_ascii=False))
    summary = {"downloads": downloads, "counts": dict(counts), "rejected_reasons": dict(rejected),
               "preflight_eligible": len(candidates), "screening_pool": len(selected), "api_cost_usd": 0,
               "released_training_problems": 0, "selection_method": "calculation/derivation, context filters, numeric-family dedup, source/topic round robin",
               "difficulty_note": "Selection scores are heuristic, not measured olympiad difficulty."}
    (output / "import_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def finalize(output: Path, screening: Path, limit: int, existing: Path | None = None) -> None:
    candidates = json.loads((output / "preflight.json").read_text())["records"]
    result = json.loads(screening.read_text())
    reports = {r["problem_id"]: r for r in result["records"]}
    if len(reports) != len(result["records"]) or len(candidates) != len(reports) or set(reports) != {r["problem_id"] for r in candidates}:
        raise ValueError("Screening must cover every candidate exactly once")
    clear, held, blocked, duplicates, quality_exclusions = [], [], [], [], []
    existing_families = set()
    if existing is not None:
        for row in pq.read_table(existing, columns=["question"]).to_pylist():
            existing_families.add(family_key(row["question"]))
    data = output.parent
    for path in sorted(data.glob("pilot_*/pilot.json")):
        for row in json.loads(path.read_text())["records"]:
            existing_families.add(family_key(row["question"]))
    for row in candidates:
        report = reports[row["problem_id"]]
        if report["question_sha256"] != row["question_sha256"]:
            raise ValueError("Screened question has changed")
        row["benchmark_screening"] = report
        row["pending_checks"] = list(dict.fromkeys([*row["pending_checks"], "underlying_contest_source_and_year_if_any"]))
        if row["problem_family_id"] in existing_families:
            duplicates.append({"problem_id": row["problem_id"], "question_sha256": row["question_sha256"],
                               "reason": "already_in_current_corpus_or_pilots"})
        elif report["status"] == "clear":
            quality_reasons = []
            if row["selection_score"] < 8:
                quality_reasons.append("below_selection_score_floor")
            if not MATH.search(row["reference_answer"]) and not re.search(r"\d", row["reference_answer"]):
                quality_reasons.append("no_closed_form_reference_evidence")
            if quality_reasons:
                quality_exclusions.append({"problem_id": row["problem_id"], "question_sha256": row["question_sha256"],
                                           "reasons": quality_reasons})
            else:
                clear.append(row)
        elif report["status"] == "review":
            row["pending_checks"].append("benchmark_similarity_review")
            held.append(row)
        else:
            # Keep only identifiers and hashes for excluded evaluation problems.
            blocked.append({k: report[k] for k in ["problem_id", "status", "question_sha256", "source_identity_hits", "exact_match"]})
    selected = balanced(clear, limit)
    write_jsonl(output / "candidates_5000.jsonl", selected)
    write_jsonl(output / "overlap_review.jsonl.gz", held)
    write_jsonl(output / "blocked.jsonl", blocked)
    write_jsonl(output / "existing_duplicates.jsonl", duplicates)
    write_jsonl(output / "quality_exclusions.jsonl", quality_exclusions)
    selected_ids = {r["problem_id"] for r in selected}
    write_jsonl(output / "reserve.jsonl.gz", [r for r in clear if r["problem_id"] not in selected_ids])
    # The working artifact contains held and blocked rows. Do not retain it as a corpus.
    (output / "preflight.json").unlink()
    summary = json.loads((output / "import_summary.json").read_text())
    summary.update({"selected_candidates": len(selected), "benchmark_clear": len(clear),
                    "screening_pool": len(candidates),
                    "screening_status_counts": dict(collections.Counter(r["status"] for r in reports.values())),
                    "overlap_review": len(held), "exact_blocked": len(blocked),
                    "existing_duplicates": len(duplicates),
                    "quality_exclusions_after_screening": len(quality_exclusions),
                    "by_source": dict(collections.Counter(r["source"] for r in selected)),
                    "by_topic": dict(collections.Counter(r["topic"] for r in selected)),
                    "benchmark_snapshot": result["manifest"]["snapshot_sha256"],
                    "benchmark_scope": result["manifest"]["scope"],
                    "candidate_file_sha256": hashlib.sha256((output / "candidates_5000.jsonl").read_bytes()).hexdigest()})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def prune_cache(output: Path, raw: Path) -> None:
    summary = json.loads((output / "summary.json").read_text())
    candidate_path = output / "candidates_5000.jsonl"
    content = candidate_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != summary["candidate_file_sha256"]:
        raise ValueError("Candidate file checksum differs; source caches have been retained")
    if len(content.splitlines()) != summary["selected_candidates"]:
        raise ValueError("Candidate file is incomplete; source caches have been retained")
    pool = output / "eligible_pool.jsonl.gz"
    if pool.exists():
        screened = {r["problem_id"] for r in json.loads((output / "screening.json").read_text())["records"]}
        unscreened = []
        with gzip.open(pool, "rt") as file:
            for line in file:
                row = json.loads(line)
                if row["problem_id"] in screened or row["selection_score"] < 8:
                    continue
                if not MATH.search(row["reference_answer"]) and not re.search(r"\d", row["reference_answer"]):
                    continue
                row["pending_checks"].append("benchmark_overlap_screening")
                unscreened.append(row)
        write_jsonl(output / "unscreened_reserve.jsonl.gz", unscreened)
        summary["unscreened_reserve"] = len(unscreened)
    removed = 0
    for download in summary["downloads"]:
        repo = download["repository"]
        path = raw / (repo.replace("/", "--") + (".parquet" if repo == TEXTBOOK else ".jsonl"))
        if path.exists():
            with path.open("rb") as file:
                checksum = hashlib.file_digest(file, "sha256").hexdigest()
            if checksum != download["sha256"]:
                raise ValueError(f"Cache checksum changed: {path}")
            removed += path.stat().st_size
            path.unlink()
    for name in ["eligible_pool.jsonl.gz", "textbook_physics.parquet", "additional_preflight.json", "additional_screening.json"]:
        path = output / name
        if path.exists():
            removed += path.stat().st_size
            path.unlink()
    summary["pruned_cache_bytes"] = summary.get("pruned_cache_bytes", 0) + removed
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Removed {removed:,} bytes of reproducible source caches", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=Path("/tmp/phy-training-downloads"))
    parser.add_argument("--screening", type=Path)
    parser.add_argument("--existing", type=Path)
    parser.add_argument("--download", action="store_true", help="Download and checksum the two pinned training releases")
    parser.add_argument("--prune-cache", action="store_true", help="Remove verified raw caches after finalization")
    parser.add_argument("--pool-size", type=int, default=14000)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    if args.prune_cache and not args.screening:
        if args.download:
            parser.error("Finalize screening before using --prune-cache with --download")
        prune_cache(args.output, args.raw_dir)
        return
    if args.screening:
        finalize(args.output, args.screening, args.limit, args.existing)
    else:
        if args.download:
            fetch_sources(args.raw_dir)
        prepare(args.raw_dir, args.output, args.pool_size)
    if args.prune_cache:
        prune_cache(args.output, args.raw_dir)


if __name__ == "__main__":
    main()
