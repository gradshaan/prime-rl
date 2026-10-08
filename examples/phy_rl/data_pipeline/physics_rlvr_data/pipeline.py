from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import os
import re
import tomllib
import urllib.parse
import urllib.request
from dataclasses import replace
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

from physics_rlvr_common import validate_release_state

from .answer_extraction import extract_answer_draft, extract_answer_key
from .dedup import deduplicate
from .extract_pdf import extract_embedded_pdf_text, sha256_file
from .filters import admissibility_rejection
from .gemini_extract import DEFAULT_MODEL as DEFAULT_GEMINI_MODEL
from .gemini_extract import extract_with_gemini
from .glm_ocr_extract import extract_with_glm_ocr
from .io import ensure_parent, read_json, read_jsonl, write_json, write_jsonl
from .policy import choose_split, get_source_config, validate_training_policy
from .schema import (
    Answer,
    ExtractedDocument,
    FinalItem,
    Provenance,
    RejectedItem,
    SourceManifestItem,
    answer_from_dict,
    final_item_from_dict,
    problem_artifact_id,
    to_dict,
)
from .subproblem_curation import build_rlvr_subproblems
from .verifiers import extract_boxed

DEFAULT_DATA_DIR = Path("examples/phy_rl/data_pipeline/data")
Extractor = Literal["gemini", "glm-ocr", "embedded", "auto"]
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
PLAIN_PHYSICS_UNITS = {
    "a",
    "a/m",
    "c",
    "cm",
    "cm/s",
    "cm/s^2",
    "deg",
    "ev",
    "f",
    "g",
    "h",
    "hz",
    "j",
    "k",
    "kg",
    "km",
    "km/h",
    "m",
    "m/s",
    "m/s^2",
    "ma",
    "mev",
    "min",
    "mm",
    "mol",
    "mpa",
    "ms",
    "mv",
    "mw",
    "n",
    "nm",
    "pa",
    "rad",
    "rad/s",
    "s",
    "t",
    "v",
    "w",
    "wb",
}


class PdfLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href and href.lower().endswith(".pdf"):
            self.links.append(href)


def init_layout(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    for relative in [
        "raw_pdfs",
        "extracted",
        "candidates",
        "verified",
        "rejected",
        "final",
        "audits",
        "manual_corrections",
    ]:
        (data_dir / relative).mkdir(parents=True, exist_ok=True)


def manifest_from_local_pdfs(pdf_root: Path, out_path: Path, source_id: str = "local_pdf") -> int:
    rows = []
    source_config = get_source_config(source_id)
    for path in sorted(pdf_root.rglob("*.pdf")):
        metadata = infer_pdf_metadata(path)
        if metadata["competition"] == "unknown":
            metadata["competition"] = source_config["competition"]
        split = choose_split(
            source=source_id,
            competition=metadata["competition"],
            year=metadata["year"],
            requested_split=None,
        )
        item = SourceManifestItem(
            source_id=source_id,
            competition=metadata["competition"],
            year=metadata["year"],
            paper_type=metadata["paper_type"],
            url=None,
            local_path=str(path),
            sha256=sha256_file(path),
            problem_number=metadata["problem_number"],
            split=split,
        )
        rows.append(to_dict(item))
    return write_jsonl(out_path, rows)


def crawl_pdf_manifest(base_url: str, out_path: Path, source_id: str) -> int:
    source_config = get_source_config(source_id)
    request = urllib.request.Request(base_url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = PdfLinkParser()
    parser.feed(html)

    rows = []
    for href in sorted(set(parser.links)):
        url = urllib.parse.urljoin(base_url, href)
        metadata = infer_pdf_metadata(Path(urllib.parse.urlparse(url).path))
        if metadata["competition"] == "unknown":
            metadata["competition"] = source_config["competition"]
        split = choose_split(
            source=source_id,
            competition=metadata["competition"],
            year=metadata["year"],
            requested_split=None,
        )
        rows.append(
            to_dict(
                SourceManifestItem(
                    source_id=source_id,
                    competition=metadata["competition"],
                    year=metadata["year"],
                    paper_type=metadata["paper_type"],
                    url=url,
                    local_path="",
                    sha256="",
                    problem_number=metadata["problem_number"],
                    split=split,
                )
            )
        )
    return write_jsonl(out_path, rows)


def download_manifest(manifest_path: Path, raw_dir: Path, out_path: Path) -> int:
    rows = []
    for raw in read_jsonl(manifest_path):
        url = raw.get("url")
        if not url:
            if not raw.get("local_path"):
                raise ValueError(f"manifest row has neither url nor local_path: {raw}")
            rows.append(raw)
            continue
        target = raw_dir / _download_name(raw, url)
        ensure_parent(target)
        _download_url(url, target)
        raw["local_path"] = str(target)
        raw["sha256"] = sha256_file(target)
        rows.append(raw)
    return write_jsonl(out_path, rows)


def _download_url(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())


def extract_manifest(
    manifest_path: Path,
    out_dir: Path,
    *,
    extractor: Extractor = "gemini",
    vlm_work_dir: Path | None = None,
    vlm_model: str = DEFAULT_GEMINI_MODEL,
    vlm_prompt: str = "",
    vlm_max_new_tokens: int = 8192,
    vlm_dpi: int = 180,
    vlm_max_pages: int | None = None,
    jobs: int = 1,
    vlm_devices: list[str] | None = None,
) -> int:
    if vlm_work_dir is None:
        vlm_work_dir = out_dir / "_vlm_raw"
    rows = list(read_jsonl(manifest_path))
    if jobs > 1:
        return _extract_manifest_parallel(
            rows,
            out_dir,
            extractor=extractor,
            vlm_work_dir=vlm_work_dir,
            vlm_model=vlm_model,
            vlm_prompt=vlm_prompt,
            vlm_max_new_tokens=vlm_max_new_tokens,
            vlm_dpi=vlm_dpi,
            vlm_max_pages=vlm_max_pages,
            jobs=jobs,
            vlm_devices=vlm_devices,
        )
    return _extract_manifest_rows(
        rows,
        out_dir,
        extractor=extractor,
        vlm_work_dir=vlm_work_dir,
        vlm_model=vlm_model,
        vlm_prompt=vlm_prompt,
        vlm_max_new_tokens=vlm_max_new_tokens,
        vlm_dpi=vlm_dpi,
        vlm_max_pages=vlm_max_pages,
        cuda_device=None,
    )


def _extract_manifest_rows(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    extractor: Extractor,
    vlm_work_dir: Path,
    vlm_model: str,
    vlm_prompt: str,
    vlm_max_new_tokens: int,
    vlm_dpi: int,
    vlm_max_pages: int | None,
    cuda_device: str | None,
) -> int:
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    count = 0
    for raw in rows:
        item = SourceManifestItem(**raw)
        if not item.local_path:
            raise ValueError(f"manifest row has no downloaded/local PDF: {item.source_id} {item.url}")
        extracted = _extract_document(
            item,
            extractor=extractor,
            vlm_work_dir=vlm_work_dir,
            vlm_model=vlm_model,
            vlm_prompt=vlm_prompt,
            vlm_max_new_tokens=vlm_max_new_tokens,
            vlm_dpi=vlm_dpi,
            vlm_max_pages=vlm_max_pages,
        )
        artifact_id = problem_artifact_id(item)
        write_json(out_dir / f"{artifact_id}.json", to_dict(extracted))
        markdown_path = out_dir / f"{artifact_id}.md"
        ensure_parent(markdown_path)
        markdown_path.write_text(extracted.canonical_markdown, encoding="utf-8")
        count += 1
    return count


def _extract_manifest_parallel(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    extractor: Extractor,
    vlm_work_dir: Path,
    vlm_model: str,
    vlm_prompt: str,
    vlm_max_new_tokens: int,
    vlm_dpi: int,
    vlm_max_pages: int | None,
    jobs: int,
    vlm_devices: list[str] | None,
) -> int:
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if vlm_devices is None:
        vlm_devices = [str(index) for index in range(jobs)]
    if not vlm_devices:
        raise ValueError("vlm_devices must contain at least one device when jobs > 1")

    worker_count = min(jobs, len(vlm_devices))
    shards = [rows[index::worker_count] for index in range(worker_count)]
    total = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _extract_manifest_rows,
                shard,
                out_dir,
                extractor=extractor,
                vlm_work_dir=vlm_work_dir,
                vlm_model=vlm_model,
                vlm_prompt=vlm_prompt,
                vlm_max_new_tokens=vlm_max_new_tokens,
                vlm_dpi=vlm_dpi,
                vlm_max_pages=vlm_max_pages,
                cuda_device=vlm_devices[index],
            )
            for index, shard in enumerate(shards)
        ]
        for future in concurrent.futures.as_completed(futures):
            total += future.result()
    return total


def _extract_document(
    item: SourceManifestItem,
    *,
    extractor: Extractor,
    vlm_work_dir: Path,
    vlm_model: str,
    vlm_prompt: str,
    vlm_max_new_tokens: int,
    vlm_dpi: int,
    vlm_max_pages: int | None,
) -> ExtractedDocument:
    if extractor == "embedded":
        return extract_embedded_pdf_text(item)
    if extractor == "auto":
        embedded = extract_embedded_pdf_text(item)
        if not any(page.ocr_needed for page in embedded.page_classifications):
            return embedded
        extractor = "gemini"
    if extractor == "gemini":
        return extract_with_gemini(
            item,
            vlm_work_dir,
            model_name=vlm_model,
            prompt=vlm_prompt or None,
            max_output_tokens=vlm_max_new_tokens,
            dpi=vlm_dpi,
            max_pages=vlm_max_pages,
        )
    elif extractor != "glm-ocr":
        raise ValueError(f"unknown extractor: {extractor}")
    return extract_with_glm_ocr(
        item,
        vlm_work_dir,
        model_name=vlm_model,
        prompt=vlm_prompt,
        max_new_tokens=vlm_max_new_tokens,
        dpi=vlm_dpi,
        max_pages=vlm_max_pages,
    )


def build_candidates(manifest_path: Path, extracted_dir: Path, out_path: Path) -> int:
    documents: dict[tuple[str, str, int | None, str | None, str | None, str | None, str], dict[str, Any]] = {}
    manifests: dict[tuple[str, str, int | None, str | None, str | None, str | None, str], SourceManifestItem] = {}
    for raw in read_jsonl(manifest_path):
        item = SourceManifestItem(**raw)
        artifact_id = problem_artifact_id(item)
        extracted_path = extracted_dir / f"{artifact_id}.json"
        if not extracted_path.exists():
            raise FileNotFoundError(f"missing extraction artifact {extracted_path}")
        key = _manifest_pair_key(item)
        if key in documents:
            raise ValueError(f"duplicate manifest identity: {key}")
        documents[key] = read_json(extracted_path)
        manifests[key] = item

    candidates: list[dict[str, Any]] = []
    for key, problem_doc in documents.items():
        source_id, competition, year, round_name, paper_id, problem_number, paper_type = key
        if paper_type != "problem":
            continue
        solution_key = (source_id, competition, year, round_name, paper_id, problem_number, "solution")
        marking_key = (source_id, competition, year, round_name, paper_id, problem_number, "marking_scheme")
        solution_item = manifests.get(solution_key) or manifests.get(marking_key)
        solution_doc = documents.get(solution_key) or documents.get(marking_key)
        problem_item = manifests[key]
        if solution_doc is None or solution_item is None:
            rejected = RejectedItem(
                problem_id=_problem_id(problem_item),
                source=problem_item.source_id,
                rejection_reason="missing_solution",
                detail="No matching solution or marking scheme in manifest",
                item=to_dict(problem_item),
            )
            candidates.append(to_dict(rejected))
            continue

        split = choose_split(
            source=problem_item.source_id,
            competition=competition,
            year=year,
            requested_split=problem_item.split,
        )
        problem_text = _curate_document_section(problem_doc["canonical_markdown"], problem_number)
        solution_text = _curate_document_section(solution_doc["canonical_markdown"], problem_number)
        page_range = _document_page_range(problem_doc)
        final_item = FinalItem(
            problem_id=_problem_id(problem_item),
            source=problem_item.source_id,
            competition=competition,
            year=year,
            problem_number=problem_number,
            subproblem_id=None,
            problem_text=problem_text,
            shared_context="",
            question=problem_text,
            official_solution=solution_text,
            answers=[],
            requires_diagram=False,
            language=problem_item.language,
            split=split,
            provenance=Provenance(
                pdf_url=problem_item.url,
                page_range=page_range,
                ocr_engine=problem_doc["extractor"],
                ocr_confidence=None,
                source_hash=problem_item.sha256,
                solution_hash=solution_item.sha256,
                license_status=problem_item.license_status,
                source_url=problem_item.url or get_source_config(problem_item.source_id).get("base_url"),
            ),
            family_id=None,
        )
        candidates.append(to_dict(final_item))
    return write_jsonl(out_path, candidates)


def import_structured_jsonl(
    input_path: Path,
    out_path: Path,
    source_id: str,
    split: str | None,
    source_split: str | None = None,
    source_revision: str | None = None,
) -> int:
    if source_id == "physics_training_release":
        raise ValueError("PHYSICS is blocked as a training source")
    rows = []
    for index, raw in enumerate(read_jsonl(input_path)):
        if "problem_id" in raw and "provenance" in raw:
            item = final_item_from_dict(raw)
            if source_id == "physics_training_release":
                language = str(raw.get("language") or item.language).casefold()
                difficulty = re.sub(
                    r"\s+", " ", str(raw.get("difficulty") or item.difficulty or "").strip()
                ).casefold().replace(" (", "(")
                if language != "en" or difficulty not in {
                    "high school olympiad",
                    "undergraduate/postgraduate (physics major)",
                }:
                    rows.append(
                        to_dict(
                            RejectedItem(
                                problem_id=item.problem_id,
                                source=source_id,
                                rejection_reason="source_scope_excluded",
                                detail="PHYSICS import keeps only English Olympiad and physics-major records",
                                item=raw,
                            )
                        )
                    )
                    continue
            provenance = replace(
                item.provenance,
                source_split=raw.get("source_split") or raw.get("dataset_split") or item.provenance.source_split or source_split,
                source_revision=source_revision
                if source_id == "physics_training_release"
                else raw.get("source_revision") or raw.get("revision") or item.provenance.source_revision or source_revision,
                source_url=item.provenance.source_url or "https://github.com/Zhengsh123/PHYSICS"
                if source_id == "physics_training_release"
                else item.provenance.source_url,
            )
            rows.append(
                to_dict(
                    replace(
                        item,
                        source=source_id if source_id == "physics_training_release" else item.source,
                        provenance=provenance,
                    )
                )
            )
            continue
        question = str(raw.get("question") or raw.get("problem") or raw.get("problem_text") or "")
        official_solution = str(raw.get("official_solution") or raw.get("solution") or "")
        row_source_split = raw.get("source_split") or raw.get("dataset_split") or source_split
        row_source_revision = (
            source_revision
            if source_id == "physics_training_release"
            else raw.get("source_revision") or raw.get("revision") or source_revision
        )
        if source_id == "physics_training_release":
            language = str(raw.get("language") or "").casefold()
            difficulty = re.sub(r"\s+", " ", str(raw.get("difficulty") or "").strip()).casefold().replace(" (", "(")
            allowed_difficulties = {
                "high school olympiad",
                "undergraduate/postgraduate(physics major)",
            }
            if language != "en" or difficulty not in allowed_difficulties:
                rows.append(
                    to_dict(
                        RejectedItem(
                            problem_id=str(raw.get("problem_id") or f"{source_id}_{index:08d}"),
                            source=source_id,
                            rejection_reason="source_scope_excluded",
                            detail="PHYSICS import keeps only English Olympiad and physics-major records",
                            item=raw,
                        )
                    )
                )
                continue
        structured_answers = raw.get("structured_answers", [])
        if source_id == "physics_training_release" and not structured_answers:
            try:
                structured_answers = _physics_release_answers(raw)
            except ValueError as exc:
                rows.append(
                    to_dict(
                        RejectedItem(
                            problem_id=str(raw.get("problem_id") or raw.get("id") or f"{source_id}_{index:08d}"),
                            source=source_id,
                            rejection_reason="native_answer_needs_curation",
                            detail=str(exc),
                            item=raw,
                        )
                    )
                )
                continue
        try:
            answers = [_answer_from_structured(answer) for answer in structured_answers]
        except (AttributeError, TypeError, ValueError) as exc:
            rows.append(
                to_dict(
                    RejectedItem(
                        problem_id=str(raw.get("problem_id") or f"{source_id}_{index:08d}"),
                        source=str(raw.get("source") or source_id),
                        rejection_reason="malformed_answer_targets",
                        detail=str(exc),
                        item=raw,
                    )
                )
            )
            continue
        source = source_id if source_id == "physics_training_release" else str(raw.get("source") or source_id)
        competition = str(
            raw.get("competition")
            or ("PHYSICS training release" if source_id == "physics_training_release" else source)
        )
        year = _optional_year(raw.get("year"))
        chosen_split = raw.get("split") or split or "train"
        digest = hashlib.sha256(f"{source}\n{question}\n{official_solution}".encode("utf-8")).hexdigest()
        source_hash = hashlib.sha256(
            json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        solution_hash = hashlib.sha256(official_solution.encode("utf-8")).hexdigest()
        item = FinalItem(
            problem_id=str(raw.get("problem_id") or raw.get("id") or f"{source_id}_{index:08d}_{digest[:12]}"),
            source=source,
            competition=competition,
            year=year,
            problem_number=raw.get("problem_number"),
            subproblem_id=raw.get("subproblem_id"),
            problem_text=question,
            shared_context=str(raw.get("shared_context") or ""),
            question=question,
            official_solution=official_solution,
            answers=answers,
            requires_diagram=bool(raw.get("requires_diagram", False)),
            language=str(raw.get("language") or "en"),
            split=chosen_split,
            provenance=Provenance(
                pdf_url=raw.get("pdf_url"),
                page_range=raw.get("page_range"),
                ocr_engine=raw.get("ocr_engine") or "structured_dataset",
                ocr_confidence=raw.get("ocr_confidence"),
                source_hash=str(raw.get("source_hash") or source_hash),
                solution_hash=str(raw.get("solution_hash") or solution_hash),
                license_status=str(raw.get("license_status") or "unknown"),
                source_split=row_source_split,
                source_revision=row_source_revision,
                source_url=raw.get("source_url")
                or raw.get("provenance_url")
                or ("https://github.com/Zhengsh123/PHYSICS" if source_id == "physics_training_release" else None),
            ),
            family_id=raw.get("family_id"),
            topic=raw.get("topic") or raw.get("domain"),
            difficulty=raw.get("difficulty"),
        )
        rows.append(to_dict(item))
    return write_jsonl(out_path, rows)


def filter_candidates(input_path: Path, verified_path: Path, rejected_path: Path) -> tuple[int, int]:
    verified = []
    rejected = []
    for raw in read_jsonl(input_path):
        if "rejection_reason" in raw:
            rejected.append(raw)
            continue
        try:
            item = final_item_from_dict(raw)
            validate_training_policy(
                source=item.source,
                competition=item.competition,
                year=item.year,
                split=item.split,
                source_split=item.provenance.source_split,
                source_revision=item.provenance.source_revision,
            )
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append(
                to_dict(
                    RejectedItem(
                        problem_id=str(raw.get("problem_id") or f"invalid_{len(rejected):08d}"),
                        source=str(raw.get("source") or "unknown"),
                        rejection_reason="policy_or_schema_rejected",
                        detail=str(exc),
                        item=raw,
                    )
                )
            )
            continue
        rejection = admissibility_rejection(item)
        if rejection is not None:
            reason, detail = rejection
            rejected.append(to_dict(_reject(item, reason, detail)))
            continue
        verified.append(to_dict(item))
    return write_jsonl(verified_path, verified), write_jsonl(rejected_path, rejected)


def apply_answer_key(input_path: Path, answer_key_path: Path, out_path: Path) -> int:
    answer_key = list(read_jsonl(answer_key_path))
    rows = []
    for raw in read_jsonl(input_path):
        if "rejection_reason" in raw:
            rows.append(raw)
            continue
        item = final_item_from_dict(raw)
        match = _find_answer_key_match(item, answer_key)
        if match is not None:
            item = FinalItem(
                problem_id=item.problem_id,
                source=item.source,
                competition=item.competition,
                year=item.year,
                problem_number=item.problem_number,
                subproblem_id=match.get("subproblem_id", item.subproblem_id),
                problem_text=item.problem_text,
                shared_context=match.get("shared_context", item.shared_context),
                question=match.get("question", item.question),
                official_solution=item.official_solution,
                answers=[answer_from_dict(answer) for answer in match["answers"]],
                requires_diagram=bool(match.get("requires_diagram", item.requires_diagram)),
                language=item.language,
                split=match.get("split", item.split),
                provenance=item.provenance,
                family_id=item.family_id,
                topic=item.topic,
                difficulty=item.difficulty,
            )
        rows.append(to_dict(item))
    return write_jsonl(out_path, rows)


def extract_answer_key_file(input_path: Path, out_path: Path, review_path: Path, min_score: int = 4) -> tuple[int, int]:
    return extract_answer_key(input_path, out_path, review_path, min_score=min_score)


def extract_answer_draft_file(input_path: Path, out_path: Path, audit_path: Path) -> tuple[int, int]:
    return extract_answer_draft(input_path, out_path, audit_path)


def build_rlvr_subproblems_file(
    input_path: Path,
    verified_path: Path,
    review_path: Path,
    rejected_path: Path,
    min_score: int = 5,
    judge_model: str | None = None,
    audit_model: str | None = None,
) -> tuple[int, int, int]:
    return build_rlvr_subproblems(
        input_path,
        verified_path,
        review_path,
        rejected_path,
        min_score=min_score,
        judge_model=judge_model,
        audit_model=audit_model,
    )


def dedup_file(input_path: Path, out_path: Path, report_path: Path) -> int:
    items = [final_item_from_dict(raw) for raw in read_jsonl(input_path)]
    kept = deduplicate(items, report_path)
    return write_jsonl(out_path, [to_dict(item) for item in kept])


def export_final(input_path: Path, out_dir: Path) -> dict[str, int]:
    by_family: dict[str, list[FinalItem]] = {}
    for raw in read_jsonl(input_path):
        validate_release_state(raw)
        item = final_item_from_dict(raw)
        validate_training_policy(
            source=item.source,
            competition=item.competition,
            year=item.year,
            split=item.split,
            source_split=item.provenance.source_split,
            source_revision=item.provenance.source_revision,
        )
        rejection = admissibility_rejection(item)
        if rejection is not None:
            reason, detail = rejection
            raise ValueError(f"refusing to export {item.problem_id}: {reason}: {detail}")
        family_id = family_fingerprint(item)
        by_family.setdefault(family_id, []).append(replace(item, family_id=family_id))

    filters = _filter_config()
    max_variants = int(filters["families"]["max_parameter_variants"])
    family_fraction = float(filters["development"]["family_fraction"])
    seed = int(filters["development"]["seed"])
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    has_source_dev = any(item.split == "dev" for family in by_family.values() for item in family)
    if family_fraction > 0 and not has_source_dev and len(by_family) < 2:
        raise ValueError("at least two independent problem families are required for a non-empty dev split")
    family_splits = {
        family_id: "dev" if any(item.split == "dev" for item in family_items) else _family_split(
            family_id, fraction=family_fraction, seed=seed
        )
        for family_id, family_items in by_family.items()
    }
    if family_fraction > 0 and not has_source_dev and not any(split == "dev" for split in family_splits.values()):
        family_splits[min(family_splits, key=lambda family_id: _family_rank(family_id, seed))] = "dev"
    for family_id, family_items in sorted(by_family.items()):
        ordered_items = sorted(family_items, key=lambda item: item.problem_id)
        curated_items = ordered_items[:max_variants]
        split = family_splits[family_id]
        for item in curated_items:
            row = to_dict(replace(item, split=split))
            row["dataset_version"] = "physics_rlvr_v3"
            splits[split].append(row)

    if not splits["train"]:
        raise ValueError("refusing to export an empty training split")

    counts = {}
    for split, rows in splits.items():
        counts[split] = write_jsonl(out_dir / f"{split}.jsonl", rows)
    registry_path = Path(__file__).parents[1] / "configs/sources.toml"
    filters_path = Path(__file__).parents[1] / "configs/filters.toml"
    write_json(
        out_dir / "metadata.json",
        {
            "dataset_version": "physics_rlvr_v3",
            "counts": counts,
            "contract": "English text-only physics tasks with labeled deterministic numeric or symbolic verifiers.",
            "development_split": {
                "fraction": family_fraction,
                "seed": seed,
                "unit": "problem family",
            },
            "max_parameter_variants_per_family": max_variants,
            "families": {split: len({row["family_id"] for row in rows}) for split, rows in splits.items()},
            "sources": _counts_by(splits, "source"),
            "topics": _counts_by(splits, "topic"),
            "competitions": _counts_by(splits, "competition"),
            "answer_types": _answer_type_counts(splits),
            "policy_sha256": {
                "sources.toml": _file_sha256(registry_path),
                "filters.toml": _file_sha256(filters_path),
            },
            "file_sha256": {
                f"{split}.jsonl": _file_sha256(out_dir / f"{split}.jsonl") for split in splits
            },
        },
    )
    return counts


def write_validation_report(input_path: Path, report_path: Path) -> int:
    from .schema import validate_final_item

    ensure_parent(report_path)
    count = 0
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["problem_id", "error"])
        writer.writeheader()
        for raw in read_jsonl(input_path):
            item = final_item_from_dict(raw)
            for error in validate_final_item(item):
                writer.writerow({"problem_id": item.problem_id, "error": error})
                count += 1
    return count


def infer_pdf_metadata(path: Path) -> dict[str, Any]:
    text = " ".join([*path.parts, path.stem]).replace("_", " ")
    year_match = re.search(r"(19|20)\d{2}", text)
    competition_match = re.search(
        r"(?<![A-Za-z0-9])(IPhO|APhO|EuPhO|NBPhO|WoPhO|USAPhO|INPhO|OPhO|RMPh|PanPhO|PanMechanics|F=MA)(?![A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    problem_match = re.search(
        r"(?:problem|prob|question|solution|sol|[qps])[_\-\s]?(\d+[a-z]?)",
        text,
        flags=re.IGNORECASE,
    )
    lowered = text.lower()
    if "mark" in lowered or "scheme" in lowered:
        paper_type = "marking_scheme"
    elif re.search(r"(?:^|[_\-\s])s\d+[a-z]?(?:$|[_\-\s])", lowered) or "sol" in lowered or "answer" in lowered:
        paper_type = "solution"
    elif re.search(r"(?:^|[_\-\s])q\d+[a-z]?(?:$|[_\-\s])", lowered) or "problem" in lowered or "question" in lowered:
        paper_type = "problem"
    else:
        paper_type = "unknown"
    return {
        "competition": competition_match.group(1) if competition_match else "unknown",
        "year": int(year_match.group(0)) if year_match else None,
        "problem_number": problem_match.group(1) if problem_match else None,
        "paper_type": paper_type,
    }


def _curate_document_section(markdown: str, problem_number: str | None) -> str:
    if not problem_number:
        return markdown
    start_match = _find_problem_section_start(markdown, problem_number)
    if start_match is None:
        return markdown
    next_match = _find_next_problem_section_start(markdown, start_match.end())
    end = next_match.start() if next_match else len(markdown)
    return markdown[start_match.start() : end].strip() + "\n"


def _find_problem_section_start(markdown: str, problem_number: str) -> re.Match[str] | None:
    escaped = re.escape(problem_number)
    patterns = [
        rf"(?mi)^\s*(?:\*\*)?(?:problem|question)\s+{escaped}\b",
        rf"(?mi)^\s*(?:\*\*)?(?:solution\s+of\s+)?(?:problem|question|task)\s+{escaped}\b",
    ]
    matches = [match for pattern in patterns if (match := re.search(pattern, markdown))]
    return min(matches, key=lambda match: match.start()) if matches else None


def _find_next_problem_section_start(markdown: str, start: int) -> re.Match[str] | None:
    pattern = r"(?mi)^\s*(?:\*\*)?(?:problem|question|solution\s+of\s+(?:problem|question|task))\s+\d+[a-z]?\b"
    return re.compile(pattern).search(markdown, pos=start)


def _download_name(raw: dict[str, Any], url: str) -> Path:
    parsed_name = Path(urllib.parse.urlparse(url).path).name or "paper.pdf"
    competition = raw.get("competition") or "unknown"
    year = raw.get("year") or "unknown"
    paper_type = raw.get("paper_type") or "unknown"
    round_name = raw.get("round") or ""
    paper_id = raw.get("paper_id") or ""
    return Path(str(competition)) / str(year) / str(round_name) / str(paper_id) / str(paper_type) / parsed_name


def _manifest_pair_key(item: SourceManifestItem) -> tuple[str, str, int | None, str | None, str | None, str | None, str]:
    return (
        item.source_id,
        item.competition,
        item.year,
        item.round,
        item.paper_id,
        item.problem_number,
        item.paper_type,
    )


def _document_page_range(document: dict[str, Any]) -> list[int] | None:
    pages = [int(block["page"]) for block in document.get("blocks", []) if "page" in block]
    if not pages:
        return None
    return [min(pages), max(pages)]


def family_fingerprint(item: FinalItem) -> str:
    text = " ".join([item.problem_text, item.shared_context, item.question]).casefold()
    text = re.sub(r"(?<![a-z])(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", "<number>", text)
    text = re.sub(r"\\(?:alpha|beta|gamma|delta|theta|phi|omega|lambda|mu|rho|sigma)\b", "<var>", text)

    def normalize_math(match: re.Match[str]) -> str:
        expression = re.sub(r"\\[a-z]+", "<var>", match.group(1))
        expression = re.sub(r"(?<![a-z])([a-z])(?:_\{?[a-z0-9,]+\}?)?", "<var>", expression)
        return f"${expression}$"

    text = re.sub(r"\$(.*?)\$", normalize_math, text, flags=re.DOTALL)
    text = re.sub(r"\b[a-z]_[a-z0-9]+\b", "<var>", text)
    text = re.sub(r"\s+", " ", text).strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"family_{digest}"


def _family_split(family_id: str, *, fraction: float, seed: int) -> str:
    if not 0 <= fraction < 1:
        raise ValueError("development family_fraction must be in [0, 1)")
    digest = hashlib.sha256(f"{seed}:{family_id}".encode("utf-8")).digest()
    threshold = int(fraction * (2**32))
    return "dev" if int.from_bytes(digest[:4], "big") < threshold else "train"


def _family_rank(family_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{family_id}".encode("utf-8")).digest()


def _counts_by(splits: dict[str, list[dict[str, Any]]], key: str) -> dict[str, dict[str, int]]:
    return {
        split: dict(sorted(_count_values(row.get(key) or "unknown" for row in rows).items()))
        for split, rows in splits.items()
    }


def _answer_type_counts(splits: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    return {
        split: dict(
            sorted(
                _count_values(answer["answer_type"] for row in rows for answer in row["answers"]).items()
            )
        )
        for split, rows in splits.items()
    }


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _filter_config() -> dict[str, Any]:
    config_path = Path(__file__).parents[1] / "configs/filters.toml"
    with config_path.open("rb") as file:
        return tomllib.load(file)


def _problem_id(item: SourceManifestItem) -> str:
    return problem_artifact_id(item).replace("__problem__", "__")


def _reject(item: FinalItem, reason: str, detail: str) -> RejectedItem:
    return RejectedItem(
        problem_id=item.problem_id,
        source=item.source,
        rejection_reason=reason,
        detail=detail,
        item=to_dict(item),
    )


def _answer_from_structured(raw: dict[str, Any]) -> Answer:
    answer_type = str(raw.get("answer_type") or "string")
    if answer_type == "numerical":
        answer_type = "numeric"
    verifier = str(raw.get("verifier") or _default_verifier(answer_type))
    value_raw = raw.get("value")
    if value_raw is None:
        value_raw = raw.get("final_answer", "")
    value = str(value_raw)
    return answer_from_dict(
        {
            **raw,
            "value": value,
            "answer_type": answer_type,
            "verifier": verifier,
            "label": raw.get("label") or raw.get("output_label") or raw.get("subproblem_id"),
            "atol": raw.get("atol")
            if raw.get("atol") is not None
            else (_answer_atol(value) if verifier == "numeric" else None),
            "rtol": (
                raw.get("rtol")
                if raw.get("rtol") is not None
                else raw.get("tolerance", 1e-6)
            )
            if verifier == "numeric"
            else None,
        }
    )


def _physics_release_answers(raw: dict[str, Any]) -> list[dict[str, Any]]:
    raw_answers = raw.get("answer")
    raw_types = raw.get("answer_type")
    if not isinstance(raw_answers, list) or not isinstance(raw_types, list) or len(raw_answers) != len(raw_types):
        raise ValueError("native PHYSICS answer and answer_type lists must align")

    question = str(raw.get("question") or "")
    solution = str(raw.get("solution") or "")
    if not question.strip() or not solution.strip():
        raise ValueError("native PHYSICS row needs its question and solution")

    answer_units = raw.get("answer_units", raw.get("units", []))
    converted: list[dict[str, Any]] = []
    for answer_group, type_group in zip(raw_answers, raw_types, strict=True):
        values = answer_group if isinstance(answer_group, list) else [answer_group]
        types = type_group if isinstance(type_group, list) else [type_group] * len(values)
        if len(values) != len(types):
            raise ValueError("native PHYSICS answer types do not align with answer values")
        for value, answer_type in zip(values, types, strict=True):
            kind = str(answer_type).casefold()
            if kind == "numerical":
                normalized_type, verifier = "numeric", "numeric"
            elif kind == "expression":
                normalized_type, verifier = "symbolic", "sympy"
            else:
                raise ValueError(f"native PHYSICS answer type {answer_type!r} needs manual curation")
            if not isinstance(value, str):
                raise ValueError("native PHYSICS answers must be strings")
            boxed = extract_boxed(value)
            if len(boxed) == 1:
                answer_value = boxed[0]
            elif not boxed:
                answer_value = value.strip()
            else:
                raise ValueError("one native PHYSICS target contains multiple boxed answers")
            if not answer_value:
                raise ValueError("native PHYSICS answer is empty")

            label = f"output_{len(converted) + 1}"
            answer_unit = _physics_answer_unit(
                answer_value,
                question,
                solution,
                answer_units,
                len(converted),
                len(raw_answers),
            )
            if answer_unit is not None and normalized_type == "numeric":
                answer_value = _strip_physics_numeric_unit(answer_value, answer_unit)
            converted.append(
                {
                    "label": label,
                    "value": answer_value,
                    "unit": answer_unit,
                    "answer_type": normalized_type,
                    "verifier": verifier,
                    "atol": _answer_atol(answer_value) if verifier == "numeric" else None,
                    "rtol": 1e-6 if verifier == "numeric" else None,
                }
            )
    return converted


def _physics_answer_unit(
    answer_value: str,
    question: str,
    solution: str,
    answer_units: Any,
    answer_index: int,
    answer_group_count: int,
) -> str | None:
    if isinstance(answer_units, list) and len(answer_units) > answer_index:
        unit = answer_units[answer_index]
        if isinstance(unit, str):
            return unit or None
    embedded_unit = re.search(
        r"\\(?:mathrm|text|operatorname)\s*\{\s*~?([A-Za-zµμ°][A-Za-z0-9µμ°*/.^ -]*)\s*\}",
        answer_value,
    )
    if embedded_unit:
        return embedded_unit.group(1).strip()

    number = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", answer_value)
    if number is not None:
        matches = list(re.finditer(re.escape(number.group()), solution))
        if matches:
            suffix = solution[matches[-1].end() : matches[-1].end() + 100]
            latex_unit = re.search(
                r"\\(?:mathrm|text|operatorname)\s*\{\s*~?([A-Za-zµμ°][A-Za-z0-9µμ°*/.^ -]*)\s*\}",
                suffix,
            )
            if latex_unit:
                return latex_unit.group(1).strip()
            plain_unit = re.search(
                r"[\(\[]\s*([A-Za-zµμ°][A-Za-z0-9µμ°*/.^ -]{0,15})\s*[\)\]]",
                suffix,
            )
            if plain_unit:
                candidate = plain_unit.group(1).strip().rstrip(".,;")
                if candidate.casefold() in PLAIN_PHYSICS_UNITS:
                    return candidate
            trailing_unit = re.match(
                r"\s*(?:(?:[eE][+-]?\d+)|(?:(?:\\times|×)\s*10\^\{?[+-]?\d+\}?))?"
                r"\s*(?:[,=]\s*|\s+)([A-Za-zµμ][A-Za-z0-9µμ*/.^-]*)\b",
                suffix,
            )
            if trailing_unit:
                candidate = trailing_unit.group(1).rstrip(".,;")
                if candidate.casefold() in PLAIN_PHYSICS_UNITS:
                    return candidate

    dimensionless_clue = re.search(
        r"\b(?:dimensionless|magnification|ratio|coefficient|refractive index|index of refraction|"
        r"quantum number|probability|efficiency|how many|number of fringes|number of modes)\b",
        question,
        flags=re.IGNORECASE,
    )
    if answer_group_count == 1 and dimensionless_clue:
        return None
    raise ValueError("native PHYSICS target has no recoverable unit or dimensionless cue")


def _strip_physics_numeric_unit(value: str, unit: str) -> str:
    cleaned = re.sub(r"\\(?:mathrm|text|operatorname)\s*\{[^{}]*\}", "", value)
    cleaned = re.sub(r"\\(?:,|;|quad)\s*", "", cleaned)
    cleaned = re.sub(r"\s*[\(\[]\s*[\)\]]", "", cleaned)
    cleaned = re.sub(r"\s*[\(\[]\s*" + re.escape(unit) + r"\s*[\)\]]\s*$", "", cleaned)
    return cleaned.strip()


def _optional_year(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("year must be an integer, not a boolean")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"year must be an integer, got {value!r}")


def _answer_atol(value: str) -> float:
    cleaned = re.sub(r"\\(?:text|mathrm|mbox)\{([^{}]*)\}", r"\1", value)
    match = re.fullmatch(
        r"\s*[+-]?(?P<int>\d+)(?:\.(?P<frac>\d+))?\s*(?:[eE](?P<plain_exp>[+-]?\d+)|\\times\s*10\^\{?(?P<tex_exp>[+-]?\d+)\}?)?\s*",
        cleaned,
    )
    if match is None or match.group("frac") is None:
        return 0.0
    exponent = int(match.group("plain_exp") or match.group("tex_exp") or 0)
    return 0.5 * 10 ** (exponent - len(match.group("frac")))


def _default_verifier(answer_type: str) -> str:
    if answer_type == "numeric":
        return "numeric"
    if answer_type in {"symbolic", "expression"}:
        return "sympy"
    if answer_type == "multiple_choice":
        return "mcq"
    if answer_type in {"set", "multi_select"}:
        return "set"
    return "string"


def _find_answer_key_match(item: FinalItem, answer_key: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in answer_key:
        if "answers" not in row:
            raise ValueError("answer key row must contain answers")
        if row.get("problem_id") == item.problem_id:
            return row
        if (
            row.get("source") == item.source
            and str(row.get("year")) == str(item.year)
            and str(row.get("problem_number")) == str(item.problem_number)
        ):
            return row
    return None
