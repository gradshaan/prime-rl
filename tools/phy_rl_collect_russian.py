"""Collect pre-2024 Russian regional/final theory papers and worked solutions."""

from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import re
from pathlib import Path
from urllib.parse import urljoin

import httpx
from phy_rl_collect import collect_pdf, fetch, read_document, tree, write_json, write_jsonl


def paper_pairs(html: str, archive: str, grade: int) -> list[dict]:
    groups = collections.defaultdict(dict)
    for anchor in tree(html).walk("a"):
        url = urljoin(archive, anchor.attrs.get("href", "").strip())
        match = re.search(r"/(tasks|sol|ans)-phys-(9|10|11)-teor-(reg|final)-(\d{2}|\d{4})-(\d{1,4})\.pdf$", url)
        if not match or int(match[2]) != grade:
            continue
        start = int(match[4]) + (2000 if len(match[4]) == 2 else 0)
        end = int(match[5])
        year = (start // 10 * 10 + end if len(match[5]) == 1 else
                2000 + end if len(match[5]) == 2 else end)
        if len(match[5]) == 1 and year <= start:
            year += 10
        if year > 2023 or year != start + 1:
            continue
        key = (year, match[3], grade)
        groups[key]["problem_url" if match[1] == "tasks" else "solution_url"] = url
    return [{"source": "russian_physics_olympiad", "competition": "RussianPhysicsOlympiad",
             "year": year, "grade": grade, "stage": stage,
             "paper_id": f"RussianPhysicsOlympiad_{year}_{stage}_grade{grade}",
             "archive_url": archive, "archive_kind": "native_competition_papers", **urls}
            for (year, stage, grade), urls in groups.items() if len(urls) == 2]


def original_headers(paper: dict, document: dict) -> list[dict]:
    grade = str(paper["grade"])
    pattern = re.compile(r"(?im)^\s*(?:Задача\s*(?:№\s*)?(?:" + grade + r"[.\-])?(\d{1,2})"
                         r"|" + grade + r"[.\-](\d{1,2}))\s*[.:)]?\s*([^\n]*)")
    identities, duplicate_labels = {}, set()
    for page in document["pages"]:
        for match in pattern.finditer(page["text"]):
            number = grade + "." + (match[1] or match[2])
            if number in identities:
                duplicate_labels.add(number)
                continue
            identities[number] = {"problem_id": paper["paper_id"] + "_problem_" + number,
                                  "problem_number": number, "heading": match[0].strip(),
                                  "source_question_page": page["page"], "topic": "unknown"}
    if duplicate_labels:
        raise ValueError("Repeated question labels need round/page review: " + ", ".join(sorted(duplicate_labels)))
    return list(identities.values())


async def collect(args) -> None:
    out = args.output
    (out / "archives").mkdir(parents=True, exist_ok=True)
    papers, errors, originals = [], [], []
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(follow_redirects=True, timeout=40) as client:
        async def archive(season: int, grade: int):
            url = f"https://olimpiada.ru/activity/74/tasks/{season}?class={grade}"
            path = out / "archives" / f"vso_{season}_{grade}.html.gz"
            try:
                raw = gzip.decompress(path.read_bytes()) if path.exists() else await fetch(client, semaphore, url)
                path.write_bytes(gzip.compress(raw, mtime=0))
                papers.extend(paper_pairs(raw.decode(), url, grade))
            except (httpx.HTTPError, ValueError) as exc:
                errors.append({"url": url, "error_type": type(exc).__name__, "reason": str(exc)[:300]})
        await asyncio.gather(*(archive(year, grade) for year in range(args.year_min - 1, 2023) for grade in [9, 10, 11]))
        papers = list({paper["paper_id"]: paper for paper in papers}.values())
        write_jsonl(out / "papers.jsonl", papers)

        async def paper_one(paper: dict):
            try:
                documents = {}
                for role in ["problem_url", "solution_url"]:
                    documents[role] = await collect_pdf(client, semaphore, out, paper[role])
                headers = original_headers(paper, read_document(out, documents["problem_url"]["document_key"]))
                if not headers:
                    raise ValueError("Question headings need OCR/boundary review; no problem count inferred")
                for header in headers:
                    originals.append({**paper, **header, "documents": documents,
                                      "status": "pdf_transcription_review", "language": "ru",
                                      "training_ready": False, "release_status": "held", "required_outputs": [],
                                      "license_status": "unknown; source attribution retained",
                                      "pending_checks": ["question_and_solution_boundary_review", "original_year_review",
                                                         "benchmark_screening", "source_translation_review", "physics_audit"]})
                print(f"Collected {paper['paper_id']}: {len(headers)} provisional original identities", flush=True)
            except (httpx.HTTPError, ValueError) as exc:
                errors.append({"paper_id": paper["paper_id"], "error_type": type(exc).__name__, "reason": str(exc)[:300]})
        await asyncio.gather(*(paper_one(paper) for paper in papers))
    originals.sort(key=lambda row: row["problem_id"])
    write_jsonl(out / "originals.jsonl", originals)
    write_json(out / "errors.json", errors)
    write_json(out / "status.json", {"state": "source_collection_complete", "paired_theory_papers": len(papers),
               "provisional_original_identities": len(originals), "released_tasks": 0, "paid_model_calls": 0,
               "api_cost_usd": 0, "collection_errors": len(errors), "year_max": 2023,
               "raw_pdfs_retained": False, "by_year": dict(collections.Counter(r["year"] for r in originals)),
               "scope": "Grades 9-11 regional and final theory, with source worked solutions. Source quarantine only; not model input."})
    print((out / "status.json").read_text(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--year-min", type=int, default=2010)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 2005 <= args.year_min <= 2023 or not 1 <= args.concurrency <= 8:
        parser.error("Use year 2005..2023 and concurrency 1..8")
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
