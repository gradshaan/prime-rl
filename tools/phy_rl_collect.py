"""Collect original olympiad sources without model calls or training exports."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import datetime as dt
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit

import fitz
import httpx

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples/phy_rl/data_pipeline/data"
REVISION = "67f7a7dc2f6fe955a346155a1b59dc85a2ad28f1"
ARCHIVES = {
    "ipho_olimpicos": ("IPhO", "https://olimpicos.net/ipho/", "mirror"),
    "apho_archive": ("APhO", "https://olimpicos.net/apho/", "mirror"),
    "wopho_archive": ("WoPhO", "https://olimpicos.net/wopho/", "mirror"),
    "usapho_archive": ("USAPhO", "https://www.aapt.org/physicsteam/PT-exams.cfm", "official"),
    "eupho_archive": ("EuPhO", "https://eupho.ee/archive/", "official"),
    "nbpho_archive": ("NBPhO", "https://nbpho.ee/archive/", "official"),
    "inpho_archive": ("INPhO", "https://olympiads.hbcse.tifr.res.in/how-to-prepare/past-papers/", "official"),
    "physics_naboj_archive": ("PhysicsNaboj", "https://physics.naboj.org/sk/en/archive/", "official"),
    "nbpho_olimpicos": ("NBPhO", "https://olimpicos.net/nbpho/", "mirror"),
    "australian_physics_olympiad": ("AustralianPhysicsOlympiad", "https://asi.edu.au/program/australian-science-olympiads/past-exams-with-answers/physics-olympiad-past-exams/", "official"),
    "czech_physics_olympiad": ("CzechPhysicsOlympiad", "https://fyzikalniolympiada.cz/archiv/zadani-a-reseni", "official"),
}
MAX_BYTES = 20 * 1024 * 1024
FIGURE = re.compile(r"\\(?:includegraphics|begin\{(?:tikzpicture|pspicture))|\b(?:joonis\w*|figure|diagram|shown)\b", re.I)
QUALITATIVE = re.compile(r"\b(?:joonista\w*|joonest\w*|põhjenda\w*|tõesta\w*|sketch|draw|prove|explain)\b", re.I)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


class Node:
    def __init__(self, tag: str, attrs: dict | None = None, parent: Node | None = None):
        self.tag, self.attrs, self.parent = tag, attrs or {}, parent
        self.children: list[Node | str] = []

    def text(self) -> str:
        return " ".join(c.text() if isinstance(c, Node) else c for c in self.children)

    def walk(self, tag: str | None = None):
        if tag is None or self.tag == tag:
            yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk(tag)


class TreeParser(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.root = self.current = Node("root")

    def handle_starttag(self, tag, attrs):
        node = Node(tag, dict(attrs), self.current)
        self.current.children.append(node)
        if tag not in self.VOID:
            self.current = node

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        node = self.current
        while node.parent is not None:
            if node.tag == tag:
                self.current = node.parent
                return
            node = node.parent

    def handle_data(self, data):
        self.current.children.append(data)


def tree(html: str) -> Node:
    parser = TreeParser()
    parser.feed(html)
    return parser.root


def heading_year(node: Node) -> int | None:
    parent = node.parent
    while parent is not None:
        headings = [h for h in parent.walk() if h.tag in {"h2", "h3", "h4"}]
        years = {int(year) for h in headings for year in re.findall(r"\b(?:19|20)\d{2}\b", h.text())}
        if len(years) == 1:
            return years.pop()
        if "wp-block-column" in parent.attrs.get("class", "").split():
            years = {int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", parent.text())}
            if len(years) == 1:
                return years.pop()
        parent = parent.parent
    return None


def archive_records(source: str, html: str) -> list[dict]:
    competition, base, kind = ARCHIVES[source]
    root, records = tree(html), []
    if kind == "mirror":
        for li in root.walk("li"):
            labels = [n.text().strip() for n in li.walk("span") if n.attrs.get("class") == "rtag"]
            if competition not in {"WoPhO", "NBPhO"} and "Theory" not in labels:
                continue
            links = list(li.walk("a"))
            problems = [a for a in links if "k-problems" in a.attrs.get("class", "").split()]
            solutions = [a for a in links if "k-solutions" in a.attrs.get("class", "").split()]
            if not problems:
                continue
            href = problems[0].attrs["href"]
            m = re.search(r"_(\d{4})_Q(\d+)\.pdf$", href)
            if not m or int(m[1]) > 2023:
                continue
            titles = [n.text().strip() for n in li.walk("span") if n.attrs.get("class") == "t"]
            records.append({"source": source, "competition": competition, "year": int(m[1]),
                            "problem_number": m[2], "problem_id": f"{competition}_{m[1]}_problem_{m[2]}",
                            "title": titles[0] if titles else "", "topic": li.attrs.get("data-t", "unknown"),
                            "problem_url": urljoin(base, href),
                            "solution_url": urljoin(base, solutions[0].attrs["href"]) if solutions else None,
                            "archive_url": base, "archive_kind": kind})
        return records
    # These archives publish whole papers. Problem identities come from PDF headings,
    # never from a presumed number of questions per year.
    papers: dict[tuple[int, str], dict] = {}
    current_edition = current_year = None
    for node in root.walk():
        if source == "czech_physics_olympiad" and node.tag == "h3":
            heading = re.search(r"(\d+)\.\s*ročník\s*\(((?:19|20)\d{2})[–-]((?:19|20)\d{2})\)", node.text())
            current_edition, current_year = (int(heading[1]), int(heading[3])) if heading else (None, None)
        if node.tag != "a" or not node.attrs.get("href", "").lower().endswith(".pdf"):
            continue
        url = urljoin(base, node.attrs["href"])
        name, label = urlsplit(url).path.rsplit("/", 1)[-1], node.text().strip().lower()
        if source == "czech_physics_olympiad":
            match = re.fullmatch(r"fo(\d+)([abc])([123])_([zr])\.pdf", name, re.I)
            if not match or int(match[1]) != current_edition:
                continue
            year, variant = current_year, f"{match[2].upper()}_round{match[3]}"
            role = "problem" if match[4].lower() == "z" else "solution"
            url = urljoin("https://fyzikalniolympiada.cz/", node.attrs["href"])
        elif source == "usapho_archive":
            parent_text = node.parent.text().lower() if node.parent else ""
            if not re.search(r"usapho|semi.?final|training camp theoretical", parent_text):
                continue
            if re.search(r"f.?ma|fnet|experiment|mystery|qtr|quarter", name, re.I):
                continue
            year = heading_year(node)
            variant = "plus" if "plus" in name.lower() else "camp" if "tst" in name.lower() else "main"
            role = "solution" if "solution" in name.lower() or "soln" in name.lower() else "problem"
        elif source == "australian_physics_olympiad":
            m = re.search(r"^((?:19|20)\d{2})[-_].*physics", name, re.I)
            if not m or "booklet" in name.lower():
                continue
            year, variant = int(m[1]), "main"
            role = "solution" if re.search(r"answer|marking|solution", name, re.I) else "problem"
        elif source == "inpho_archive":
            m = re.search(r"(?:INPHO|IOQP)[_-]?(\d{4})", name, re.I)
            if not m or re.search(r"(?:-hi\b|answerbooklet)", name, re.I):
                continue
            year, variant = int(m[1]), "ioqp" if name.lower().startswith("ioqp") else "main"
            role = "combined" if re.search(r"-Q-S\.pdf", name, re.I) else "solution" if re.search(r"solution|-S\.pdf", name, re.I) else "problem"
        else:
            if source == "eupho_archive" and label not in {"theory problems", "theory solutions"}:
                continue
            if source == "nbpho_archive" and label not in {"problems in english", "solutions in english"}:
                continue
            year = heading_year(node)
            variant = "main"
            role = "solution" if "solutions" in label else "problem"
        if year is None or not 1967 <= year <= 2023:
            continue
        paper = papers.setdefault((year, variant), {"source": source, "competition": competition, "year": year,
            "paper_variant": variant, "paper_id": f"{competition}_{year}_{variant}", "archive_url": base,
            "archive_kind": kind, "problem_url": None, "solution_url": None})
        paper[f"{'solution' if role == 'solution' else 'problem'}_url"] = url
        if role == "combined":
            paper["solution_url"] = url
        if source == "czech_physics_olympiad":
            paper["language"] = "cs"
            paper["category"] = variant.split("_")[0]
            paper["round"] = variant.split("_")[1]
            paper["competition_edition"] = current_edition
    return list(papers.values())


async def fetch(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, url: str) -> bytes:
    async with semaphore:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            raw = bytearray()
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > MAX_BYTES:
                    raise ValueError(f"Source exceeds {MAX_BYTES} bytes: {url}")
            return bytes(raw)


def read_document(out: Path, key: str) -> dict:
    return json.loads(gzip.decompress((out / "documents" / f"{key}.json.gz").read_bytes()))


async def collect_pdf(client, semaphore, out: Path, url: str) -> dict:
    key = digest(url.encode())
    path = out / "documents" / f"{key}.json.gz"
    if path.exists():
        doc = read_document(out, key)
        if doc["url"] != url or digest(json.dumps(doc["pages"], sort_keys=True).encode()) != doc["extraction_sha256"]:
            raise ValueError(f"Corrupt cached extraction: {path}")
        meta = {k: v for k, v in doc.items() if k != "pages"}
        meta["private_use_glyph_count"] = sum(sum(0xE000 <= ord(c) <= 0xF8FF for c in p["text"]) for p in doc["pages"])
        return meta
    raw = await fetch(client, semaphore, url)
    if not raw.startswith(b"%PDF-"):
        raise ValueError(f"Not a PDF: {url}")
    with fitz.open(stream=raw, filetype="pdf") as pdf:
        pages = [{"page": i + 1, "text": p.get_text("text"), "image_count": len(p.get_images()),
                  "drawing_count": len(p.get_drawings()), "replacement_chars": p.get_text("text").count("\ufffd")}
                 for i, p in enumerate(pdf)]
    doc = {"url": url, "document_key": key, "pdf_sha256": digest(raw), "download_bytes": len(raw),
           "page_count": len(pages), "extraction_sha256": digest(json.dumps(pages, sort_keys=True).encode()),
           "extractor": "pymupdf_embedded_text", "needs_ocr_pages": [p["page"] for p in pages if len(p["text"].strip()) < 80 or p["replacement_chars"] > 2],
           "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "pages": pages,
           "raw_pdf_retained": False, "formula_transcription_review_required": True}
    doc["private_use_glyph_count"] = sum(sum(0xE000 <= ord(c) <= 0xF8FF for c in p["text"]) for p in pages)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(doc, ensure_ascii=False).encode(), mtime=0))
    return {k: v for k, v in doc.items() if k != "pages"}


def paper_problems(paper: dict, doc: dict) -> list[dict]:
    full = "\n".join(p["text"] for p in doc["pages"])
    if paper["competition"] == "USAPhO":
        pattern = r"(?m)^\s*(?:Question|Problem)\s+([AB][1-6])\b[^\n]*"
    elif paper["competition"] == "EuPhO":
        pattern = r"(?m)^\s*(?:T|Problem\s+)?([1-3])\s*[.:]\s+[^\n]{3,100}"
    elif paper["competition"] == "NBPhO":
        pattern = r"(?m)^\s*(\d{1,2})\.\s*[^\n]{0,100}\(\s*\d+\s*(?:p|points)\b[^\n]*"
    elif paper["competition"] == "PhysicsNaboj":
        full = re.split(r"(?m)^\s*(?:Solutions|Riešenia|Řešení)\s*$", full, maxsplit=1)[0]
        pattern = r"(?m)^(\d{1,2})[ \t]*\n(?!(?:https?://|info-|Problems|Solutions))"
    elif paper["competition"] == "CzechPhysicsOlympiad":
        pattern = r"(?m)^\s*([1-7])\.\s+[^\n]{3,100}"
    else:
        pattern = r"(?m)^\s*(?:Question|Problem|Q\.)?\s*([1-9])\s*[.)]\s+[^\n]{3,100}"
    matches = list(re.finditer(pattern, full, re.I))
    if paper["competition"] == "INPhO":
        instructions = re.compile(r"booklet|roll number|roll no|submit|programmable|rough work|answer ?sheet|answer book|returned|examination|blue or black|marks will be awarded", re.I)
        matches = [m for m in matches if not instructions.search(m[0])]
    seen, rows = set(), []
    for i, match in enumerate(matches):
        number = match[1]
        if paper["competition"] == "PhysicsNaboj" and int(number) != len(rows) + 1:
            continue
        if number in seen:
            continue
        seen.add(number)
        identity = f"{paper['competition']}_{paper['year']}_problem_{number}"
        if paper.get("paper_variant") != "main":
            identity = f"{paper['competition']}_{paper['year']}_{paper['paper_variant']}_problem_{number}"
        excerpt = full[match.start():matches[i + 1].start() if i + 1 < len(matches) else len(full)]
        rows.append({**paper, "problem_id": identity, "problem_number": number,
                     "heading": match[0].strip(), "question_excerpt": excerpt,
                     "boundary_detection": "heuristic_requires_review", "topic": "unknown"})
    return rows


async def naboj_papers(client, semaphore, out: Path, html: str) -> list[dict]:
    source = "physics_naboj_archive"
    competition, base, _ = ARCHIVES[source]
    pages = []
    for anchor in tree(html).walk("a"):
        href = anchor.attrs.get("href", "")
        if not re.search(r"/archive/\d+/problems/", href):
            continue
        year = heading_year(anchor)
        if year is not None and year <= 2023:
            pages.append((year, urljoin(base, href)))

    async def one(year, url):
        path = out / "archives" / f"naboj_{year}.html.gz"
        raw = gzip.decompress(path.read_bytes()) if path.exists() else await fetch(client, semaphore, url)
        path.write_bytes(gzip.compress(raw, mtime=0))
        links = [urljoin(url, a.attrs["href"]) for a in tree(raw.decode()).walk("a")
                 if a.attrs.get("href", "").endswith(".pdf")]
        if len(links) != 1:
            raise ValueError(f"Expected one official combined problem/solution PDF: {url}")
        language = "en" if "-en.pdf" in links[0] else "sk"
        return {"source": source, "competition": competition, "year": year, "paper_variant": "main",
                "paper_id": f"{competition}_{year}_main", "archive_url": url, "archive_kind": "official",
                "problem_url": links[0], "solution_url": links[0], "language": language,
                "priority": "later_numbered_problems_first; verify_difficulty_from_content"}

    return await asyncio.gather(*(one(year, url) for year, url in pages))


async def collect_native(client, semaphore, out: Path, evaluation_ids: set[str], used: set[str]) -> list[dict]:
    tree_url = f"https://api.github.com/repos/Majakas/physics-collection/git/trees/{REVISION}?recursive=1"
    cache = out / "estonia_tree.json"
    if not cache.exists():
        write_json(cache, json.loads(await fetch(client, semaphore, tree_url)))
    listing = json.loads(cache.read_text())
    if listing["sha"] != REVISION or listing.get("truncated"):
        raise ValueError("Native repository tree is incomplete or has a different revision")

    async def one(entry):
        sid = Path(entry["path"]).stem
        year = int(sid[:4])
        row = {"source": "estonian_physics_olympiad", "competition": "Estonian Physics Olympiad", "year": year,
               "problem_id": f"estonian-{sid}", "source_id": sid, "problem_number": sid.rsplit("-", 1)[-1],
               "source_revision": REVISION, "git_blob_sha1": entry["sha"], "archive_kind": "native_repository",
               "source_url": f"https://raw.githubusercontent.com/Majakas/physics-collection/{REVISION}/{entry['path']}",
               "language": "et", "license_status": "CC-BY-NC-4.0; commercial use needs permission",
               "already_curated": sid in used, "release_state": "source_staging"}
        if row["problem_id"] in evaluation_ids:
            return {**row, "status": "excluded_evaluation_identity"}
        path = out / "native" / f"{sid}.tex"
        raw = path.read_bytes() if path.exists() else await fetch(client, semaphore, row["source_url"])
        if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != entry["sha"]:
            raise ValueError(f"Native source hash mismatch: {sid}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        tex = raw.decode()
        statement = re.split(r"\\prob\{[^}]*\}", tex, maxsplit=1)
        statement = re.split(r"\\(?:hint|solu)\b", statement[1], maxsplit=1)[0].strip() if len(statement) == 2 else ""
        solution = re.split(r"\\(?:probeng|probend)\b", tex.split(r"\solu", 1)[1], maxsplit=1)[0].strip() if r"\solu" in tex else ""
        english = re.split(r"\\probeng\{[^}]*\}", tex, maxsplit=1)
        english = re.split(r"\\(?:hinteng|solueng|probend)\b", english[1], maxsplit=1)[0].strip() if len(english) == 2 else ""
        meta = {k: m[1] if (m := re.search(r"\\set" + k + r"\{([^}]*)\}", tex)) else None for k in ["Topic", "Difficulty"]}
        reasons = []
        if not statement or not solution:
            reasons.append("missing_statement_or_solution")
        if FIGURE.search(statement) or FIGURE.search(english):
            reasons.append("diagram_dependency_review")
        if QUALITATIVE.search(statement) or QUALITATIVE.search(english):
            reasons.append("qualitative_output_review")
        return {**row, "source_hash": digest(raw), "local_source": str(path.resolve()),
                "question": english or statement, "original_statement": statement,
                "screening_language": "en" if english else "et", "existing_english_translation": bool(english),
                "statement_sha256": digest(statement.encode()),
                "solution_present": bool(solution), "topic": meta["Topic"] or "unknown", "difficulty": meta["Difficulty"],
                "hold_reasons": reasons, "status": "already_curated" if sid in used else "source_review_pending" if reasons else "native_preflight_candidate"}

    tasks = [one(entry) for entry in listing["tree"] if re.fullmatch(r"problems/(?:200[5-9]|201[0-8])-[\w-]+\.tex", entry["path"])]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        raise ExceptionGroup("Native source collection failed; rerun resumes verified cache", failures)
    return results


def prepare_native_queue(out: Path, native: list[dict], snapshot: str) -> dict:
    path = out / "native_screening.json"
    if not path.exists():
        return {"screened_native_candidates": 0, "queued_for_curation": 0}
    screening = json.loads(path.read_text())
    if screening["manifest"]["snapshot_sha256"] != snapshot:
        raise ValueError("Native screening used a different evaluation snapshot")
    checks = {r["problem_id"]: r for r in screening["records"]}
    pending = [r for r in native if r["status"] == "native_preflight_candidate"]
    queue, held = [], []
    for row in pending:
        check = checks.get(row["problem_id"])
        if check is None or check["question_sha256"] != digest(row["question"].encode()):
            held.append({**row, "status": "screening_missing_or_stale"})
        elif check["status"] == "clear":
            queue.append({**row, "status": "queued_for_curation", "screening": check,
                          "admission_scope": "source_preflight_only; final English question must be screened again"})
        else:
            held.append({**row, "status": "overlap_review", "screening": check})
    buckets = {}
    for row in queue:
        buckets.setdefault(row["topic"], []).append(row)
    ordered = []
    while any(buckets.values()):
        for topic in sorted(buckets):
            if buckets[topic]:
                ordered.append(buckets[topic].pop(0))
    write_jsonl(out / "curation_queue.jsonl", ordered)
    write_jsonl(out / "overlap_review.jsonl", held)
    items = [{k: r[k] for k in ["source_id", "source", "local_source", "source_url", "source_hash",
                              "source_revision", "license_status", "git_blob_sha1"]} for r in ordered]
    write_json(out / "next_native_batch.json", {"source_ids": [r["source_id"] for r in ordered], "items": items,
               "release_state": "source_staging", "evaluation_snapshot_sha256": snapshot})
    return {"screened_native_candidates": len(checks), "queued_for_curation": len(ordered),
            "native_overlap_review": len(held), "queued_native_topics": dict(Counter(r["topic"] for r in ordered))}


def pilot_review_queue(out: Path) -> dict:
    held, checked, total = [], 0, 0
    for name in ["pilot_50", "pilot_diverse_12", "pilot_expand_12"]:
        path = DATA / name / "pilot.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text())["records"]:
            total += 1
            if row["status"] == "model_checked":
                checked += 1
                continue
            reasons = row.get("failed_checks", [])
            category = "incomplete_independent_audit" if not row["checks"].get("independent_answer_schema") else "physics_or_answer_contract_disagreement"
            held.append({"problem_id": row["problem_id"], "source_id": row["source_id"], "batch": name,
                         "question_sha256": digest(row["question"].encode()), "review_category": category,
                         "failed_checks": reasons, "existing_notes": row.get("review_notes"),
                         "status": "held_for_final_review", "release_state": "staging",
                         "artifact_path": str(path), "next_action": "source-to-target derivation and output coverage review; retain failed audit"})
    write_jsonl(out / "pilot_final_review.jsonl", held)
    return {"existing_pilot_originals": total, "existing_model_checked": checked, "existing_held_for_review": len(held)}


def summary(out: Path, native: list[dict], papers: list[dict], originals: list[dict], docs: list[dict], errors: list[dict]) -> dict:
    unique = {r["problem_id"]: r for r in [*native, *originals]}
    counts = Counter(r["status"] for r in unique.values())
    result = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "release_state": "source_staging",
              "paid_model_calls": 0, "api_cost_usd": 0, "year_max": 2023, "target_original_problems": 4000,
              "indexed_original_identities": len(unique), "identity_counts_by_status": dict(counts),
              "identity_counts_by_source": dict(Counter(r["source"] for r in unique.values())),
              "native_topics": dict(Counter(r.get("topic", "unknown") for r in native if r["status"] == "native_preflight_candidate")),
              "archive_papers": len(papers), "extracted_documents": len(docs), "collection_errors": len(errors),
              "documents_with_private_use_glyphs": sum(bool(d.get("private_use_glyph_count")) for d in docs),
              "documents_with_ocr_needed_pages": sum(bool(d.get("needs_ocr_pages")) for d in docs),
              "raw_pdf_bytes_not_retained": sum(d.get("download_bytes", 0) for d in docs),
              "retained_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
              "prepared_training_safe_problems": 0,
              "notes": ["Original problem identities are counted once; subanswers and solution PDFs do not add problems.",
                        "PDF problem boundaries are provisional until checked against the source pages.",
                        "Native preflight candidates still need translation, semantic overlap review and answer verification.",
                        "Known evaluation identities are excluded before problem-specific downloads where possible.",
                        "Whole-paper extracts can contain excluded problems: these are source quarantine, never model input.",
                        "PDFs are read in memory and discarded. URL, SHA-256 and compressed page extraction allow checked refetch.",
                        "PHYSICS, benchmark-origin training sources, post-2023 competitions and IPhO 2026 are excluded.",
                        "OpenStax is skipped because current terms require permission for LLM ingestion."]}
    write_json(out / "summary.json", result)
    rows = ["# Physics source collection", "", "This directory holds source material, not a training release.", "",
            f"Indexed original identities: **{len(unique)}**. Paid model calls: **0**.", "",
            "| Status | Original identities |", "| --- | ---: |"]
    rows.extend(f"| {status} | {count} |" for status, count in sorted(counts.items()))
    rows += ["", "| Source | Original identities |", "| --- | ---: |"]
    rows.extend(f"| {source} | {count} |" for source, count in sorted(result["identity_counts_by_source"].items()))
    rows += ["", f"Extracted {len(docs)} documents; retained approximately {result['retained_bytes'] / 1024**2:.1f} MiB.",
             f"{len(errors)} collection errors are recorded in `errors.json`; failed downloads remain held.", "",
             "## Admission", "", *[f"- {note}" for note in result["notes"]], "",
             "## Next batch", "", "`native_preflight.json` indexes unused native statements and their local solution files.",
             "Run the frozen evaluation screen, inspect similarity holds, then curate and audit each accepted source.",
             "`papers.jsonl` and `originals.jsonl` record PDF provenance and provisional problem boundaries.",
             "Check PDF formulas and figure dependencies before turning these extracts into model inputs.", ""]
    (out / "REPORT.md").write_text("\n".join(rows))
    return result


async def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((args.eval_cache / "questions.json").read_text())
    evaluation_ids = {r["source_id"] for r in evaluation if isinstance(r.get("source_id"), str)}
    evaluation_manifest = json.loads((args.eval_cache / "manifest.json").read_text())
    write_json(out / "exclusion_snapshot.json", {"snapshot_sha256": evaluation_manifest["snapshot_sha256"],
               "benchmarks": evaluation_manifest["scope"], "excluded_source_ids": sorted(evaluation_ids)})
    used = {p.stem for batch in DATA.glob("pilot*") for p in (batch / "raw").glob("*.tex")}
    used_identities = {f"estonian-{sid}" for sid in used if re.fullmatch(r"\d{4}-[\w-]+", sid)}
    used_identities.update(f"USAPhO_{match[1]}_problem_{match[2].upper()}" for sid in used
                           if (match := re.fullmatch(r"usapho-(\d{4})-([a-z]\d+)", sid)))
    semaphore = asyncio.Semaphore(args.workers)
    errors, papers, originals, docs = [], [], [], []
    def offline_request(request):
        raise httpx.RequestError(f"Offline mode: source has no cached extraction: {request.url}", request=request)

    transport = httpx.MockTransport(offline_request) if args.offline else None
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30), transport=transport,
                               limits=httpx.Limits(max_connections=args.workers),
                               headers={"User-Agent": "PhysicsSourceCuration/1.0 (private research; bounded concurrency)"}) as client:
        native = await collect_native(client, semaphore, out, evaluation_ids, used)
        print(f"Native sources: {dict(Counter(r['status'] for r in native))}", flush=True)
        for source, (_, url, _) in ARCHIVES.items():
            cache = out / "archives" / f"{source}.html.gz"
            try:
                raw = gzip.decompress(cache.read_bytes()) if cache.exists() else await fetch(client, semaphore, url)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(gzip.compress(raw, mtime=0))
                entries = await naboj_papers(client, semaphore, out, raw.decode()) if source == "physics_naboj_archive" else archive_records(source, raw.decode())
                for entry in entries:
                    entry["archive_sha256"] = digest(raw)
                    entry["license_status"] = "Redistribution and model-training permission unverified; private source staging"
                    entry["release_state"] = "source_staging"
                    if entry.get("problem_id") in evaluation_ids:
                        entry["status"] = "excluded_evaluation_identity"
                    elif not entry.get("problem_url") or not entry.get("solution_url"):
                        entry["status"] = "missing_problem_or_solution"
                    else:
                        entry["status"] = "source_review_pending"
                    if "problem_id" in entry:
                        originals.append(entry)
                    else:
                        papers.append(entry)
                print(f"{source}: {len(entries)} source records", flush=True)
            except (httpx.HTTPError, ValueError) as exc:
                errors.append({"stage": "archive", "source": source, "url": url, "error": str(exc)})
        urls = sorted({r[k] for r in [*papers, *originals] if r["status"] == "source_review_pending"
                       for k in ("problem_url", "solution_url") if r.get(k)})
        if args.max_documents is not None:
            urls = urls[:args.max_documents]

        async def one_pdf(url):
            try:
                doc = await collect_pdf(client, semaphore, out, url)
                docs.append(doc)
                if len(docs) % 25 == 0:
                    print(f"Extracted {len(docs)}/{len(urls)} PDFs", flush=True)
            except (httpx.HTTPError, ValueError, fitz.FileDataError) as exc:
                errors.append({"stage": "pdf", "url": url, "error": str(exc)})

        await asyncio.gather(*(one_pdf(url) for url in urls))
    by_url = {d["url"]: d for d in docs}
    for row in [*papers, *originals]:
        if row["status"] != "source_review_pending":
            continue
        row["documents"] = {k: by_url[row[k]] for k in ("problem_url", "solution_url") if row.get(k) in by_url}
        if len(row["documents"]) != 2:
            row["status"] = "download_pending"
            continue
        row["status"] = "pdf_transcription_review"
        if "paper_id" in row:
            doc = read_document(out, row["documents"]["problem_url"]["document_key"])
            found = paper_problems(row, doc)
            for item in found:
                item["status"] = "excluded_evaluation_identity" if item["problem_id"] in evaluation_ids else "pdf_transcription_review"
            originals.extend(found)
            row["provisional_original_problem_count"] = len(found)
    native.sort(key=lambda r: r["problem_id"])
    unique = {}
    for row in originals:
        previous = unique.get(row["problem_id"])
        if previous is None or previous.get("boundary_detection"):
            unique[row["problem_id"]] = row
        elif row.get("boundary_detection"):
            previous.setdefault("alternate_paper_sources", []).append({
                "source": row["source"], "problem_url": row["problem_url"], "solution_url": row["solution_url"]})
    originals = list(unique.values())
    for row in originals:
        row["already_curated"] = row["problem_id"] in used_identities
        if row["already_curated"] and row["status"] != "excluded_evaluation_identity":
            row["status"] = "already_curated"
    write_jsonl(out / "native.jsonl", native)
    write_jsonl(out / "papers.jsonl", papers)
    write_jsonl(out / "originals.jsonl", sorted(originals, key=lambda r: r["problem_id"]))
    write_json(out / "documents.json", docs)
    write_json(out / "errors.json", errors)
    write_json(out / "native_preflight.json", {"records": [r for r in native if r["status"] == "native_preflight_candidate"]})
    result = summary(out, native, papers, originals, docs, errors)
    result.update(prepare_native_queue(out, native, evaluation_manifest["snapshot_sha256"]))
    result.update(pilot_review_queue(out))
    write_json(out / "summary.json", result)
    if result.get("queued_for_curation"):
        with (out / "REPORT.md").open("a") as report:
            report.write(f"\n## Screened source queue\n\n{result['queued_for_curation']} unused native problems are queued for curation; "
                         f"{result['native_overlap_review']} are held for overlap review. "
                         "The runnable manifest is `next_native_batch.json`. No paid generation has started.\n"
                         f"\nExisting pilots: {result['existing_pilot_originals']} originals, {result['existing_model_checked']} model-checked, "
                         f"{result['existing_held_for_review']} held in `pilot_final_review.jsonl`.\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--offline", action="store_true", help="Rebuild inventories from saved sources without network access")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    asyncio.run(run(args))
