"""Extract named olympiad-preparation problems and human-written solutions."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from physics_rlvr_data.verifiers import extract_boxed

SAVCHENKO_REVISION = "148ff8c47b99b0edf5e00d1a79d3a1e71a6e6ed4"
KALDA_REVISION = "598e2642febcf5f3aa9602fb8446372970f1b5f7"
FIGURE = re.compile(r"\\(?:includegraphics|begin\{(?:asy|tikzpicture|circuitikz))|\b(?:figure|diagram|graph|shown|photograph|picture)\b", re.I)
QUALITATIVE = re.compile(r"\b(?:draw|sketch|plot|prove|explain|construct a graph)\b", re.I)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


class SolutionPage(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.phase = None
        self.heading = None
        self.heading_text = []
        self.suppressed = []
        self.parts = {"question": [], "solution": [], "answer": []}
        self.images = {"question": [], "solution": [], "answer": []}

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"head", "header", "footer", "script", "style", "iframe"}:
            self.suppressed.append(tag)
        if self.suppressed:
            return
        if tag in {"h3", "h4"}:
            self.heading, self.heading_text = tag, []
        elif tag == "img" and self.phase:
            self.images[self.phase].append(dict(attrs).get("src", ""))
        elif tag in {"p", "br", "li", "div"} and self.phase:
            self.parts[self.phase].append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.suppressed:
            if tag == self.suppressed[-1]:
                self.suppressed.pop()
            return
        if tag == self.heading:
            heading = "".join(self.heading_text).strip()
            title, _, inline = heading.partition(":")
            phases = {"statement": "question", "problem": "question", "solution": "solution", "answer": "answer",
                      "условие": "question", "решение": "solution", "ответ": "answer"}
            if title.strip().casefold() in phases:
                self.phase = phases[title.strip().casefold()]
                if inline.strip():
                    self.parts[self.phase].append(inline.strip())
            self.heading = None
        elif tag in {"p", "li", "div"} and self.phase:
            self.parts[self.phase].append("\n")

    def handle_data(self, data: str) -> None:
        if self.suppressed:
            return
        if self.heading:
            self.heading_text.append(data)
        elif self.phase:
            self.parts[self.phase].append(data)

    def text(self, phase: str) -> str:
        text = "".join(self.parts[phase])
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def textbook_statements(text: str) -> dict[str, str]:
    """Keep nested subparts inside one original numbered problem."""
    token = re.compile(r"\\begin\{enumerate\}(?:\[([^\]]*)\])?|\\end\{enumerate\}|\\item\b(?:\[[^\]]*\])?")
    depth, number, start, section = 0, 0, None, None
    result = {}
    for match in token.finditer(text):
        marker = match[0]
        if marker.startswith(r"\begin"):
            if depth == 0:
                label = re.search(r"label=(\d+\.\d+)\.\\arabic", match[1] or "")
                section = label[1] if label else None
                number = 0
            depth += 1
        elif marker.startswith(r"\end"):
            if depth == 1 and start is not None and section:
                result[f"{section}.{number}"] = text[start:match.start()].strip()
                start = None
            depth -= 1
            if depth == 0:
                section = None
        elif depth == 1 and section:
            if start is not None:
                result[f"{section}.{number}"] = text[start:match.start()].strip()
            number += 1
            start = match.end()
    return result


def record(source: str, sid: str, question: str, solution: str, answer: str, provenance: dict) -> dict:
    boxes = extract_boxed(solution)
    reasons = []
    if len(question.strip()) < 50:
        reasons.append("short_or_missing_statement")
    if len(solution.strip()) < 80:
        reasons.append("short_or_missing_solution")
    if FIGURE.search(question) or provenance.get("statement_images"):
        reasons.append("statement_diagram_review")
    if QUALITATIVE.search(question):
        reasons.append("requested_non_closed_form_output_review")
    if not answer and not boxes:
        reasons.append("final_target_extraction_review")
    return {"problem_id": source + "-" + sid, "source": source, "source_id": sid,
            "question": question, "reference_solution": solution,
            "reference_answer": answer, "boxed_answer_candidates": boxes,
            "question_sha256": sha(question), "solution_sha256": sha(solution), "answer_sha256": sha(answer),
            "language": "en", "source_split": "training_material", "source_difficulty": None,
            "difficulty": None, "difficulty_status": "not_measured", "training_ready": False,
            "required_outputs": [], "release_status": "held", "status": "source_candidate",
            "pending_checks": ["independent_physics_audit", "complete_per_output_contract", "deterministic_verifier_checks",
                               "translation_review", "underlying_contest_source_and_year_if_any"],
            "hold_reasons": reasons, "provenance": provenance}


def savchenko(output: Path) -> tuple[list[dict], list[dict]]:
    base = output / "native/savchenko-physics"
    text = (base / "src/database/main.tex").read_text()
    statements = textbook_statements(text)
    indexed = [{"source": "savchenko_solutions", "source_id": sid, "question": question,
                "source_revision": SAVCHENKO_REVISION, "solution_paired": False,
                "release_status": "held", "training_ready": False} for sid, question in statements.items()]
    pairs = []
    english_paths = sorted((base / "en").glob("*/index.html"))
    english_ids = {path.parent.name for path in english_paths}
    russian_paths = [path for path in sorted(base.glob("*/*/index.html"))
                     if re.fullmatch(r"\d+", path.parent.parent.name) and path.parent.name not in english_ids]
    for path in [*english_paths, *russian_paths]:
        sid = path.parent.name
        if not re.fullmatch(r"\d+\.\d+\.\d+", sid):
            continue
        parser = SolutionPage()
        parser.feed(path.read_text())
        question, solution, answer = (parser.text(phase) for phase in ["question", "solution", "answer"])
        # The author's name sits outside the solution body on these pages.
        solution = re.sub(r"\n\s*Aliaksandr Melnichenka\s*$", "", solution).strip()
        answer = re.sub(r"\n\s*Aliaksandr Melnichenka\s*$", "", answer).strip()
        rel = path.relative_to(base).as_posix()
        original = statements.get(sid)
        russian = path in russian_paths
        native_question = question
        if russian:
            question = original or ""
        row = record("savchenko_solutions", sid, question, solution, answer,
                     {"repository": "savchenko-physics/savchenko-physics.github.io", "revision": SAVCHENKO_REVISION,
                      "file": rel, "source_url": f"https://savchenko-physics.github.io/{path.parent.relative_to(base).as_posix()}/",
                      "original_book": "O. Ya. Savchenko, Problems in Physics", "problem_number": sid,
                      "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "statement_images": parser.images["question"], "solution_images": parser.images["solution"],
                      "content_origin": "community_worked_solution", "license_status": "private staging; commercial use needs author permission"})
        row["alternate_native_statement"] = original
        row["reference_solution_language"] = "ru" if russian else "en"
        if russian:
            row["original_language_statement"] = native_question
            row["provenance"]["statement_file"] = "src/database/main.tex"
            row["provenance"]["statement_file_sha256"] = sha(text)
            row["pending_checks"].append("bilingual_statement_answer_alignment")
        row["source_difficulty"] = "author_starred" if (re.search(r"^\$?\s*" + re.escape(sid) + r"[^$\n]{0,12}\*", native_question) or (original and r"\hard" in original)) else None
        if original is None:
            row["hold_reasons"].append("native_statement_identity_missing")
        if not row["reference_answer"] and row["boxed_answer_candidates"]:
            row["reference_answer"] = "\n".join(row["boxed_answer_candidates"])
            row["answer_sha256"] = sha(row["reference_answer"])
            row["answer_origin"] = "balanced_box_candidates_not_yet_validated_as_complete"
        else:
            row["answer_origin"] = "explicit_source_answer_section" if answer else "not_extracted"
        pairs.append(row)
    paired = {row["source_id"] for row in pairs}
    for row in indexed:
        row["solution_paired"] = row["source_id"] in paired
    return pairs, indexed


def kalda(output: Path) -> tuple[list[dict], list[dict]]:
    base = output / "native/physoly"
    pairs, unpaired = [], []
    cm = base / "KaldaCM-ENG/main.tex"
    text = cm.read_text()
    questions = re.findall(r"\\hypertarget\{P(\d+)\}\{\}\s*\\begin\{solution\}\{[^{}]*\}(.*?)\\end\{solution\}", text, re.S)
    answers = re.findall(r"\\begin\{answer\}\{[^{}]*\}\s*%\s*sol\s*(\d+)\s*\n(.*?)\\end\{answer\}", text, re.S)
    answers = dict(answers)
    for sid, question in questions:
        if sid not in answers:
            unpaired.append({"source_id": "celestial-" + sid, "reason": "no_matching_solution"})
            continue
        solution = answers[sid].strip()
        row = record("kalda_latex", "celestial-" + sid, question.strip(), solution, "",
                     {"repository": "physoly/physoly-latex", "revision": KALDA_REVISION,
                      "file": "KaldaCM-ENG/main.tex", "problem_number": sid,
                      "source_url": f"https://github.com/physoly/physoly-latex/blob/{KALDA_REVISION}/KaldaCM-ENG/main.tex",
                      "content_origin": "community_translation_and_solution", "source_year_upper_bound": 2023,
                      "source_file_sha256": sha(text), "license": "CC-BY-SA-4.0 text; GPL-3.0 source files"})
        pairs.append(row)
    for path in sorted((base / "Kalda Electricity and Magnetism Questions").glob("S*.tex")):
        text = path.read_text()
        for sid, question in re.findall(r"\\hypertarget\{P(\d+)\}\{\}\s*\\begin\{solution\}\{[^{}]*\}(.*?)\\end\{solution\}", text, re.S):
            exact = re.search(r"(?<!Similar to )\(Kalda Circuits P(\d+)\)", question)
            if not exact:
                continue
            number = exact[1]
            solution_file = base / "Kalda Circuits Solutions Manual" / (number.zfill(2) + ".tex")
            if not solution_file.exists():
                unpaired.append({"source_id": "circuits-" + number, "reason": "no_solution_file"})
                continue
            solution = solution_file.read_text().strip()
            if len(solution) < 80:
                unpaired.append({"source_id": "circuits-" + number, "reason": "empty_solution_placeholder"})
                continue
            question = re.sub(r"\s*\(Kalda Circuits P\d+\)", "", question).strip()
            row = record("kalda_latex", "circuits-" + number, question, solution, "",
                         {"repository": "physoly/physoly-latex", "revision": KALDA_REVISION,
                          "file": path.relative_to(base).as_posix(), "statement_number": path.stem + "-" + sid,
                          "solution_file": solution_file.relative_to(base).as_posix(), "problem_number": number,
                          "source_url": f"https://github.com/physoly/physoly-latex/tree/{KALDA_REVISION}",
                          "content_origin": "community_translation_and_solution", "source_year_upper_bound": 2023,
                          "source_file_sha256": sha(text), "solution_file_sha256": sha(solution),
                          "license": "CC-BY-SA-4.0 text; GPL-3.0 source files"})
            row["reference_answer"] = "\n".join(row["boxed_answer_candidates"])
            row["answer_sha256"] = sha(row["reference_answer"])
            row["answer_origin"] = "balanced_box_candidates_not_yet_validated_as_complete"
            pairs.append(row)
    unique = {row["problem_id"]: row for row in pairs}
    return list(unique.values()), unpaired


def build(output: Path, data: Path) -> None:
    pairs, indexed = savchenko(output)
    more, unpaired = kalda(output)
    pairs.extend(more)
    existing = set()
    for path in [data / "training_sources/candidates_5000.jsonl", *sorted(data.glob("pilot_*/pilot.json"))]:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.open()]
        else:
            rows = json.loads(path.read_text())["records"]
        existing.update(sha(re.sub(r"\s+", " ", row["question"]).strip()) for row in rows)
    candidates, review = [], []
    for row in pairs:
        if sha(re.sub(r"\s+", " ", row["question"]).strip()) in existing:
            row["hold_reasons"].append("already_in_collected_corpus")
        (review if row["hold_reasons"] else candidates).append(row)
    write_jsonl(output / "native_pairs.jsonl", pairs)
    write_jsonl(output / "native_review.jsonl.gz", review)
    write_jsonl(output / "unpaired_statements.jsonl.gz", [row for row in indexed if not row["solution_paired"]])
    (output / "preflight.json").write_text(json.dumps({"records": candidates}, ensure_ascii=False))
    summary = {"source_pairs": len(pairs), "pairs_by_source": dict(collections.Counter(row["source"] for row in pairs)),
               "reference_solution_languages": dict(collections.Counter(row.get("reference_solution_language", "en") for row in pairs)),
               "native_statement_count": len(indexed), "unpaired_native_statements": sum(not row["solution_paired"] for row in indexed),
               "native_preflight_candidates": len(candidates), "source_review": len(review),
               "hold_reasons": dict(collections.Counter(reason for row in review for reason in row["hold_reasons"])),
               "author_starred_pairs": sum(row.get("source_difficulty") == "author_starred" for row in pairs),
               "missing_kalda_solutions": unpaired, "api_cost_usd": 0, "training_release_count": 0}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=Path("examples/phy_rl/data_pipeline/data"))
    args = parser.parse_args()
    build(args.output, args.data)


if __name__ == "__main__":
    main()
