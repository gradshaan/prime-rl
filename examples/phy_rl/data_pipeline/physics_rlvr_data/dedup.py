from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

from .io import ensure_parent
from .schema import FinalItem


def canonical_text(text: str) -> str:
    lowered = text.lower()
    normalized = re.sub(r"[^\w.+\-=<>/]", " ", lowered)
    return re.sub(r"\s+", " ", normalized).strip()


def problem_hash(item: FinalItem) -> str:
    payload = "\n".join(
        [
            canonical_text(item.problem_text),
            canonical_text(item.shared_context),
            canonical_text(item.question),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_answers(item: FinalItem) -> str:
    parts = []
    for answer in item.answers:
        parts.append(
            "|".join(
                [
                    canonical_text(answer.output_label or ""),
                    canonical_text(answer.value),
                    canonical_text(answer.unit or ""),
                    answer.answer_type,
                    answer.verifier,
                    str(answer.atol),
                    str(answer.rtol if answer.rtol is not None else answer.tolerance),
                    "\n".join(sorted(canonical_text(value) for value in answer.equivalent_forms)),
                    "\n".join(sorted(canonical_text(value) for value in answer.assumptions)),
                ]
            )
        )
    return "\n".join(sorted(parts))


def deduplicate(items: list[FinalItem], report_path: Path) -> list[FinalItem]:
    ensure_parent(report_path)
    seen: dict[str, FinalItem] = {}
    kept: list[FinalItem] = []
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["duplicate_problem_id", "kept_problem_id", "hash", "answers_match"],
        )
        writer.writeheader()
        for item in items:
            digest = problem_hash(item)
            if digest in seen:
                answers_match = canonical_answers(item) == canonical_answers(seen[digest])
                writer.writerow(
                    {
                        "duplicate_problem_id": item.problem_id,
                        "kept_problem_id": seen[digest].problem_id,
                        "hash": digest,
                        "answers_match": answers_match,
                    }
                )
                if not answers_match:
                    raise ValueError(
                        f"duplicate prompt has conflicting answers: {seen[digest].problem_id} and {item.problem_id}"
                    )
                continue
            seen[digest] = item
            kept.append(item)
    return kept
