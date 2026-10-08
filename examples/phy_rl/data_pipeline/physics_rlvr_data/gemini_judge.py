from __future__ import annotations

import json
import re
from typing import Any

from .answer_extraction import answer_proposals_to_review, normalize_subproblem_id
from .gemini_extract import DEFAULT_MODEL, generate_gemini_content
from .schema import FinalItem, to_dict

JUDGE_PROMPT = """You are curating physics RLVR training rows from OCR text.

Return only JSON with this schema:
{
  "items": [
    {
      "subproblem_id": "A.1 or null",
      "keep": true,
      "reason": "short audit note",
      "shared_context": "context needed to answer the question",
      "question": "single full question",
      "official_solution": "solution text supporting the answer",
      "answers": [
        {
          "label": "unique output label such as acceleration or tension",
          "value": "verifiable answer value",
          "unit": null,
          "answer_type": "numeric|symbolic",
          "atol": 0.005,
          "rtol": 0.000001,
          "verifier": "numeric|sympy",
          "equivalent_forms": [],
          "subproblem_id": "same label or null"
        }
      ]
    }
  ]
}

Keep only rows with a complete question and answer pair that can be checked by a deterministic verifier.
Reject proof-only, explanation-only, drawing-only, qualitative, ambiguous, or diagram-dependent prompts.
Use the candidate answers as evidence, but correct obvious OCR formatting when needed.
Enumerate every requested output; each answer must have a unique label.
Do not invent answers that are not supported by the solution text."""

AUDIT_PROMPT = """Independently audit extracted physics RLVR answers.

For every candidate item, solve the question independently and compare your result
with the candidate answers and the provided official solution. Confirm that every
requested output appears exactly once, values and units are correct, and the
question is self-contained without missing diagrams. Return only JSON:
{"items":[{"subproblem_id":"same id or null","approved":true,"reason":"short audit note"}]}.
Approve only complete answer sets supported by both the question and solution.
If an output is missing, duplicated, mislabeled, or numerically/symbolically wrong,
set approved=false. Do not rewrite candidate answers."""


def judge_subproblems_with_gemini(
    parent: FinalItem,
    candidate_rows: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    payload = {
        "parent": {
            "problem_id": parent.problem_id,
            "source": parent.source,
            "year": parent.year,
            "problem_number": parent.problem_number,
            "problem_text": parent.problem_text,
            "official_solution": parent.official_solution,
        },
        "candidate_rows": candidate_rows,
    }
    text = generate_gemini_content(
        model_name,
        [
            {"text": JUDGE_PROMPT},
            {"text": json.dumps(payload, ensure_ascii=True)},
        ],
        max_output_tokens=8192,
        temperature=0.0,
    )
    items = _json_object(text).get("items", [])
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("judge response must contain an items list of objects")
    return items


def audit_judged_rows_with_gemini(
    parent: FinalItem,
    judged_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    *,
    model_name: str,
) -> dict[str | None, dict[str, Any]]:
    payload = {
        "parent": {
            "problem_id": parent.problem_id,
            "problem_text": parent.problem_text,
            "official_solution": parent.official_solution,
        },
        "candidate_items": judged_rows,
        "all_requested_subproblems": [
            {"subproblem_id": row.get("subproblem_id"), "question": row.get("question")}
            for row in expected_rows
        ],
    }
    text = generate_gemini_content(
        model_name,
        [
            {"text": AUDIT_PROMPT},
            {"text": json.dumps(payload, ensure_ascii=True)},
        ],
        max_output_tokens=4096,
        temperature=0.0,
    )
    items = _json_object(text).get("items", [])
    if not isinstance(items, list):
        raise ValueError("audit response must contain an items list")

    audited: dict[str | None, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("approved"), bool):
            raise ValueError("audit response has a malformed item")
        subproblem_id = item.get("subproblem_id")
        if isinstance(subproblem_id, str):
            subproblem_id = normalize_subproblem_id(subproblem_id)
        if subproblem_id in audited:
            raise ValueError("audit response repeats a subproblem id")
        audited[subproblem_id] = item

    expected_id_list = [normalize_subproblem_id(row["subproblem_id"]) if row.get("subproblem_id") else None for row in expected_rows]
    expected_ids = set(expected_id_list)
    if len(expected_ids) != len(expected_id_list):
        raise ValueError("requested subproblems have duplicate ids")
    if set(audited) != expected_ids:
        raise ValueError("audit response does not cover every kept subproblem exactly once")
    return audited


def candidate_row_for_judge(item: FinalItem, proposals: list[Any]) -> dict[str, Any]:
    return {
        "subproblem_id": item.subproblem_id,
        "shared_context": item.shared_context,
        "question": item.question,
        "official_solution": item.official_solution,
        "heuristic_answers": answer_proposals_to_review(proposals, limit=12),
        "current_item": to_dict(item),
    }


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    return json.loads(stripped)
