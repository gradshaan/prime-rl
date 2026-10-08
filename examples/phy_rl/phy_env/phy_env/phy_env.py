"""English, text-only physics RLVR environment using the v3 data contract."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from physics_rlvr_common import (
    SYSTEM_PROMPT,
    task_prompt,
    validate_answer,
    validate_release_state,
    validate_training_policy,
    verify_prediction,
)
from physics_rlvr_common.verifier import answer_from_dict

DEFAULT_TRAIN_PATH = Path(__file__).parents[2] / "data_pipeline/data/final/v3/train.jsonl"


def correctness_reward(completion: list[dict[str, Any]], answer: str, **kwargs: Any) -> float:
    assistant_text = next(
        (message.get("content", "") for message in reversed(completion) if message.get("role") == "assistant"),
        "",
    )
    if not isinstance(assistant_text, str):
        return 0.0
    expected = [answer_from_dict(raw) for raw in json.loads(answer)]
    if any(validate_answer(item, require_label=True) for item in expected):
        raise ValueError("environment received malformed verifier targets")
    return verify_prediction(assistant_text, expected)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is invalid JSONL") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                rows.append(row)
    return rows


def _validated_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    sources: list[str] | None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        validate_release_state(row)
        if row.get("dataset_version") != "physics_rlvr_v3":
            raise ValueError("environment accepts only physics_rlvr_v3 records")
        answers = row.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"{row.get('problem_id', '<unknown>')} has no answers")
        parsed_answers = [answer_from_dict(answer) for answer in answers]
        if any(validate_answer(answer, require_label=True) for answer in parsed_answers):
            raise ValueError(f"{row.get('problem_id', '<unknown>')} has malformed verifier targets")
        source = row["source"]
        validate_training_policy(
            source=source,
            competition=row["competition"],
            year=row.get("year"),
            split=row["split"],
            source_split=row.get("provenance", {}).get("source_split"),
            source_revision=row.get("provenance", {}).get("source_revision"),
        )
        if sources and source not in sources:
            continue
        if row["split"] != split:
            raise ValueError(f"{row['problem_id']} is in the {row['split']} split, expected {split}")
        labels = [answer.output_label for answer in parsed_answers]
        if any(not label for label in labels) or len(labels) != len(set(labels)):
            raise ValueError(f"{row['problem_id']} must have unique labels for every answer")
        question = "\n\n".join(part for part in [row.get("shared_context", ""), row["question"]] if part).strip()
        if not question or "<image_start>" in question or "[problem_image" in question.lower():
            raise ValueError(f"{row['problem_id']} is not a complete text-only question")
        if "the solution is:" in question.casefold():
            raise ValueError(f"{row['problem_id']} includes its solution in the prompt")
        selected.append(row)
    return selected


def _to_verifiers_row(row: dict[str, Any]) -> dict[str, str]:
    answers = [
        {
            "label": answer.get("label") or answer.get("output_label"),
            "value": answer["value"],
            "unit": answer.get("unit"),
            "answer_type": answer["answer_type"],
            "verifier": answer["verifier"],
            "atol": answer.get("atol"),
            "rtol": answer.get("rtol", answer.get("tolerance")),
            "equivalent_forms": answer.get("equivalent_forms", []),
            "assumptions": answer.get("assumptions", []),
        }
        for answer in row["answers"]
    ]
    prompt = task_prompt(row.get("shared_context", ""), row["question"], [answer["label"] for answer in answers])
    return {"question": prompt, "answer": json.dumps(answers)}


def load_environment(
    *,
    dataset_path: str | Path | None = None,
    dev_path: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_revision: str | None = None,
    hf_token: str | None = None,
    sources: str | list[str] | None = None,
    num_train: int | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> vf.Environment:
    if dataset_path is not None and dataset_name is not None:
        raise ValueError("set either dataset_path or dataset_name, not both")
    if dataset_path is None and dataset_name is None:
        dataset_path = DEFAULT_TRAIN_PATH
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if isinstance(sources, str):
        sources = [source.strip() for source in sources.split(",") if source.strip()]
    allowed_sources = set(sources) if sources else None
    if num_train is not None and num_train < 0:
        raise ValueError("num_train must be non-negative")

    if dataset_path is not None:
        train_file = Path(dataset_path)
        metadata_path = train_file.parent / "metadata.json"
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("dataset_version") != "physics_rlvr_v3":
            raise ValueError(f"{metadata_path} is not a v3 dataset manifest")
        _validate_local_artifact(train_file, metadata, "train")
        train_rows = _validated_rows(_read_jsonl(train_file), split="train", sources=allowed_sources)
    else:
        if dataset_revision is None or re.fullmatch(r"[0-9a-f]{40}", dataset_revision) is None:
            raise ValueError("dataset_revision must be an exact 40-character Hugging Face commit hash")
        train_dataset = load_dataset(
            dataset_name,
            revision=dataset_revision,
            split="train",
            token=token,
        )
        train_rows = _validated_rows(list(train_dataset), split="train", sources=allowed_sources)

    random.Random(seed).shuffle(train_rows)
    if num_train is not None:
        train_rows = train_rows[:num_train]
    if not train_rows:
        raise ValueError("no training rows remain after policy and source filtering")

    dev_dataset = None
    if dev_path is not None:
        dev_file = Path(dev_path)
        if dataset_path is not None and dev_file.parent != Path(dataset_path).parent:
            raise ValueError("train and development data must come from the same v3 artifact directory")
        dev_metadata_path = dev_file.parent / "metadata.json"
        if dataset_path is None:
            with dev_metadata_path.open(encoding="utf-8") as file:
                dev_metadata = json.load(file)
            if dev_metadata.get("dataset_version") != "physics_rlvr_v3":
                raise ValueError(f"{dev_metadata_path} is not a v3 dataset manifest")
        else:
            dev_metadata = metadata
        _validate_local_artifact(dev_file, dev_metadata, "dev")
        dev_rows = _validated_rows(_read_jsonl(dev_file), split="dev", sources=allowed_sources)
        if not dev_rows:
            raise ValueError("no development rows remain after policy and source filtering")
        dev_dataset = Dataset.from_list([_to_verifiers_row(row) for row in dev_rows])

    return vf.SingleTurnEnv(
        dataset=Dataset.from_list([_to_verifiers_row(row) for row in train_rows]),
        eval_dataset=dev_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=vf.Rubric(funcs=[correctness_reward], weights=[1.0]),
    )


def _validate_local_artifact(path: Path, metadata: dict[str, Any], split: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_count = metadata.get("counts", {}).get(split)
    rows = _read_jsonl(path)
    if not isinstance(expected_count, int) or expected_count != len(rows):
        raise ValueError(f"{path} row count does not match its metadata")
    expected_hash = metadata.get("file_sha256", {}).get(path.name)
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_hash is None or expected_hash != actual_hash:
        raise ValueError(f"{path} checksum does not match its metadata")
