from __future__ import annotations

import re
import tomllib
from typing import Any
from functools import lru_cache
from pathlib import Path


def normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


@lru_cache(maxsize=1)
def _source_registry() -> dict:
    path = Path(__file__).parents[2] / "data_pipeline/configs/sources.toml"
    with path.open("rb") as file:
        return tomllib.load(file)


@lru_cache(maxsize=1)
def _filter_config() -> dict:
    path = Path(__file__).parents[2] / "data_pipeline/configs/filters.toml"
    with path.open("rb") as file:
        return tomllib.load(file)


TRAIN_YEAR_MAX = int(_filter_config()["years"]["train_max_year"])
APPROVED_TRAIN_SOURCES = frozenset(_source_registry()["source"])
BLOCKED_TRAIN_SOURCES = frozenset(
    normalize_identifier(value) for value in _filter_config()["blocked_training"]["sources"]
)
BLOCKED_TRAIN_COMPETITIONS = frozenset(
    normalize_identifier(value)
    for value in [
        *_filter_config()["blocked_training"]["competitions"],
        *_filter_config()["blocked_training"]["sources"],
    ]
)

YEAR_REQUIRED_COMPETITIONS = {
    "ipho",
    "apho",
    "eupho",
    "nbpho",
    "wopho",
    "usapho",
    "usapho_training_camp",
    "inpho",
    "opho",
    "physics_cup",
    "physics_naboj",
    "australian_physics_olympiad",
    "estonian_physics_olympiad",
    "knzhou_handouts",
    "zhou_handouts",
}

YEAR_REQUIRED_SOURCES = {
    "ipho_archive",
    "ipho_olimpicos",
    "ipho_open_train",
    "apho_archive",
    "eupho_archive",
    "nbpho_archive",
    "nbpho_olimpicos",
    "wopho_archive",
    "usapho_archive",
    "usapho_training_camp",
    "inpho_archive",
    "opho_archive",
    "physics_cup_archive",
    "physics_naboj_archive",
    "australian_physics_olympiad",
    "czech_physics_olympiad",
    "knzhou_handouts",
    "zhou_handouts",
    "kalda_latex",
    "estonian_physics_olympiad",
}

SOURCE_SPLIT_REQUIRED = {"physics_training_release": "train"}


def validate_source_provenance(row: dict[str, Any]) -> None:
    """Reject forbidden upstream repositories even if the source label was changed."""
    repositories = _filter_config()["blocked_training"].get("dataset_repositories", [])

    def check(value: Any) -> None:
        if isinstance(value, str):
            identifier = normalize_identifier(value)
            for repository in repositories:
                blocked = normalize_identifier(repository)
                if re.search(r"(?:^|_)" + re.escape(blocked) + r"(?:_|$)", identifier):
                    raise ValueError(f"{repository} is blocked in source provenance")
        elif isinstance(value, dict):
            for nested in value.values():
                check(nested)
        elif isinstance(value, list):
            for nested in value:
                check(nested)

    for field in ["source", "dataset_repository", "upstream_dataset", "dataset_url", "source_url", "provenance", "metadata"]:
        check(row.get(field))
    evidence = row.get("source_evidence")
    if isinstance(evidence, dict):
        validate_source_provenance(evidence)


def validate_release_state(row: dict[str, Any]) -> None:
    """Prevent explicitly staged or failed review records from entering training."""
    problem_id = row.get("problem_id", "<unknown>")
    validate_source_provenance(row)
    if "release_status" in row and row["release_status"] != "ready":
        raise ValueError(f"{problem_id} is not released for training: {row['release_status']}")
    if "status" in row and row["status"] not in {"accepted", "model_checked"}:
        raise ValueError(f"{problem_id} has unresolved review status: {row['status']}")
    if "checks" in row and (not row["checks"] or not all(value is True for value in row["checks"].values())):
        raise ValueError(f"{problem_id} has failed validation checks")


def validate_training_policy(
    *,
    source: str,
    competition: str,
    year: int | None,
    split: str,
    source_split: str | None = None,
    source_revision: str | None = None,
) -> None:
    source_id = normalize_identifier(source)
    competition_id = normalize_identifier(competition)
    if split not in {"train", "dev"}:
        raise ValueError(f"dataset rows cannot use split {split!r}; evaluation records stay outside the dataset")

    for identifier, blocked_values, description in (
        (source_id, BLOCKED_TRAIN_SOURCES, source),
        (competition_id, BLOCKED_TRAIN_COMPETITIONS, competition),
    ):
        if identifier in blocked_values or any(identifier.startswith(blocked + "_") for blocked in blocked_values):
            raise ValueError(f"{description} is blocked from the training corpus")
    source_config = _source_registry()["source"].get(source_id)
    if source_config is None:
        raise ValueError(f"{source} is not an approved training source")
    if source_id not in APPROVED_TRAIN_SOURCES:
        raise ValueError(f"{source} is not an approved training source")
    accepted_splits = source_config.get("accepted_source_splits")
    if accepted_splits and source_split not in accepted_splits:
        raise ValueError(f"{source} requires an admitted source split")
    accepted_revisions = source_config.get("accepted_revisions")
    if accepted_revisions and source_revision not in accepted_revisions:
        raise ValueError(f"{source} requires a pinned admitted source revision")
    if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
        raise ValueError("year must be an integer")
    if year is None and (source_id in YEAR_REQUIRED_SOURCES or competition_id in YEAR_REQUIRED_COMPETITIONS):
        raise ValueError(f"{competition} rows need an explicit year")
    if year is not None:
        source_min_year = int(source_config.get("include_year_min", 0))
        source_max_year = int(source_config.get("include_year_max", TRAIN_YEAR_MAX))
        if year < source_min_year:
            raise ValueError(f"year {year} is outside the approved range for {source}")
        if year > min(TRAIN_YEAR_MAX, source_max_year):
            raise ValueError(f"year {year} is blocked from training")

    required_source_split = SOURCE_SPLIT_REQUIRED.get(source_id)
    if required_source_split is not None and source_split != required_source_split:
        raise ValueError(f"{source} can only use source split {required_source_split!r}")
    if source_id == "physics_training_release" and (
        not isinstance(source_revision, str) or re.fullmatch(r"[0-9a-f]{64}", source_revision) is None
    ):
        raise ValueError(f"{source} requires the SHA-256 of the staged training artifact")
