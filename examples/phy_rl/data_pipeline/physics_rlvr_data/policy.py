from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from physics_rlvr_common.policy import TRAIN_YEAR_MAX
from physics_rlvr_common.policy import validate_training_policy as _validate_training_policy

Split = Literal["train", "dev", "frozen_test"]

def choose_split(
    *,
    source: str,
    competition: str,
    year: int | None,
    requested_split: Split | None = None,
) -> Split:
    split = requested_split or "train"
    validate_training_policy(source=source, competition=competition, year=year, split=split)
    return split


def validate_training_policy(
    *,
    source: str,
    competition: str,
    year: int | None,
    split: str,
    source_split: str | None = None,
    source_revision: str | None = None,
) -> None:
    _validate_training_policy(
        source=source,
        competition=competition,
        year=year,
        split=split,
        source_split=source_split,
        source_revision=source_revision,
    )
    source_config = _source_registry().get("source", {}).get(source)
    if source_config is None:
        raise ValueError(f"{source} is not listed in the source registry")
    if year is not None and year > int(source_config.get("include_year_max", TRAIN_YEAR_MAX)):
        raise ValueError(f"{source} records after its training cutoff are blocked")
    if year is not None and year < int(source_config.get("include_year_min", 0)):
        raise ValueError(f"{source} records before its curated year range are blocked")


def get_source_config(source: str) -> dict:
    source_config = _source_registry().get("source", {}).get(source)
    if source_config is None:
        raise ValueError(f"{source} is not listed in the source registry")
    return source_config


@lru_cache(maxsize=1)
def _source_registry() -> dict:
    path = Path(__file__).parents[1] / "configs/sources.toml"
    with path.open("rb") as file:
        return tomllib.load(file)
