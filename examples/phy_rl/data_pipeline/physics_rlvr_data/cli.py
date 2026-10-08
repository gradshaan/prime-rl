from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_config import BaseConfig, cli

from .pipeline import (
    DEFAULT_DATA_DIR,
    apply_answer_key,
    build_candidates,
    build_rlvr_subproblems_file,
    crawl_pdf_manifest,
    dedup_file,
    download_manifest,
    export_final,
    extract_answer_draft_file,
    extract_answer_key_file,
    extract_manifest,
    filter_candidates,
    import_structured_jsonl,
    init_layout,
    manifest_from_local_pdfs,
    write_validation_report,
)


class PipelineConfig(BaseConfig):
    command: Literal[
        "init",
        "manifest-local",
        "crawl",
        "download",
        "extract",
        "build-candidates",
        "import-structured",
        "filter-candidates",
        "apply-answer-key",
        "extract-answer-key",
        "extract-answer-draft",
        "build-rlvr-subproblems",
        "dedup",
        "export-final",
        "validate-final",
    ]
    data_dir: Path = DEFAULT_DATA_DIR
    pdf_root: Path | None = None
    base_url: str | None = None
    source_id: str | None = None
    manifest: Path | None = None
    raw_dir: Path | None = None
    extracted_dir: Path | None = None
    input: Path | None = None
    out: Path | None = None
    out_dir: Path | None = None
    verified: Path | None = None
    rejected: Path | None = None
    review: Path | None = None
    audit: Path | None = None
    report: Path | None = None
    answer_key: Path | None = None
    extractor: Literal["gemini", "glm-ocr", "embedded", "auto"] = "gemini"
    vlm_work_dir: Path | None = None
    vlm_model: str = "gemini-3.1-flash-lite"
    vlm_prompt: str = ""
    vlm_max_new_tokens: int = 8192
    vlm_dpi: int = 180
    vlm_max_pages: int | None = None
    jobs: int = 1
    vlm_devices: list[str] | None = None
    split: Literal["train", "dev", "frozen_test"] | None = None
    source_split: str | None = None
    source_revision: str | None = None
    min_score: int = 3
    judge_model: str | None = None
    audit_model: str | None = None

    @model_validator(mode="after")
    def validate_command_inputs(self) -> PipelineConfig:
        required_fields = {
            "init": [],
            "manifest-local": ["pdf_root", "out", "source_id"],
            "crawl": ["base_url", "out", "source_id"],
            "download": ["manifest", "raw_dir", "out"],
            "extract": ["manifest", "out_dir"],
            "build-candidates": ["manifest", "extracted_dir", "out"],
            "import-structured": ["input", "out", "source_id"],
            "filter-candidates": ["input", "verified", "rejected"],
            "apply-answer-key": ["input", "answer_key", "out"],
            "extract-answer-key": ["input", "out", "review"],
            "extract-answer-draft": ["input", "out", "audit"],
            "build-rlvr-subproblems": ["input", "verified", "review", "rejected"],
            "dedup": ["input", "out", "report"],
            "export-final": ["input", "out_dir"],
            "validate-final": ["input", "report"],
        }
        missing = [name for name in required_fields[self.command] if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.command} requires: {', '.join(missing)}")
        if self.jobs < 1:
            raise ValueError("jobs must be >= 1")
        if self.split == "frozen_test":
            raise ValueError("evaluation records cannot be imported into the training dataset")
        if self.command == "build-rlvr-subproblems" and self.judge_model and not self.audit_model:
            raise ValueError("an independent audit_model is required when judge_model is used")
        if self.command == "import-structured" and self.source_id == "physics_training_release":
            if self.source_split != "train" or not self.source_revision:
                raise ValueError("PHYSICS import requires --source-split train and the staged JSONL SHA-256")
        if self.judge_model and self.audit_model == self.judge_model:
            raise ValueError("judge_model and audit_model must be different model IDs")
        return self


def main() -> None:
    config = cli(PipelineConfig, description="Physics RLVR data curation pipeline")
    if config.command == "init":
        init_layout(config.data_dir)
        print(f"initialized {config.data_dir}")
    elif config.command == "manifest-local":
        _count("manifest rows", manifest_from_local_pdfs(config.pdf_root, config.out, config.source_id))
    elif config.command == "crawl":
        _count("crawled PDF rows", crawl_pdf_manifest(config.base_url, config.out, config.source_id))
    elif config.command == "download":
        _count("downloaded manifest rows", download_manifest(config.manifest, config.raw_dir, config.out))
    elif config.command == "extract":
        _count(
            "documents",
            extract_manifest(
                config.manifest,
                config.out_dir,
                extractor=config.extractor,
                vlm_work_dir=config.vlm_work_dir,
                vlm_model=config.vlm_model,
                vlm_prompt=config.vlm_prompt,
                vlm_max_new_tokens=config.vlm_max_new_tokens,
                vlm_dpi=config.vlm_dpi,
                vlm_max_pages=config.vlm_max_pages,
                jobs=config.jobs,
                vlm_devices=config.vlm_devices,
            ),
        )
    elif config.command == "build-candidates":
        _count("candidates", build_candidates(config.manifest, config.extracted_dir, config.out))
    elif config.command == "import-structured":
        _count(
            "structured candidates",
            import_structured_jsonl(
                config.input,
                config.out,
                config.source_id,
                config.split,
                source_split=config.source_split,
                source_revision=config.source_revision,
            ),
        )
    elif config.command == "filter-candidates":
        verified, rejected = filter_candidates(config.input, config.verified, config.rejected)
        print(f"verified {verified}; rejected {rejected}")
    elif config.command == "apply-answer-key":
        _count("answer-keyed candidates", apply_answer_key(config.input, config.answer_key, config.out))
    elif config.command == "extract-answer-key":
        accepted, review = extract_answer_key_file(config.input, config.out, config.review, config.min_score)
        print(f"wrote {accepted} answer-key rows; wrote {review} review rows")
    elif config.command == "extract-answer-draft":
        rows, audit = extract_answer_draft_file(config.input, config.out, config.audit)
        print(f"wrote {rows} draft answer-key rows; wrote {audit} audit rows")
    elif config.command == "build-rlvr-subproblems":
        verified, review, rejected = build_rlvr_subproblems_file(
            config.input,
            config.verified,
            config.review,
            config.rejected,
            config.min_score,
            judge_model=config.judge_model,
            audit_model=config.audit_model,
        )
        print(f"verified {verified}; review {review}; rejected {rejected}")
    elif config.command == "dedup":
        _count("deduplicated items", dedup_file(config.input, config.out, config.report))
    elif config.command == "export-final":
        print(f"exported {export_final(config.input, config.out_dir)}")
    elif config.command == "validate-final":
        _count("validation errors", write_validation_report(config.input, config.report))


def _count(name: str, count: int) -> None:
    print(f"wrote {count} {name}")
