"""Freeze evaluation questions outside the corpus and screen candidate prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
PHYSREASON_REVISION = "b63b7fa6e6bc99563038cafb1e136b060da2f91d"
SPECS = {
    "HiPhO": ("SciYu/HiPhO", "8e196c09a71e4e68b75c422defa512473359e0e5"),
    "OlympiadBench": ("Hothan/OlympiadBench", "91184b52131e7fc9455fef848035173aea8cc01a"),
    "PHYBench": ("Eureka-Lab/PHYBench", "d6d91c787b7abb865eb2490a328bf85a9f5095f0"),
    "PHYSICS-test": ("desimfj/PHYSICS", "6479757368c7d620ab826ba3ff7af44cb7eb3b5d"),
    "PhysOlym-A": ("shanyangmie/physolym-a", "808a5313a7bc1fdaf231d0bc52733c318e2e3f02"),
    "UGPhysics": ("UGPhysics/ugphysics", "523e71bc4e33356bb19de4af38c128796e8a8770"),
}


def normalize(text: str) -> str:
    text = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", text)
    return re.sub(r"\s+", " ", re.sub(r"[^\w]", " ", text.casefold())).strip()


def shingles(text: str) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


def shingle_index(sets: list[set[tuple[str, ...]]]) -> dict:
    index = defaultdict(list)
    for i, grams in enumerate(sets):
        for gram in grams:
            index[gram].append(i)
    return index


def nearest_lexical(grams: set, index: dict, sizes: list[int]) -> tuple[int, float]:
    intersections = Counter()
    for gram in grams:
        intersections.update(index.get(gram, ()))
    best_index, best_score = 0, 0.0
    for i, overlap in intersections.items():
        score = overlap / max(1, len(grams) + sizes[i] - overlap)
        if score > best_score or (score == best_score and i < best_index):
            best_index, best_score = i, score
    return best_index, best_score


def download(url: str, path: Path) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def extend_physreason(cache: Path, base: Path) -> dict:
    """Create an independent exclusion snapshot; preserve earlier pilot snapshots."""
    if cache.resolve() == base.resolve():
        raise ValueError("The extended evaluation cache must differ from the base cache")
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((base / "manifest.json").read_text())
    records = json.loads((base / "questions.json").read_text())
    if "PhysReason" in manifest["scope"]:
        raise ValueError("Base snapshot already contains PhysReason")
    filename = "PhysReason-full.zip"
    url = f"https://huggingface.co/datasets/zhibei1204/PhysReason/resolve/{PHYSREASON_REVISION}/{filename}"
    raw = download(url, cache / "PhysReason" / filename)
    extra = []
    with zipfile.ZipFile(cache / "PhysReason" / filename) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith("/problem.json"):
                continue
            original = json.loads(archive.read(name))
            structure = original["question_structure"]
            text = "\n\n".join(value for value in structure.values() if value)
            if not text:
                raise ValueError(f"Empty PhysReason question: {name}")
            extra.append({"benchmark": "PhysReason", "id": name, "text": text, "source_id": None})
    if len(extra) != 1200:
        raise ValueError(f"Unexpected PhysReason question count: {len(extra)}")
    manifest["scope"].append("PhysReason")
    manifest["files"].append({"benchmark": "PhysReason", "repository": "zhibei1204/PhysReason",
                              "revision": PHYSREASON_REVISION, "file": filename,
                              "sha256": hashlib.sha256(raw).hexdigest(), "question_count": len(extra),
                              "missing_statements": []})
    manifest["question_count"] = len(records) + len(extra)
    base_snapshot = manifest.pop("snapshot_sha256")
    manifest["base_snapshot_sha256"] = base_snapshot
    manifest["snapshot_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (cache / "questions.json").write_text(json.dumps(records + extra, ensure_ascii=False))
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    base_vectors = base / f"embeddings-{base_snapshot}.pt"
    if base_vectors.exists():
        vectors = torch.cat([torch.load(base_vectors, weights_only=True), embed([r["text"] for r in extra], cache)])
        torch.save(vectors, cache / f"embeddings-{manifest['snapshot_sha256']}.pt")
    return manifest


def freeze(cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    tasks = []
    for name, (repo, revision) in SPECS.items():
        data = json.loads(download(f"https://huggingface.co/api/datasets/{repo}/revision/{revision}",
                                   cache / name / "repository.json"))
        files = [s["rfilename"] for s in data["siblings"]]
        if name == "HiPhO":
            files = [f for f in files if f.startswith("data/") and f.endswith(".json")]
        elif name == "OlympiadBench":
            files = [f for f in files if "physics" in f and f.endswith(".parquet")]
        elif name == "PHYBench":
            files = ["PHYBench-questions_v1.json"]
        elif name == "PHYSICS-test":
            files = ["data/test.jsonl"]
        elif name == "PhysOlym-A":
            files = ["physolym-a.jsonl"]
        else:
            files = [f for f in files if f.endswith("/en.jsonl")]
        if not files:
            raise ValueError(f"No evaluation question files found for {name}")
        tasks.extend((name, repo, revision, f) for f in sorted(files))

    def load(task: tuple[str, str, str, str]) -> tuple[list[dict], dict]:
        name, repo, revision, filename = task
        path = cache / name / filename
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
        raw = download(url, path)
        if filename.endswith(".parquet"):
            columns = pq.ParquetFile(path).schema.names
            selected = [c for c in ["id", "question", "source"] if c in columns]
            rows = pq.read_table(path, columns=selected).to_pylist()
        elif filename.endswith(".jsonl"):
            rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        else:
            rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError(f"Unexpected evaluation file structure: {name}/{filename}")
        questions, missing = [], []
        for index, row in enumerate(rows):
            if name == "HiPhO" and "information" in row:
                continue
            key = {"PHYBench": "content", "UGPhysics": "problem"}.get(name, "question")
            text = row["messages"][0]["content"] if name == "PhysOlym-A" else row.get(key)
            if not isinstance(text, str) or not text.strip():
                if name != "PHYBench" or row.get("id") != 778 or text != "":
                    raise ValueError(f"Missing evaluation question: {name}/{filename}/{index}")
                missing.append({"id": row["id"], "reason": "empty statement in pinned upstream release"})
                continue
            context = row.get("context", "") if name == "HiPhO" else ""
            text = text + "\n\n" + (context if isinstance(context, str) else "")
            questions.append({"benchmark": name, "id": f"{filename}:{row.get('id', row.get('index', index))}",
                              "text": text.strip(), "source_id": row.get("source")})
        return questions, {"benchmark": name, "repository": repo, "revision": revision,
                           "file": filename, "sha256": hashlib.sha256(raw).hexdigest(),
                           "question_count": len(questions), "missing_statements": missing}

    records, files = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for questions, manifest in pool.map(load, tasks):
            records.extend(questions)
            files.append(manifest)
            print(f"Frozen {manifest['benchmark']}/{manifest['file']}: {len(questions)} questions", flush=True)
    manifest = {"scope": list(SPECS), "files": files, "question_count": len(records),
                "semantic_model": MODEL, "semantic_revision": MODEL_REVISION,
                "semantic_max_tokens": 256, "semantic_review_threshold": 0.80,
                "lexical_review_threshold": 0.35, "policy": "exact/source matches blocked; similarity candidates held for review"}
    manifest["coverage_notes"] = [
        "PHYBench id 778 has an empty statement in the pinned release; 499 available statements are indexed.",
        "Similarity screening is a review aid, not proof that all paraphrases or unlisted benchmarks are absent.",
    ]
    manifest["snapshot_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (cache / "questions.json").write_text(json.dumps(records, ensure_ascii=False))
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def embed(texts: list[str], cache: Path) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    vectors = []
    with torch.inference_mode():
        for offset in range(0, len(texts), 32):
            inputs = tokenizer(texts[offset:offset + 32], padding=True, truncation=True,
                               max_length=256, return_tensors="pt").to(device)
            tokens = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            vector = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(torch.nn.functional.normalize(vector, dim=1).cpu())
            if offset % 512 == 0:
                print(f"Semantic index: {min(offset + 32, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(vectors)


def prune_downloads(cache: Path) -> int:
    manifest = json.loads((cache / "manifest.json").read_text())
    questions = json.loads((cache / "questions.json").read_text())
    if len(questions) != manifest["question_count"]:
        raise ValueError("Cannot prune downloads before the extracted index is complete")
    total = 0
    for entry in manifest["files"]:
        path = cache / entry["benchmark"] / entry["file"]
        if path.exists():
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Source checksum changed before pruning: {path}")
            total += len(content)
            path.unlink()
    return total


def screen(cache: Path, artifact_path: Path) -> dict:
    manifest = json.loads((cache / "manifest.json").read_text())
    evaluation = json.loads((cache / "questions.json").read_text())
    candidates = json.loads(artifact_path.read_text())["records"]
    index_path = cache / f"embeddings-{manifest['snapshot_sha256']}.pt"
    if index_path.exists():
        vectors = torch.load(index_path, weights_only=True)
    else:
        vectors = embed([r["text"] for r in evaluation], cache)
        torch.save(vectors, index_path)
    if vectors.shape[0] != len(evaluation):
        raise ValueError("Cached evaluation embeddings do not match the question index")
    candidate_vectors = embed([r["question"] for r in candidates], cache)
    exact = {normalize(r["text"]): i for i, r in enumerate(evaluation)}
    eval_shingles = [shingles(r["text"]) for r in evaluation]
    lexical_index = shingle_index(eval_shingles)
    lexical_sizes = [len(grams) for grams in eval_shingles]
    reports = []
    for i, row in enumerate(candidates):
        if i % 256 == 0:
            scores = candidate_vectors[i:i + 256] @ vectors.T
        grams = shingles(row["question"])
        lex_index, lexical_score = nearest_lexical(grams, lexical_index, lexical_sizes)
        value, indices = scores[i % 256].topk(3)
        identity = row.get("evaluation_identity")
        if not identity and row.get("source", "estonian_physics_olympiad") == "estonian_physics_olympiad":
            identity = "estonian-" + row["source_id"]
        elif not identity and row.get("source") == "usapho_archive":
            match = re.fullmatch(r"usapho-(\d{4})-([a-z]\d+)", row["source_id"])
            if match:
                identity = f"USAPhO_{match[1]}_problem_{match[2].upper()}"
        sources = [e for e in evaluation if identity is not None and e["source_id"] == identity]
        exact_index = exact.get(normalize(row["question"]))
        overlaps = bool(sources) or exact_index is not None
        near = float(value[0]) >= manifest["semantic_review_threshold"] or lexical_score >= manifest["lexical_review_threshold"]
        report = {"problem_id": row["problem_id"], "status": "blocked" if overlaps else "review" if near else "clear",
                  "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                  "snapshot_sha256": manifest["snapshot_sha256"],
                  "source_identity_hits": [{"benchmark": e["benchmark"], "id": e["id"]} for e in sources],
                  "exact_match": exact_index is not None,
                  "semantic_nearest": [{"benchmark": evaluation[int(j)]["benchmark"], "id": evaluation[int(j)]["id"],
                                        "cosine": round(float(v), 6)} for v, j in zip(value, indices)],
                  "lexical_nearest": {"benchmark": evaluation[lex_index]["benchmark"], "id": evaluation[lex_index]["id"],
                                      "jaccard": round(lexical_score, 6)}}
        reports.append(report)
        if len(candidates) < 100 or i % 100 == 0 or i + 1 == len(candidates):
            print(f"Screened {i + 1}/{len(candidates)}: {report['status']} (cosine {float(value[0]):.3f})", flush=True)
    return {"manifest": manifest, "records": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--keep-downloads", action="store_true")
    parser.add_argument("--extend-base", type=Path, help="Extend an existing snapshot with PhysReason in a separate cache")
    args = parser.parse_args()
    if not (args.cache / "manifest.json").exists():
        if args.extend_base:
            extend_physreason(args.cache, args.extend_base)
        else:
            freeze(args.cache)
    if not args.keep_downloads:
        print(f"Removed {prune_downloads(args.cache) / 1_000_000:.1f} MB of parsed downloads; question index and source hashes retained", flush=True)
    result = screen(args.cache, args.candidates) if args.candidates else json.loads((args.cache / "manifest.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
