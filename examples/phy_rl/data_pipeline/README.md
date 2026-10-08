# Physics RLVR data pipeline

This pipeline builds English, text-only physics problems with deterministic,
per-output numeric or symbolic rewards. Source admission is allowlisted in
[`configs/sources.toml`](configs/sources.toml); the shared policy rejects
benchmark sources and competition material after 2023. IPhO 2024–2026 and all
other post-2023 contest problems stay out of training. Evaluation datasets do
not belong in the pipeline's train or development artifacts.
`Darkyy/phy-rl-base` and `desimfj/PHYSICS` are excluded in full. The shared
policy checks upstream provenance as well as source labels, so relabeling a
record does not admit material from either repository.

## Setup

```bash
uv sync --project examples/phy_rl/data_pipeline --all-extras
```

Run commands from the repository root. The CLI uses pydantic-config, so options
work as typed `--field value` overrides or from a TOML file with `@file.toml`:

```bash
uv run --project examples/phy_rl/data_pipeline physics-rlvr-data --help
uv run --project examples/phy_rl/data_pipeline physics-rlvr-data \
  --command init --data-dir examples/phy_rl/data_pipeline/data
```

## Curation flow

Prepare a manifest containing separately identified problem and solution PDFs.
Contest records need an explicit year, source ID, competition, and problem or
paper identity. `manifest-local` infers metadata from filenames; use a reviewed
manifest JSONL when filenames do not clearly identify the contest, year, paper
type, round, and problem number.

```bash
uv run --project examples/phy_rl/data_pipeline physics-rlvr-data \
  --command manifest-local \
  --pdf-root /path/to/pre-2024-ipho-pdfs \
  --out examples/phy_rl/data_pipeline/data/manifest.jsonl \
  --source-id ipho_archive

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data \
  --command download \
  --manifest examples/phy_rl/data_pipeline/data/manifest.jsonl \
  --raw-dir examples/phy_rl/data_pipeline/data/raw_pdfs \
  --out examples/phy_rl/data_pipeline/data/manifest_downloaded.jsonl

GEMINI_API_KEY=... uv run --project examples/phy_rl/data_pipeline physics-rlvr-data \
  --command extract \
  --manifest examples/phy_rl/data_pipeline/data/manifest_downloaded.jsonl \
  --out-dir examples/phy_rl/data_pipeline/data/extracted \
  --extractor gemini

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data \
  --command build-candidates \
  --manifest examples/phy_rl/data_pipeline/data/manifest_downloaded.jsonl \
  --extracted-dir examples/phy_rl/data_pipeline/data/extracted \
  --out examples/phy_rl/data_pipeline/data/candidates/candidates.jsonl
```

Create answer candidates and send them through two distinct model passes. The
second model independently checks output coverage, units, correctness, and
self-containment. Rows that either pass cannot verify stay in the review file.
No model API is called unless this command is run.

```bash
GEMINI_API_KEY=... uv run --project examples/phy_rl/data_pipeline physics-rlvr-data \
  --command build-rlvr-subproblems \
  --input examples/phy_rl/data_pipeline/data/candidates/candidates.jsonl \
  --verified examples/phy_rl/data_pipeline/data/verified/verified.jsonl \
  --review examples/phy_rl/data_pipeline/data/audits/review.jsonl \
  --rejected examples/phy_rl/data_pipeline/data/rejected/rejected.jsonl \
  --judge-model <first-model> --audit-model <different-model>

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data \
  --command dedup \
  --input examples/phy_rl/data_pipeline/data/verified/verified.jsonl \
  --out examples/phy_rl/data_pipeline/data/verified/deduped.jsonl \
  --report examples/phy_rl/data_pipeline/data/audits/duplicates.csv

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data \
  --command export-final \
  --input examples/phy_rl/data_pipeline/data/verified/deduped.jsonl \
  --out-dir examples/phy_rl/data_pipeline/data/final/v3
```

The export assigns stable family IDs, caps number-only variants per family,
keeps each family in one split, and creates a deterministic five-percent dev
holdout. Its `metadata.json` records counts by source, topic, competition and
answer type, policy-file checksums, and checksums for both JSONL files. The
environment verifies those checksums before loading local data.

## Competition PDF batches

`tools/phy_rl_extract_archives.py` reads paired question and solution PDFs from
the pinned archive inventory. It checks each refetched PDF's SHA-256, crops the
original problem, and transcribes its givens and requested parts. A second model
checks the statement against question images without the solution. One targeted
transcription repair is permitted; failed repairs stay held. Unmarked whole
papers have no fallback to the full document. The bounded runner currently uses
the individual-problem IPhO, APhO, Nordic/Baltic and WoPhO PDFs.

Dependent subparts stay together. Independent calculations can become separate
tasks, each retaining its original parent ID and labels. Qualitative requests
remain in the parent transcription but cannot be converted into scalar targets.
Part contexts contain source givens and explicit diagram geometry. Each admitted
candidate's English question is locked through answer curation; a change to that
question holds the row. Source PDFs and rendered pages remain in memory only.

`tools/phy_rl_competition_run.py` runs small extraction and physics-audit batches
under a cumulative spending cap. Gemini 3 Flash curates answer contracts; a blind
Gemini 2.5 Flash solve receives the question and output labels, without targets or
the worked solution. The release requires complete label coverage, answer
agreement, verifier perturbation checks and a final benchmark screen. The runner
stops if fewer than 40% of a ten-row physics pilot pass, fewer than 25% are
released, or fewer than 25% of source parents produce screened candidates.

```sh
uv --no-config run --project examples/phy_rl/data_pipeline \
  tools/phy_rl_competition_run.py \
  examples/phy_rl/data_pipeline/data/competition_5k/scale_100 \
  --limit 100 --budget 2.6 --batch-size 20 --concurrency 4
```

This runner requires the existing `competition_5k/source_queue.jsonl` and
`repair_input.jsonl`, and the repaired pilot's `curation_queue.jsonl`. It writes a
fixed plan for new original parents and resumes each stage from its checkpoints.
File locks prevent two controllers or curators from using the same output
directory. Cost ledgers and raw model responses are retained. With the explicit
`--allow-reserved-unconfirmed` option, an interrupted request's complete ceiling
counts against the cap; the matching request is never sent again.

Within the run directory, `status.json` tracks original parents, extracted tasks,
completed checks, released tasks and costs separately. `accepted.jsonl` contains
only released competition tasks. The merge also rejects repeated source
subparts and question families across batches. `viewer.json` contains that same
release with its current summary. The source queue and transcription count do
not represent a training release, and `--limit 100` counts original parents.

## Reuse published training datasets

`tools/phy_rl_gather_training.py` imports pinned releases of
[TextbookReasoning](https://huggingface.co/datasets/MegaScience/TextbookReasoning)
and [Nemotron RL Science](https://huggingface.co/datasets/nvidia/Nemotron-RL-Science-v1).
It keeps the physics partition, original question, reference answer, source row
index and attribution. TextbookReasoning also supplies a worked solution. The
importer preserves that text; it does not extract a new gold answer from it.

```bash
uv run --project examples/phy_rl/data_pipeline tools/phy_rl_gather_training.py \
  examples/phy_rl/data_pipeline/data/training_sources --download --pool-size 14000

uv run --project examples/phy_rl/data_pipeline tools/phy_rl_screen.py \
  /tmp/phy-rl-training-eval-cache \
  examples/phy_rl/data_pipeline/data/training_sources/screening.json \
  --extend-base /tmp/phy-rl-eval-cache \
  --candidates examples/phy_rl/data_pipeline/data/training_sources/preflight.json

uv run --project examples/phy_rl/data_pipeline tools/phy_rl_gather_training.py \
  examples/phy_rl/data_pipeline/data/training_sources \
  --screening examples/phy_rl/data_pipeline/data/training_sources/screening.json \
  --existing /path/to/current-corpus.parquet --limit 5000 --prune-cache
```

The screening command needs the base exclusion cache from the source-collection
flow. `--extend-base` creates a separate snapshot with PhysReason; earlier pilot
snapshots stay intact. Evaluation question text and downloaded evaluation files
remain outside the training directory. Saved screening reports contain matched
evaluation IDs and similarity scores.

The importer removes explicit image dependencies, source placeholders, pure
proof requests, named benchmarks, post-2023 named competitions and some
applications outside the selected physics topics. Selection favors worked
solutions, derivations and richer statements, then samples across sources and
topics. Numeric substitutions share one family ID. Neither the score nor the
topic label measures difficulty; base-model rollouts still have to establish it.

`candidates_5000.jsonl` holds at most 5,000 candidates, with possible benchmark
overlaps in `overlap_review.jsonl.gz`. Each candidate has `release_status: held`
and an empty `required_outputs` list. Independent answer checks, output coverage,
units and verifier tests must be complete before release. TextbookReasoning lacks
book/page attribution; Nemotron's original answer and underlying contest source,
if any, need review. A clear similarity screen is evidence for review, not proof
that all benchmark paraphrases are absent.

The final command removes reproducible raw downloads after it checks the saved
candidate checksum and row count. It retains selected records, overlap holds,
clear reserve records, release revisions, source hashes and screening evidence.

## Data contract

Each final record has a complete question and solution, source and solution
hashes, source/license provenance, topic and difficulty when known, a family
ID, and one labeled target for every requested output. Numeric targets carry
units plus explicit absolute and relative tolerances. Symbolic targets use
SymPy equivalence checks. The model must return one `<final>` JSON array with
the requested labels; omitted outputs receive per-output partial credit, while
duplicate or unrequested outputs receive zero.

The environment loads only `physics_rlvr_v3` artifacts. Local data must match
its manifest hashes; remote Hugging Face data must use an exact commit hash and
contain only policy-approved train rows. Keep benchmark and test sets in a
separate evaluation repository and never use them for RL training or dev-set
selection.

`desimfj/PHYSICS` and `physics_training_release` are blocked as training
sources. The public PHYSICS test release stays in the separate exclusion
index so candidate questions can be checked for overlap. Its questions and
solutions are never supplied to the curation models or exported into training.

## Native-source pilot

The low-cost OpenRouter runner is `tools/phy_rl_pilot.py`. It accepts a pinned
Estonian source-ID manifest, checkpoints each problem, and uses a shared async
HTTP client. `--concurrency 8 --requests-per-minute 60` bounds traffic; a 429
reduces concurrency and triggers a shared cooldown. The budget includes charged
responses and reservations for in-flight or unconfirmed requests. Qwen uses a
required result tool and local JSON Schema validation. Returned invalid or
truncated results receive no automatic paid repair retry.

The completed 50-problem batch is in `data/pilot_50`. Replay cached responses
without making paid requests:

```sh
uv --no-config run --project examples/phy_rl/data_pipeline tools/phy_rl_pilot.py \
  examples/phy_rl/data_pipeline/data/pilot_50 --offline \
  --manifest examples/phy_rl/data_pipeline/data/pilot_50/source_manifest.json
```

If question text changes, rerun `tools/phy_rl_screen.py` against the external
pinned evaluation index before analysis. Parsed evaluation downloads are pruned
after their hashes and statements are preserved; `--keep-downloads` disables
that pruning. Model responses and original training-source text are retained.

```sh
uv --no-config run --project examples/phy_rl/data_pipeline tools/phy_rl_analyze.py \
  examples/phy_rl/data_pipeline/data/pilot_50 \
  --site-dir artifacts/phy-rl-pilot-viewer/dist
```

Analysis validates source hashes, joins overlap checks with documented manual
comparisons, and writes `REPORT.md`, `checked_candidates.staging.jsonl`, and
`review.jsonl`. These remain explicitly staged: the exporter and environment
reject them as training input. A hosted viewer refresh requires publication of
the updated Site snapshot.

## Curate the screened 5,000-problem pool

`tools/phy_rl_curate.py` processes collected training records with the same
answer schema and reward checks as the native pilot. Its default input uses native
olympiad/preparation material; the mixed pool adds pinned TextbookReasoning and
Nemotron physics training releases. Those release sources require their admitted revision and
source split; evaluation releases remain blocked.

Prepare an immutable, family-deduplicated input once. The default pool uses
competition and preparation queues. It requires at least 50% actual olympiad
problems and at least one IPhO problem. Indexed PDFs do not meet this requirement:
the statement and solution must first be transcribed and screened. Preparation
textbooks such as Savchenko do not count toward the olympiad fraction.
Use `--candidate-file` to include another prepared, screened source queue.
`--pool mixed` also includes the admitted general training releases.

```sh
uv --no-config run --project examples/phy_rl/data_pipeline tools/phy_rl_curate.py \
  prepare examples/phy_rl/data_pipeline/data/curation_5k --count 5000
```

With `OPENROUTER_API_KEY` in the process environment, start or resume:

```sh
uv --no-config run --project examples/phy_rl/data_pipeline tools/phy_rl_curate.py \
  run examples/phy_rl/data_pipeline/data/curation_5k \
  --budget 20 --auditor qwen/qwen3.5-flash-02-23 \
  --minimum-model-pass-rate 0.4 --minimum-release-rate 0.25 \
  --allow-reserved-unconfirmed \
  --concurrency 8 --requests-per-minute 90
```

The budget is cumulative for this output directory, including charged responses
and reserved costs for unknown charges. Curation and blind solving use different
models. The blind solver receives the statements and output labels, without
reference answers or solutions. A rejected source statement skips blind solving.
Completed responses are cached; interrupted requests with unknown charges are
held rather than sent again. `--allow-reserved-unconfirmed` lets other source
records continue while the maximum possible costs of those interrupted requests
remain reserved inside the cap. It does not clear reservations or approve the
affected rows. Omit this option to stop until charges are reconciled. Symbolic checks run in separate processes with a
60-second limit. Invalid expressions and SymPy recursion failures stay in review.
Unexpected worker errors propagate after the other workers have drained their
requests and saved completed responses.

The official Alibaba Qwen Flash route uses native JSON Schema output. Its
endpoint does not support the forced tool-choice values used by the separate
Qwen 35B route. Both response paths require complete output and local schema
validation. The native JSON solver uses plain algebra inside string fields to
avoid provider failures on unescaped LaTeX commands. Local notation conversion
preserves the raw response and records its changes before reward validation.

After the first 50 records, the runner checks failures, agreement rate and
projected cost. Healthy runs continue in batches of 100. Each batch screens the
final English statements against the frozen external evaluation index, then
checks for repeated question families before admitting rows. Excessive API
failures, unknown charges or excessive projected cost stop expansion. A nonzero
`--minimum-model-pass-rate` also stops a window whose pass rate is below that
threshold. Separately, `--minimum-release-rate` stops when fewer than 25% of the
recent rows pass final admission, including decontamination. Investigate low
yield on the first 50 rows before expanding. Setting either threshold to zero
disables that guard; it does not change per-row admission checks.
Creating a `STOP` file inside the output directory requests a stop
after the current batch. Remove it before an intentional resume.

The run writes `status.json`, per-record checkpoints and a charged-call ledger.
`accepted.jsonl` contains rows that passed the model, verifier and final
decontamination checks; `review.jsonl` retains failed rows and their reasons.
`viewer.json` provides the current batch snapshot. Accepted rows record their
validation level as two-model agreement plus programmatic checks. Human review
and base-model difficulty measurements are separate steps. Auditing 5,000 inputs
does not guarantee 5,000 admitted outputs.

The detached local session uses a dedicated tmux server:

```sh
tmux -L phy-rl attach -t phy-rl-curate-5k
```

Its runner uses `caffeinate -i` to prevent idle system sleep. Checkpoints allow
resumption after a process or machine interruption. Credentials belong in the
process environment, never in the run configuration, source records or logs.

## Collect original sources without model calls

`tools/phy_rl_collect.py` collects pinned Estonian LaTeX and competition
archives through 2023. The PDF sources cover IPhO, APhO, WoPhO, USAPhO,
EuPhO, NBPhO, INPhO, Physics Náboj, Australian examinations and Czech
categories A–C. Czech competition years use the end of the school year;
PDF upload dates do not determine admission. The original identity includes
category and round where those identify different papers.

```sh
uv --no-config run --project examples/phy_rl/data_pipeline tools/phy_rl_collect.py \
  examples/phy_rl/data_pipeline/data/source_pool \
  --eval-cache /tmp/phy-rl-eval-cache --workers 6

uv --no-config run --project examples/phy_rl/data_pipeline tools/phy_rl_screen.py \
  /tmp/phy-rl-eval-cache \
  examples/phy_rl/data_pipeline/data/source_pool/native_screening.json \
  --candidates examples/phy_rl/data_pipeline/data/source_pool/native_preflight.json
```

Rerun the collector with `--offline` after screening to build `next_native_batch.json` and
`curation_queue.jsonl`. The queue contains unused native problems whose source
checks and overlap screen passed. Similarity matches stay in
`overlap_review.jsonl`. Screen the final English question again after curation,
because its wording and conditions can differ from the source rendering.

The collector resumes from checked source caches. PDFs are extracted in
memory, then discarded; compressed page text, image/drawing counts, source
URLs and SHA-256 hashes are retained. Refetch a PDF and match its stored hash
when reviewing a formula or figure. Private-use font characters are flagged:
they can encode missing digits even when extraction reports no replacement
characters. Embedded text alone is insufficient evidence that math survived.

Whole-paper extracts remain source quarantine. They can contain a problem
that belongs to a benchmark; only the reviewed, isolated problem can reach
the curation runner. Known benchmark identities are excluded before downloads
of individually indexed problems. Subanswers, translations and solution PDFs
do not increase the original-problem count. Heuristic PDF boundaries remain
provisional until a source review confirms them.

`REPORT.md` separates indexed source material, the curation queue and the
existing pilot review backlog. `pilot_final_review.jsonl` identifies held
outputs without changing their acceptance status. No collector output is a
training release. PDF sources still require formula transcription, figure
review, complete targets and independent validation; source permissions are
recorded separately. OpenStax is excluded from this collector because its
current terms require permission for LLM ingestion.

## Competition extraction and answer review

`tools/phy_rl_extract_archives.py` transcribes paired problem and solution
pages. A separate source checker receives both sets of rendered pages and
checks each task's statement, worked solution and targets. Unresolved source
differences stay in review after one transcription repair. Missing geometry
cannot be supplied from the solution as an extra condition in the question.
The record preserves the page hashes and both source-check results.

The curator freezes the checked English question. A contract review checks
that the output labels cover what the question asks, with exact quotations
from that question. This reviewer receives no reference answers. A numerical
specialization is a separate target only when the question requests it.
The blind solver then receives the question and requested labels.

`--resolver` enables one stronger blind solve for unresolved answer agreement,
output coverage or symbol domains. Source, schema and contract failures must
be repaired before this step. The resolution preserves the initial audit and
records the stronger model separately. No reference solution is sent to either
blind solver.

Symbolic verification normalizes arithmetic notation, applies reviewed sign
assumptions and converts compatible units, including temperature offsets.
Approximate symbolic answers require explicit numeric parameter bindings and
a declared relative tolerance. Each binding must have a question-bound review,
an exact source quotation containing the value, and whole-task scope. Initial
conditions cannot serve as unrestricted substitutions. The relative error
must simplify to a constant; unresolved symbols cannot pass by numerical
sampling.

Provider responses undergo local schema validation. One complete JSON code
fence can be removed while preserving the raw response. Confirmed truncated
responses can receive the configured output repair; requests with unknown
charges remain reserved and are not automatically repeated. The client bounds
request duration and drains submitted work on a stop signal. A dead tmux pane
means that its worker has stopped.

`tools/phy_rl_recover_formats.py` rechecks saved records without model calls.
Use `--all-records` to check both accepted and held rows in a fresh directory.
Keep its output separate from the original responses. Released counts refer
to admitted tasks, while source-parent and candidate counts remain separate.
