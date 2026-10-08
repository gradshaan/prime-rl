"""English, text-only physics RLVR taskset on the v3 data contract."""

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from physics_rlvr_common import SYSTEM_PROMPT, task_prompt, verify_prediction
from physics_rlvr_common.verifier import answer_from_dict

DATA_DIR = Path(__file__).parents[2] / "data_pipeline/data/final/v3"


class PhysicsData(vf.TaskData):
    answers: list[dict]


class PhysicsTask(vf.Task[PhysicsData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        answers = [answer_from_dict(a) for a in self.data.answers]
        # sympy can take seconds on odd expressions; a thread keeps the event loop free
        return await asyncio.to_thread(verify_prediction, trace.last_reply, answers)

    async def validate(self, runtime: vf.Runtime) -> bool:
        gold = [{"label": a["label"], "value": a["value"], "unit": a["unit"]} for a in self.data.answers]
        answers = [answer_from_dict(a) for a in self.data.answers]
        return verify_prediction(f"<final>{json.dumps(gold)}</final>", answers) == 1.0


class PhysicsConfig(vf.TasksetConfig):
    split: str = "train"
    dataset_name: str | None = None
    """Hugging Face dataset to load, e.g. gradshaan/verifiable-physics. None reads the local v3 export."""
    dataset_revision: str | None = None


class PhysicsTaskset(vf.Taskset[PhysicsTask, PhysicsConfig]):
    def load(self) -> Iterator[PhysicsTask]:
        if self.config.dataset_name is None:
            lines = (DATA_DIR / f"{self.config.split}.jsonl").read_text().splitlines()
            rows = [json.loads(line) for line in lines if line.strip()]
        else:
            from datasets import load_dataset

            rows = load_dataset(self.config.dataset_name, split=self.config.split, revision=self.config.dataset_revision)
        for row in rows:
            labels = [a["label"] for a in row["answers"]]
            yield PhysicsTask(
                PhysicsData(
                    id=row["problem_id"],
                    prompt=task_prompt(row["shared_context"], row["question"], labels),
                    system_prompt=SYSTEM_PROMPT,
                    answers=row["answers"],
                ),
                self.config.task,
            )
