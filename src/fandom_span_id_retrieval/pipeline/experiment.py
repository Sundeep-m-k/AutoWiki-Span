from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


# Project root = /home/sundeep/Fandom_Span_ID_Retrieval
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ExperimentConfig:
    task: str      # "span-id", "retrieval", "rerank"
    model: str     # e.g. "deberta-v3-base"
    variant: str   # e.g. "baseline"
    domain: str    # e.g. "money-heist"
    seed: int      # e.g. 42

    @property
    def run_name(self) -> str:
        return f"{self.task}__{self.model}__{self.variant}__seed{self.seed}"

    @property
    def outputs_root(self) -> Path:
        return PROJECT_ROOT / "outputs"

    @property
    def outputs_dir(self) -> Path:
        return self.outputs_root / self.task / self.domain / self.run_name

    @property
    def logs_root(self) -> Path:
        return PROJECT_ROOT / "data" / "logs"

    @property
    def logs_dir(self) -> Path:
        return self.logs_root / self.domain


def load_experiment_cfg(cfg: Dict[str, Any]) -> ExperimentConfig:
    exp = cfg["experiment"]
    return ExperimentConfig(
        task=str(exp["task"]),
        model=str(exp["model"]),
        variant=str(exp["variant"]),
        domain=str(exp["domain"]),
        seed=int(exp.get("seed", 0)),
    )
