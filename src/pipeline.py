"""Stage registry + dependency graph shared by every orchestrator.

The DAG is declared once here and consumed by:
  * ``dags/recommendation_flow.py``   – Prefect flow (primary orchestrator)
  * ``run_pipeline.py``               – dependency-free scheduled Python runner
"""
from __future__ import annotations

from src.evaluation import evaluate
from src.features import build_features
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate

STAGES = {
    "ingest": ingest.run,
    "validate": validate.run,
    "prepare": prepare.run,
    "transform": transform.run,
    "features": build_features.run,
    "train": train.run,
    "evaluate": evaluate.run,
}

# stage -> upstream stages it depends on
DEPENDENCIES = {
    "ingest": [],
    "validate": ["ingest"],
    "prepare": ["validate"],
    "transform": ["prepare"],
    "features": ["transform"],
    "train": ["features"],
    "evaluate": ["train"],
}


def topological_order() -> list[str]:
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        for d in DEPENDENCIES[n]:
            visit(d)
        seen.add(n)
        order.append(n)

    for n in DEPENDENCIES:
        visit(n)
    return order
