"""Prefect orchestration of the end-to-end recommendation pipeline.

Run once:                 python dags/recommendation_flow.py
Run on a schedule:        python dags/recommendation_flow.py --serve   (daily 02:00, Asia/Kolkata)
UI (optional):            prefect server start   →  http://127.0.0.1:4200

Each stage is a Prefect task. Explicit ``wait_for`` edges encode the DAG

  ingest → validate → prepare → transform → features → train → evaluate

so a stage only starts after its upstream task succeeded; ingestion retries on
network errors; a data-quality gate fails the run if too many rows are rejected.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from prefect import flow, get_run_logger, task  # noqa: E402

from src.common.utils import load_config  # noqa: E402
from src.pipeline import STAGES  # noqa: E402

MAX_QUARANTINE_RATE = 0.01  # data-quality gate: fail the run if >1% of interactions are rejected


@task(name="1-ingest", retries=2, retry_delay_seconds=10)
def ingest_task(cfg):
    return STAGES["ingest"](cfg)


@task(name="2-validate")
def validate_task(cfg):
    report = STAGES["validate"](cfg)
    q = report["interactions"]["rows_quarantined"] / report["interactions"]["rows_in"]
    get_run_logger().info("Data-quality gate: quarantine rate %.4f%% (limit %.2f%%)", 100 * q, 100 * MAX_QUARANTINE_RATE)
    if q > MAX_QUARANTINE_RATE:
        raise ValueError(f"Data-quality gate failed: {q:.2%} of interactions quarantined")
    return report


@task(name="3-prepare")
def prepare_task(cfg):
    return STAGES["prepare"](cfg)


@task(name="4-transform")
def transform_task(cfg):
    return STAGES["transform"](cfg)


@task(name="5-features")
def features_task(cfg):
    return STAGES["features"](cfg)


@task(name="6-train")
def train_task(cfg):
    return STAGES["train"](cfg)


@task(name="7-evaluate")
def evaluate_task(cfg):
    df = STAGES["evaluate"](cfg)
    return df.drop(columns="key").round(4).to_dict(orient="records")


@flow(name="movielens-recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str = "configs/pipeline.yaml"):
    cfg = load_config(ROOT / config_path)
    log = get_run_logger()
    log.info("Starting pipeline with config %s (seed=%s)", config_path, cfg["project"]["seed"])
    a = ingest_task.submit(cfg)
    b = validate_task.submit(cfg, wait_for=[a])
    c = prepare_task.submit(cfg, wait_for=[b])
    d = transform_task.submit(cfg, wait_for=[c])
    e = features_task.submit(cfg, wait_for=[d])
    f = train_task.submit(cfg, wait_for=[e])
    g = evaluate_task.submit(cfg, wait_for=[f])
    results = g.result()
    for r in results:
        log.info("%-30s NDCG@10=%.4f P@10=%.4f Coverage@10=%.4f", r["model"], r["ndcg_at_10"],
                 r["precision_at_10"], r["coverage_at_10"])
    return results


if __name__ == "__main__":
    if "--serve" in sys.argv:
        # long-running scheduler: re-runs the whole DAG every day at 02:00 IST
        recommendation_pipeline.serve(name="daily-retrain", cron="0 2 * * *", timezone="Asia/Kolkata")  # type: ignore[call-arg]
    else:
        recommendation_pipeline()
