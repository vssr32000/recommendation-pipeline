"""Plain-Python orchestrator (no Prefect needed) – runs the DAG in dependency order.

    python run_pipeline.py                     # full pipeline
    python run_pipeline.py --stage validate    # a single stage
    python run_pipeline.py --from train        # a stage and everything downstream

Usable from cron, e.g.  0 2 * * *  cd /path/to/repo && python run_pipeline.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.common.utils import get_logger, load_config  # noqa: E402
from src.pipeline import DEPENDENCIES, STAGES, topological_order  # noqa: E402


def downstream(start: str) -> list[str]:
    order = topological_order()
    keep = {start}
    for s in order:
        if any(d in keep for d in DEPENDENCIES[s]):
            keep.add(s)
    return [s for s in order if s in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stage", choices=list(STAGES))
    g.add_argument("--from", dest="from_stage", choices=list(STAGES))
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    log = get_logger("orchestrator", cfg)
    plan = [a.stage] if a.stage else downstream(a.from_stage) if a.from_stage else topological_order()
    log.info("Execution plan: %s", " -> ".join(plan))
    status, t0 = {}, time.perf_counter()
    for s in plan:
        ts = time.perf_counter()
        try:
            STAGES[s](cfg)
            status[s] = f"SUCCESS ({time.perf_counter() - ts:.1f}s)"
        except Exception:
            status[s] = "FAILED"
            log.exception("Stage '%s' failed – downstream stages skipped", s)
            for rest in plan[plan.index(s) + 1:]:
                status[rest] = "SKIPPED (upstream failed)"
            break
    log.info("Run summary (%.1fs total):", time.perf_counter() - t0)
    for s, st in status.items():
        log.info("  %-10s %s", s, st)
    sys.exit(0 if all(v.startswith("SUCCESS") for v in status.values()) else 1)


if __name__ == "__main__":
    main()
