"""Print the latest tracked training runs (params + metrics) from MLflow.

    python scripts/show_mlflow_runs.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.utils import load_config  # noqa: E402
from src.training.train import setup_mlflow  # noqa: E402

cfg = load_config()
setup_mlflow(cfg)
runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]], filter_string="tags.stage = 'training'",
                          order_by=["start_time DESC"], max_results=4)
pd.set_option("display.width", 220)
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: {cfg['mlflow']['experiment']}  |  latest pipeline execution (4 training runs)\n")
p = {"tags.model": "model", "params.neighbours": "neighbours", "params.shrinkage": "shrinkage", "params.factors": "factors",
     "params.reg": "reg", "params.min_ratings": "min_ratings", "params.data.train_rows": "train_rows",
     "params.features.registry_version": "feat_ver"}
print("Parameters:")
print(runs[["run_id"] + list(p)].rename(columns=p).assign(run_id=lambda d: d.run_id.str[:12]).fillna("–").to_string(index=False))
m = {"tags.model": "model", "metrics.precision_at_10": "P@10", "metrics.recall_at_10": "R@10", "metrics.ndcg_at_10": "NDCG@10",
     "metrics.coverage_at_10": "Cov@10", "metrics.diversity_at_10": "Div@10", "metrics.rmse": "RMSE", "metrics.mae": "MAE",
     "metrics.train_seconds": "train_s"}
print("\nMetrics:")
print(runs[list(m)].rename(columns=m).round(4).to_string(index=False, na_rep="–"))
arts = mlflow.artifacts.list_artifacts(run_id=runs.iloc[0]["run_id"])
print(f"\nArtifacts of run {runs.iloc[0]['run_id'][:12]}: {[a.path for a in arts]}")
