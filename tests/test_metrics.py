import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import catalog_coverage, intra_list_diversity, ranking_metrics, rating_metrics  # noqa: E402
from src.pipeline import topological_order  # noqa: E402


def _recs(user, items):
    return pd.DataFrame({"user_id": user, "item_id": items, "rank": range(1, len(items) + 1)})


def test_perfect_ranking():
    recs = _recs(1, [10, 11, 12, 13, 14])
    m = ranking_metrics(recs, {1: {10, 11, 12, 13, 14}}, k=5)
    assert m["precision_at_5"] == 1.0 and m["recall_at_5"] == 1.0 and abs(m["ndcg_at_5"] - 1.0) < 1e-9


def test_partial_ranking_ndcg():
    recs = _recs(1, [99, 10, 98, 97, 96])          # one hit at rank 2
    m = ranking_metrics(recs, {1: {10, 11}}, k=5)
    assert m["precision_at_5"] == 0.2 and m["recall_at_5"] == 0.5
    idcg = 1 + 1 / np.log2(3)
    assert abs(m["ndcg_at_5"] - (1 / np.log2(3)) / idcg) < 1e-9


def test_coverage_and_diversity():
    recs = pd.concat([_recs(1, [1, 2]), _recs(2, [1, 3])])
    assert catalog_coverage(recs, catalog_size=10, k=2) == 0.3
    genres = pd.DataFrame({"g_a": [1, 1, 0], "g_b": [0, 0, 1]}, index=[1, 2, 3])
    # user1: identical genres -> 0 ; user2: disjoint -> 1  => mean 0.5
    assert abs(intra_list_diversity(recs, genres, k=2) - 0.5) < 1e-9


def test_rating_metrics():
    r = rating_metrics(np.array([4, 2]), np.array([3, 2]))
    assert abs(r["rmse"] - np.sqrt(0.5)) < 1e-9 and r["mae"] == 0.5


def test_dag_order():
    assert topological_order() == ["ingest", "validate", "prepare", "transform", "features", "train", "evaluate"]
