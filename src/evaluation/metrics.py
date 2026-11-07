"""Ranking, catalogue and rating metrics for top-K recommendation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ranking_metrics(recs: pd.DataFrame, relevant: dict[int, set], k: int) -> dict:
    """Precision@K, Recall@K, NDCG@K, HitRate@K, MAP@K averaged over users with >=1 relevant item."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    p, r, n, h, ap = [], [], [], [], []
    top = recs[recs["rank"] <= k].sort_values(["user_id", "rank"])
    for uid, items in top.groupby("user_id")["item_id"]:
        rel = relevant.get(uid)
        if not rel:
            continue
        hits = np.fromiter((i in rel for i in items.to_numpy()[:k]), dtype=float, count=min(k, len(items)))
        n_hit = hits.sum()
        p.append(n_hit / k)
        r.append(n_hit / len(rel))
        idcg = discounts[: min(len(rel), k)].sum()
        n.append((hits * discounts[: len(hits)]).sum() / idcg)
        h.append(float(n_hit > 0))
        prec_at_i = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        ap.append((prec_at_i * hits).sum() / min(len(rel), k))
    return {f"precision_at_{k}": float(np.mean(p)), f"recall_at_{k}": float(np.mean(r)),
            f"ndcg_at_{k}": float(np.mean(n)), f"hit_rate_at_{k}": float(np.mean(h)),
            f"map_at_{k}": float(np.mean(ap)), "evaluated_users": len(p)}


def catalog_coverage(recs: pd.DataFrame, catalog_size: int, k: int) -> float:
    """Share of the catalogue that appears in at least one user's top-K list."""
    return recs.loc[recs["rank"] <= k, "item_id"].nunique() / catalog_size


def intra_list_diversity(recs: pd.DataFrame, item_genres: pd.DataFrame, k: int) -> float:
    """Mean pairwise genre dissimilarity (1 − Jaccard) inside each top-K list, averaged over users."""
    G = item_genres.to_numpy(dtype=np.float32)
    pos = pd.Index(item_genres.index).get_indexer
    top = recs[recs["rank"] <= k].sort_values(["user_id", "rank"])
    idx = pos(top["item_id"]).reshape(-1, k)
    L = G[idx]                                            # users x k x genres
    inter = np.einsum("ukg,ulg->ukl", L, L)
    sizes = L.sum(2)
    union = sizes[:, :, None] + sizes[:, None, :] - inter
    jac = np.where(union > 0, inter / np.maximum(union, 1e-9), 1.0)
    iu = np.triu_indices(k, 1)
    return float((1.0 - jac[:, iu[0], iu[1]]).mean())


def novelty(recs: pd.DataFrame, item_pop: pd.Series, n_users: int, k: int) -> float:
    """Mean self-information −log2(p(i)) of recommended items (higher = less obvious)."""
    top = recs[recs["rank"] <= k]
    p = item_pop.reindex(top["item_id"]).fillna(0).to_numpy() / n_users
    return float(np.mean(-np.log2(np.clip(p, 1e-6, 1))))


def rating_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}
