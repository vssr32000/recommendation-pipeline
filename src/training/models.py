"""Recommender models.

All models share one interface so training, evaluation and serving are generic:

    fit(ctx)                       -> self
    score(user_ids) -> ndarray     (|users| x |catalog| preference scores)
    recommend(user_ids, k)         -> DataFrame[user_id, item_id, rank, score]
    predict_rating(pairs)          -> ndarray  (only for explicit-rating models)

Models
------
* PopularityRecommender – non-personalised baseline (most-rated items, from the feature store)
* ItemKNNRecommender    – item-based collaborative filtering (shrunk cosine, top-N neighbours)
* ALSMatrixFactorization– explicit biased matrix factorisation trained with ALS-WR (predicts ratings)
* HybridRecommender     – weighted blend of item-kNN, ALS, genre affinity and popularity (feature store)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.features.feature_store import FeatureStore


@dataclass
class TrainingContext:
    R: sp.csr_matrix              # users x items rating matrix (train)
    user_ids: np.ndarray          # u_idx -> user_id
    item_ids: np.ndarray          # i_idx -> item_id
    fs: FeatureStore
    seed: int = 42

    def u_idx(self, user_ids) -> np.ndarray:
        idx = np.searchsorted(self.user_ids, user_ids)
        idx = np.clip(idx, 0, len(self.user_ids) - 1)
        if not np.all(self.user_ids[idx] == np.asarray(user_ids)):
            raise KeyError("unknown (cold-start) user id passed to a personalised model")
        return idx


class BaseRecommender:
    name = "base"
    rating_model = False

    def __init__(self, **params):
        self.params = params
        self.ctx: TrainingContext | None = None

    def fit(self, ctx: TrainingContext):
        self.ctx = ctx
        return self

    def score(self, user_ids) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_ids, k: int = 10, exclude_seen: bool = True, batch: int = 512) -> pd.DataFrame:
        user_ids = np.asarray(user_ids)
        out = []
        for s in range(0, len(user_ids), batch):
            ub = user_ids[s:s + batch]
            sc = self.score(ub).astype(np.float64)
            if exclude_seen:
                seen = self.ctx.R[self.ctx.u_idx(ub)]
                sc[seen.nonzero()] = -np.inf
            top = np.argpartition(-sc, k, axis=1)[:, :k]
            top_sc = np.take_along_axis(sc, top, 1)
            order = np.argsort(-top_sc, axis=1)
            top, top_sc = np.take_along_axis(top, order, 1), np.take_along_axis(top_sc, order, 1)
            out.append(pd.DataFrame({"user_id": np.repeat(ub, k), "item_id": self.ctx.item_ids[top.ravel()],
                                     "rank": np.tile(np.arange(1, k + 1), len(ub)), "score": top_sc.ravel()}))
        return pd.concat(out, ignore_index=True)

    def predict_rating(self, pairs: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not predict explicit ratings")

    def __getstate__(self):  # keep pickles small: the context is rebuilt at load time
        d = self.__dict__.copy()
        d["ctx"] = None
        return d


# --------------------------------------------------------------------------- baseline
class PopularityRecommender(BaseRecommender):
    """Recommends the most-rated items the user has not seen (same list for everyone)."""
    name = "popularity"
    rating_model = True  # predicts the item's Bayesian-average rating

    def fit(self, ctx):
        super().fit(ctx)
        f = ctx.fs.get_item_features(ctx.item_ids, ["item_popularity", "item_bayes_rating"])
        pop = f["item_popularity"].to_numpy(np.float64)
        pop[pop < self.params.get("min_ratings", 0)] = 0.0
        self.item_scores = pop / pop.max()
        self.item_rating = f["item_bayes_rating"].to_numpy(np.float64)
        self.global_mean = float(ctx.R.data.mean())
        return self

    def score(self, user_ids):
        return np.tile(self.item_scores, (len(user_ids), 1))

    def predict_rating(self, pairs):
        idx = pd.Index(self.ctx.item_ids).get_indexer(pairs["item_id"])
        return np.where(idx >= 0, self.item_rating[idx], self.global_mean)


# --------------------------------------------------------------------------- item-kNN CF
class ItemKNNRecommender(BaseRecommender):
    """Item-based CF: score(u, i) = sum_j in history(u) sim(i, j), shrunk cosine, top-N neighbours."""
    name = "item_knn"

    def fit(self, ctx):
        super().fit(ctx)
        X = (ctx.R > 0).astype(np.float32).tocsc()
        co = (X.T @ X).toarray()                       # item co-occurrence
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.params.get("shrinkage", 0) + 1e-9)
        np.fill_diagonal(sim, 0.0)
        n = self.params.get("neighbours", 100)
        if n < sim.shape[0]:                           # keep only the top-n neighbours per item
            thr = -np.partition(-sim, n - 1, axis=0)[n - 1]
            sim[sim < thr[None, :]] = 0.0
        self.S = sp.csr_matrix(sim.astype(np.float32))
        return self

    def score(self, user_ids):
        X = (self.ctx.R[self.ctx.u_idx(user_ids)] > 0).astype(np.float32)
        return np.asarray((X @ self.S).todense())


# --------------------------------------------------------------------------- ALS MF
class ALSMatrixFactorization(BaseRecommender):
    """Biased explicit MF: r_ui ≈ mu + b_u + b_i + p_u·q_i, solved with ALS-WR (λ scaled by #ratings)."""
    name = "als_mf"
    rating_model = True

    def fit(self, ctx):
        super().fit(ctx)
        rng = np.random.default_rng(ctx.seed)
        k, lam, lam_b, iters = (self.params[p] for p in ("factors", "reg", "bias_reg", "iterations"))
        R = ctx.R.tocsr()
        Rc = R.tocsc()
        nu, ni = R.shape
        self.mu = float(R.data.mean())
        # --- biases (a few alternating ridge updates) --------------------------
        bu, bi = np.zeros(nu), np.zeros(ni)
        rows, cols = R.nonzero()
        vals = np.asarray(R[rows, cols]).ravel()
        for _ in range(5):
            bi = np.bincount(cols, vals - self.mu - bu[rows], ni) / (np.bincount(cols, minlength=ni) + lam_b)
            bu = np.bincount(rows, vals - self.mu - bi[cols], nu) / (np.bincount(rows, minlength=nu) + lam_b)
        self.bu, self.bi = bu, bi
        resid = sp.csr_matrix((vals - self.mu - bu[rows] - bi[cols], (rows, cols)), shape=R.shape)
        resid_c = resid.tocsc()
        # --- ALS on residuals ---------------------------------------------------
        P = rng.normal(0, 0.1, (nu, k))
        Q = rng.normal(0, 0.1, (ni, k))
        I = np.eye(k)
        self.history = []
        for it in range(iters):
            for u in range(nu):
                s, e = resid.indptr[u], resid.indptr[u + 1]
                if e == s:
                    continue
                Qi, r = Q[resid.indices[s:e]], resid.data[s:e]
                P[u] = np.linalg.solve(Qi.T @ Qi + lam * (e - s) * I, Qi.T @ r)
            for i in range(ni):
                s, e = resid_c.indptr[i], resid_c.indptr[i + 1]
                if e == s:
                    continue
                Pu, r = P[resid_c.indices[s:e]], resid_c.data[s:e]
                Q[i] = np.linalg.solve(Pu.T @ Pu + lam * (e - s) * I, Pu.T @ r)
            pred = (P[rows] * Q[cols]).sum(1) + self.mu + bu[rows] + bi[cols]
            rmse = float(np.sqrt(np.mean((vals - pred) ** 2)))
            self.history.append(rmse)
        self.P, self.Q = P.astype(np.float32), Q.astype(np.float32)
        return self

    def score(self, user_ids):
        u = self.ctx.u_idx(user_ids)
        return self.mu + self.bu[u, None] + self.bi[None, :] + self.P[u] @ self.Q.T

    def predict_rating(self, pairs):
        u = self.ctx.u_idx(pairs["user_id"].to_numpy())
        i = pd.Index(self.ctx.item_ids).get_indexer(pairs["item_id"])
        known = i >= 0
        pred = np.full(len(pairs), self.mu)
        ii = i[known]
        pred[known] = self.mu + self.bu[u[known]] + self.bi[ii] + (self.P[u[known]] * self.Q[ii]).sum(1)
        return np.clip(pred, 1.0, 5.0)


# --------------------------------------------------------------------------- hybrid
def _minmax_rows(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(1, keepdims=True), x.max(1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-9)


class HybridRecommender(BaseRecommender):
    """Collaborative + content + popularity blend; content/popularity signals come from the feature store."""
    name = "hybrid"

    def __init__(self, components: dict[str, BaseRecommender], **params):
        super().__init__(**params)
        self.components = components

    def fit(self, ctx):
        super().fit(ctx)
        for c in self.components.values():
            if c.ctx is None:
                c.fit(ctx)
        self.pop = ctx.fs.get_item_features(ctx.item_ids, ["item_popularity_pct"])["item_popularity_pct"].to_numpy()
        return self

    def attach(self, ctx):
        self.ctx = ctx
        for c in self.components.values():
            c.ctx = ctx
        return self

    def score(self, user_ids):
        w = self.params["weights"]
        s = w["item_knn"] * _minmax_rows(self.components["item_knn"].score(user_ids))
        s += w["als_mf"] * _minmax_rows(self.components["als_mf"].score(user_ids))
        s += w["genre_affinity"] * self.ctx.fs.genre_affinity_matrix(user_ids, self.ctx.item_ids)
        s += w["popularity"] * self.pop[None, :]
        return s
