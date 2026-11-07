"""A lightweight, file-based feature store (custom feature layer).

The same object serves features to model training, offline evaluation and
online-style prediction (``src/serving/recommend.py``), so there is exactly one
definition of each feature (train/serve consistency).

Layout under ``data/features/``::

    user_features.parquet   item_features.parquet   feature_metadata.json
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.utils import read_json


class FeatureStore:
    def __init__(self, feature_dir: str | Path):
        self.dir = Path(feature_dir)
        self.metadata = read_json(self.dir / "feature_metadata.json")

    # ------------------------------------------------------------------ tables
    @cached_property
    def user_table(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "user_features.parquet").set_index("user_id")

    @cached_property
    def item_table(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "item_features.parquet").set_index("item_id")

    @property
    def genre_cols(self) -> list[str]:
        return [c for c in self.item_table.columns if c.startswith("genre_")]

    @property
    def user_genre_cols(self) -> list[str]:
        return [c for c in self.user_table.columns if c.startswith("user_genre_share_")]

    # ------------------------------------------------------------------ getters
    def get_user_features(self, user_ids=None, features: list[str] | None = None) -> pd.DataFrame:
        t = self.user_table if user_ids is None else self.user_table.reindex(pd.Index(user_ids, name="user_id"))
        return t if features is None else t[features]

    def get_item_features(self, item_ids=None, features: list[str] | None = None) -> pd.DataFrame:
        t = self.item_table if item_ids is None else self.item_table.reindex(pd.Index(item_ids, name="item_id"))
        return t if features is None else t[features]

    def genre_affinity_matrix(self, user_ids, item_ids) -> np.ndarray:
        """Dense |users| x |items| cosine similarity of user genre profile vs item genres."""
        U = self.get_user_features(user_ids, self.user_genre_cols).fillna(0.0).to_numpy(np.float32)
        I = self.get_item_features(item_ids, self.genre_cols).fillna(0).to_numpy(np.float32)
        Un = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
        In = I / np.maximum(np.linalg.norm(I, axis=1, keepdims=True), 1e-9)
        return Un @ In.T

    def get_user_item_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """On-demand interaction features for arbitrary (user_id, item_id) pairs."""
        u = self.get_user_features(pairs["user_id"].values)
        i = self.get_item_features(pairs["item_id"].values)
        U = u[self.user_genre_cols].fillna(0.0).to_numpy(np.float32)
        I = i[self.genre_cols].fillna(0).to_numpy(np.float32)
        num = (U * I).sum(1)
        den = np.linalg.norm(U, axis=1) * np.linalg.norm(I, axis=1)
        out = pairs[["user_id", "item_id"]].copy()
        out["ui_genre_affinity"] = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
        out["ui_rating_gap"] = i["item_bayes_rating"].to_numpy() - u["user_avg_rating"].to_numpy()
        out["ui_popularity_vs_user"] = i["item_popularity_pct"].to_numpy() - u["user_mainstreaminess"].to_numpy()
        return out
