"""Builds the TrainingContext (rating matrix + id maps + feature store) shared by all models."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.utils import path
from src.features.feature_store import FeatureStore
from src.training.models import BaseRecommender, HybridRecommender, TrainingContext


def load_context(cfg: dict) -> TrainingContext:
    der = path(cfg, "derived")
    return TrainingContext(
        R=sp.load_npz(der / "user_item_matrix.npz").tocsr(),
        user_ids=pd.read_parquet(der / "user_index.parquet")["user_id"].to_numpy(),
        item_ids=pd.read_parquet(der / "item_index.parquet")["item_id"].to_numpy(),
        fs=FeatureStore(path(cfg, "features")),
        seed=cfg["project"]["seed"],
    )


def context_from_frame(df: pd.DataFrame, fs: FeatureStore, seed: int) -> TrainingContext:
    """Context from an arbitrary interactions frame (used for the validation split during tuning)."""
    uids, iids = np.sort(df["user_id"].unique()), np.sort(df["item_id"].unique())
    R = sp.csr_matrix((df["rating"].to_numpy(np.float32),
                       (np.searchsorted(uids, df["user_id"]), np.searchsorted(iids, df["item_id"]))),
                      shape=(len(uids), len(iids)))
    return TrainingContext(R=R, user_ids=uids, item_ids=iids, fs=fs, seed=seed)


def save_model(model: BaseRecommender, p: Path) -> None:
    with open(p, "wb") as fh:
        pickle.dump(model, fh)


def load_model(p: Path, ctx: TrainingContext) -> BaseRecommender:
    with open(p, "rb") as fh:
        m = pickle.load(fh)
    return m.attach(ctx) if isinstance(m, HybridRecommender) else _attach(m, ctx)


def _attach(m, ctx):
    m.ctx = ctx
    return m
