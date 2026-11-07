"""Stage 4 – Data Transformation.

Builds reusable derived datasets from the TRAIN split only (so nothing from the
test period leaks into features or models):

* ``user_activity_summary``  – per-user activity & rating behaviour
* ``item_popularity``        – per-item counts, mean & Bayesian-smoothed rating, rank
* ``item_genre_matrix``      – one-hot genre encoding of every item
* ``user_genre_profile``     – share of each user's ratings per genre
* ``user_item_matrix``       – sparse CSR rating matrix + id↔index mappings
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.utils import banner, get_logger, load_config, path, timed, write_json

STAGE = "transformation"


def user_activity_summary(train: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = pd.DataFrame({
        "n_interactions": g.size(),
        "mean_rating": g["rating"].mean(),
        "std_rating": g["rating"].std().fillna(0.0),
        "pct_high_ratings": g["rating"].apply(lambda r: (r >= 4).mean()),
        "first_ts": g["timestamp"].min(),
        "last_ts": g["timestamp"].max(),
        "active_days": g["event_time"].apply(lambda t: t.dt.normalize().nunique()),
    })
    out["tenure_days"] = (out["last_ts"] - out["first_ts"]) / 86400
    out["days_since_last"] = (cutoff - out["last_ts"]) / 86400
    return out.reset_index()


def item_popularity(train: pd.DataFrame, cutoff: int, m: int) -> pd.DataFrame:
    g = train.groupby("item_id")
    out = pd.DataFrame({"n_ratings": g.size(), "mean_rating": g["rating"].mean(),
                        "pct_high_ratings": g["rating"].apply(lambda r: (r >= 4).mean()),
                        "last_ts": g["timestamp"].max(), "first_ts": g["timestamp"].min()})
    c = train["rating"].mean()
    # IMDb-style Bayesian average: shrinks items with few ratings toward the global mean
    out["bayes_rating"] = (out["n_ratings"] * out["mean_rating"] + m * c) / (out["n_ratings"] + m)
    out["popularity_rank"] = out["n_ratings"].rank(ascending=False, method="min").astype(int)
    out["popularity_pct"] = out["n_ratings"].rank(pct=True)
    out["days_since_last"] = (cutoff - out["last_ts"]) / 86400
    return out.reset_index().sort_values("popularity_rank")


def item_genre_matrix(items: pd.DataFrame) -> pd.DataFrame:
    oh = items["genres"].str.get_dummies(sep="|")
    oh.columns = [f"genre_{c.replace('-', '_').replace(' ', '_')}" for c in oh.columns]
    return pd.concat([items[["item_id"]], oh.astype("int8")], axis=1)


def user_genre_profile(train: pd.DataFrame, igm: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in igm.columns if c.startswith("genre_")]
    liked = train[train["rating"] >= 4][["user_id", "item_id"]]
    j = liked.merge(igm, on="item_id", how="left")
    prof = j.groupby("user_id")[gcols].sum()
    prof = prof.div(prof.sum(axis=1).replace(0, 1), axis=0)  # share of liked-genre mass
    all_users = pd.Index(train["user_id"].unique(), name="user_id")
    return prof.reindex(all_users, fill_value=0.0).reset_index()


def user_item_matrix(train: pd.DataFrame) -> tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame]:
    uids = np.sort(train["user_id"].unique())
    iids = np.sort(train["item_id"].unique())
    u_index = pd.DataFrame({"user_id": uids, "u_idx": np.arange(len(uids))})
    i_index = pd.DataFrame({"item_id": iids, "i_idx": np.arange(len(iids))})
    rows = np.searchsorted(uids, train["user_id"].values)
    cols = np.searchsorted(iids, train["item_id"].values)
    mat = sp.csr_matrix((train["rating"].values.astype(np.float32), (rows, cols)), shape=(len(uids), len(iids)))
    return mat, u_index, i_index


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger(STAGE, cfg)
    banner(log, "4. DATA TRANSFORMATION  (derived datasets from the train split)")
    with timed(log, "Transformation"):
        proc = path(cfg, "processed")
        train = pd.read_parquet(proc / "train.parquet")
        items = pd.read_parquet(proc / "items.parquet")
        cutoff = int(pd.read_json(proc / "preparation_summary.json", typ="series")["split"]["cutoff_timestamp"])
        out = path(cfg, "derived")

        uas = user_activity_summary(train, cutoff)
        pop = item_popularity(train, cutoff, cfg["models"]["popularity"]["min_ratings"])
        igm = item_genre_matrix(items)
        ugp = user_genre_profile(train, igm)
        mat, u_index, i_index = user_item_matrix(train)

        uas.to_parquet(out / "user_activity_summary.parquet", index=False)
        pop.to_parquet(out / "item_popularity.parquet", index=False)
        igm.to_parquet(out / "item_genre_matrix.parquet", index=False)
        ugp.to_parquet(out / "user_genre_profile.parquet", index=False)
        sp.save_npz(out / "user_item_matrix.npz", mat)
        u_index.to_parquet(out / "user_index.parquet", index=False)
        i_index.to_parquet(out / "item_index.parquet", index=False)

        density = mat.nnz / (mat.shape[0] * mat.shape[1])
        meta = {"user_activity_summary": list(uas.shape), "item_popularity": list(pop.shape),
                "item_genre_matrix": list(igm.shape), "user_genre_profile": list(ugp.shape),
                "user_item_matrix": {"shape": list(mat.shape), "nnz": int(mat.nnz), "density": density}}
        write_json(meta, out / "derived_manifest.json")
        log.info("user_activity_summary : %s rows x %s cols", *uas.shape)
        log.info("item_popularity       : %s rows x %s cols  (top item: %s, %s ratings)", *pop.shape,
                 items.set_index("item_id").loc[pop.iloc[0]["item_id"], "title"], int(pop.iloc[0]["n_ratings"]))
        log.info("item_genre_matrix     : %s rows x %s cols  (%d genres)", *igm.shape, igm.shape[1] - 1)
        log.info("user_genre_profile    : %s rows x %s cols", *ugp.shape)
        log.info("user_item_matrix      : %s x %s sparse CSR, nnz=%s, density=%.2f%%", mat.shape[0], mat.shape[1],
                 f"{mat.nnz:,}", 100 * density)
        log.info("Sample user_activity_summary:\n%s", uas.head(3).round(2).to_string(index=False))
        log.info("Top-5 item_popularity:\n%s", pop.head(5).merge(items[["item_id", "clean_title"]], on="item_id")
                 [["item_id", "clean_title", "n_ratings", "mean_rating", "bayes_rating", "popularity_rank"]].round(3).to_string(index=False))
    return meta


if __name__ == "__main__":
    run()
