"""Prediction / serving entry point – reuses the SAME feature store and model artefacts.

    python -m src.serving.recommend --user 635 --model hybrid --k 10

Cold-start users (not in the train period) fall back to the popularity model.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.common.utils import load_config, path
from src.training.context import load_context, load_model


def recommend(user_id: int, model_name: str = "hybrid", k: int = 10, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ctx = load_context(cfg)
    used = model_name if user_id in set(ctx.user_ids) else "popularity"
    model = load_model(path(cfg, "models", f"{used}.pkl"), ctx)
    if used == "popularity" and user_id not in set(ctx.user_ids):
        recs = pd.DataFrame({"item_id": ctx.item_ids[model.item_scores.argsort()[::-1][:k]]})
        recs["rank"], recs["user_id"] = range(1, k + 1), user_id
    else:
        recs = model.recommend([user_id], k)
    items = pd.read_parquet(path(cfg, "processed", "items.parquet"))[["item_id", "clean_title", "release_year", "genres"]]
    feats = ctx.fs.get_item_features(recs["item_id"], ["item_popularity", "item_bayes_rating"]).reset_index()
    ui = ctx.fs.get_user_item_features(recs[["user_id", "item_id"]]) if used != "popularity" else None
    out = recs.merge(items, on="item_id").merge(feats, on="item_id")
    if ui is not None:
        out = out.merge(ui[["item_id", "ui_genre_affinity"]], on="item_id")
    out.attrs["model_used"] = used
    return out.sort_values("rank")


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-K recommendations for one user")
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--model", default="hybrid", choices=["popularity", "item_knn", "als_mf", "hybrid"])
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config()
    ctx = load_context(cfg)
    if a.user in set(ctx.user_ids):
        prof = ctx.fs.get_user_features([a.user], ["user_activity_count", "user_avg_rating", "user_preferred_genre",
                                                   "user_mainstreaminess"])
        print("User profile (from feature store):")
        print(prof.round(3).to_string(), "\n")
    recs = recommend(a.user, a.model, a.k, cfg)
    print(f"Top-{a.k} recommendations for user {a.user} (model: {recs.attrs['model_used']}):")
    cols = [c for c in ["rank", "item_id", "clean_title", "release_year", "genres", "item_popularity",
                        "item_bayes_rating", "ui_genre_affinity"] if c in recs.columns]
    print(recs[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
