"""Stage 5 – Feature Management.

Materialises every feature declared in ``feature_registry.yaml`` into the
file-based feature store (``data/features``), checks that the produced tables
match the registry exactly, and generates the feature catalog
(``reports/feature_catalog.md``) with live statistics.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common.utils import banner, df_to_markdown, get_logger, load_config, md5, path, read_json, timed, write_json
from src.features.feature_store import FeatureStore

STAGE = "features"
REGISTRY = Path(__file__).with_name("feature_registry.yaml")


def build_user_features(cfg, users, uas, ugp, pop, train) -> pd.DataFrame:
    gcols = [c for c in ugp.columns if c.startswith("genre_")]
    share = ugp.set_index("user_id")[gcols]
    p = share.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(p > 0, p * np.log(p), 0)).sum(1) / np.log(len(gcols))
    pref = pd.Series(np.where(p.sum(1) > 0, np.array([c.replace("genre_", "") for c in gcols])[p.argmax(1)], "None"),
                     index=share.index)
    mainstream = train.merge(pop[["item_id", "popularity_pct"]], on="item_id").groupby("user_id")["popularity_pct"].mean()

    f = uas.set_index("user_id").rename(columns={
        "n_interactions": "user_activity_count", "mean_rating": "user_avg_rating", "std_rating": "user_rating_std",
        "pct_high_ratings": "user_pct_high_ratings", "active_days": "user_active_days",
        "tenure_days": "user_tenure_days", "days_since_last": "user_days_since_last_interaction"})
    f = f.drop(columns=["first_ts", "last_ts"])
    f["user_preferred_genre"] = pref
    f["user_genre_diversity"] = pd.Series(ent, index=share.index)
    f["user_mainstreaminess"] = mainstream
    demo = users.set_index("user_id")[["gender", "age_group", "occupation"]].rename(
        columns={"gender": "user_gender", "age_group": "user_age_group", "occupation": "user_occupation"})
    f = f.join(demo, how="left")
    f = f.join(share.add_prefix("user_genre_share_").rename(columns=lambda c: c.replace("share_genre_", "share_")))
    for c in ["user_gender", "user_age_group", "user_preferred_genre"]:
        f[c] = f[c].astype("string")
    return f.reset_index()


def build_item_features(cfg, items, pop, igm, cutoff_year) -> pd.DataFrame:
    p = pop.set_index("item_id").rename(columns={
        "n_ratings": "item_popularity", "popularity_rank": "item_popularity_rank", "popularity_pct": "item_popularity_pct",
        "mean_rating": "item_avg_rating", "bayes_rating": "item_bayes_rating", "pct_high_ratings": "item_pct_high_ratings",
        "days_since_last": "item_days_since_last_interaction"}).drop(columns=["first_ts", "last_ts"])
    base = items.set_index("item_id")[["release_year"]].rename(columns={"release_year": "item_release_year"})
    f = base.join(p, how="left")
    # items never rated in train (cold items): zero popularity, global-mean rating
    f["item_popularity"] = f["item_popularity"].fillna(0).astype(int)
    f["item_popularity_pct"] = f["item_popularity_pct"].fillna(0.0)
    f["item_popularity_rank"] = f["item_popularity_rank"].fillna(len(f)).astype(int)
    f["item_bayes_rating"] = f["item_bayes_rating"].fillna(pop["mean_rating"].mean())
    f["item_age_years"] = (cutoff_year - f["item_release_year"].astype("float")).clip(lower=0)
    g = igm.set_index("item_id")
    f["item_num_genres"] = g.sum(axis=1).astype(int)
    f = f.join(g)
    return f.reset_index()


def check_against_registry(registry, uf, if_, log) -> None:
    for feat in registry["features"]:
        name, ent = feat["name"], feat["entity"]
        if ent == "user_item":
            continue
        table = uf if ent == "user" else if_
        if "<genre>" in name:
            prefix = name.split("<")[0]
            ok = any(c.startswith(prefix) for c in table.columns)
        else:
            ok = name in table.columns
        if not ok:
            raise ValueError(f"Feature '{name}' declared in registry but not produced")
    log.info("Registry check passed: all %d declared features materialised", len(registry["features"]))


def write_catalog(cfg, registry, uf, if_, fs, train, log) -> None:
    rows = []
    sample = train.sample(2000, random_state=cfg["project"]["seed"])[["user_id", "item_id"]]
    uif = fs.get_user_item_features(sample)
    for feat in registry["features"]:
        name, ent = feat["name"], feat["entity"]
        t = {"user": uf, "item": if_, "user_item": uif}[ent]
        cols = [c for c in t.columns if c.startswith(name.split("<")[0])] if "<" in name else [name]
        s = t[cols[0]] if len(cols) == 1 else t[cols].stack()
        if pd.api.types.is_numeric_dtype(s):
            stat = f"min={s.min():.2f}, mean={s.mean():.2f}, max={s.max():.2f}"
        else:
            vc = s.value_counts()
            stat = f"{s.nunique()} levels; top='{vc.index[0]}' ({vc.iloc[0] / len(s):.0%})"
        rows.append({"Feature": f"`{name}`", "Entity": ent, "Type": feat["dtype"], "Description": feat["description"],
                     "Source / derivation": feat["source"], "Null %": f"{100 * s.isna().mean():.1f}", "Statistics": stat})
    cat = pd.DataFrame(rows)
    md = ["# Feature Catalog", "", f"*Registry version {registry['version']} · {registry['point_in_time']}*", "",
          f"Feature store: `data/features/` · cut-off: {fs.metadata['as_of_utc']} · user rows: {len(uf):,} · item rows: {len(if_):,}",
          "", "Served by `src/features/feature_store.py::FeatureStore` for training, evaluation and prediction.", "",
          df_to_markdown(cat), ""]
    path(cfg, "reports", "feature_catalog.md").write_text("\n".join(md), encoding="utf-8")
    cat.to_csv(path(cfg, "reports", "feature_catalog.csv"), index=False)
    log.info("Feature catalog (%d features) -> reports/feature_catalog.md", len(cat))


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger(STAGE, cfg)
    banner(log, "5. FEATURE MANAGEMENT  (registry → feature store → catalog)")
    with timed(log, "Feature build"):
        registry = yaml.safe_load(REGISTRY.read_text())
        proc, der, fdir = path(cfg, "processed"), path(cfg, "derived"), path(cfg, "features")
        users, items, train = (pd.read_parquet(proc / f) for f in ("users.parquet", "items.parquet", "train.parquet"))
        uas, pop, igm, ugp = (pd.read_parquet(der / f"{n}.parquet") for n in
                              ("user_activity_summary", "item_popularity", "item_genre_matrix", "user_genre_profile"))
        split = read_json(proc / "preparation_summary.json")["split"]
        cutoff_year = pd.to_datetime(split["cutoff_timestamp"], unit="s").year

        uf = build_user_features(cfg, users, uas, ugp, pop, train)
        if_ = build_item_features(cfg, items, pop, igm, cutoff_year)
        check_against_registry(registry, uf, if_, log)
        uf.to_parquet(fdir / "user_features.parquet", index=False)
        if_.to_parquet(fdir / "item_features.parquet", index=False)
        meta = {"registry_version": registry["version"], "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "as_of_timestamp": split["cutoff_timestamp"], "as_of_utc": split["cutoff_utc"],
                "user_features": {"rows": len(uf), "columns": len(uf.columns) - 1, "md5": md5(fdir / "user_features.parquet")},
                "item_features": {"rows": len(if_), "columns": len(if_.columns) - 1, "md5": md5(fdir / "item_features.parquet")}}
        write_json(meta, fdir / "feature_metadata.json")
        log.info("user_features : %s users x %s columns", f"{len(uf):,}", len(uf.columns) - 1)
        log.info("item_features : %s items x %s columns", f"{len(if_):,}", len(if_.columns) - 1)

        fs = FeatureStore(fdir)
        write_catalog(cfg, registry, uf, if_, fs, train, log)
        test = pd.read_parquet(proc / "test.parquet")
        demo_users = [int(u) for u in sorted(test["user_id"].unique())[:2]]
        demo_u = fs.get_user_features(demo_users, ["user_activity_count", "user_avg_rating", "user_preferred_genre",
                                                "user_days_since_last_interaction"])
        demo_i = fs.get_item_features([2858, 260], ["item_popularity", "item_avg_rating", "item_bayes_rating",
                                                    "item_popularity_rank"])
        demo_ui = fs.get_user_item_features(pd.DataFrame({"user_id": [demo_users[0]] * 2, "item_id": [2858, 260]}))
        log.info("FeatureStore.get_user_features(%s):\n%s", demo_users, demo_u.round(3).to_string())
        log.info("FeatureStore.get_item_features([2858, 260]):\n%s", demo_i.round(3).to_string())
        log.info("FeatureStore.get_user_item_features(...):\n%s", demo_ui.round(3).to_string(index=False))
    return meta


if __name__ == "__main__":
    run()
