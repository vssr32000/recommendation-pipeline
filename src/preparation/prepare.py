"""Stage 3 – Data Preparation.

* Cleans validated data: de-duplicates, imputes missing / invalid attributes,
  standardises formats (types, genre names, zip codes, UTC datetimes, release year).
* Produces a **time-based** train/test split (no future information leaks into
  training): by default a single global cut-off at the 80th time percentile.
* Writes clean tables + split to ``data/processed`` and a preparation summary.
"""
from __future__ import annotations

import pandas as pd

from src.common.utils import banner, get_logger, load_config, path, timed, write_json

STAGE = "preparation"
AGE_DESC = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
GENRE_FIX = {"Children's": "Children", "Film-Noir": "Film-Noir", "Sci-Fi": "Sci-Fi"}


def clean_users(df: pd.DataFrame, cfg: dict, stats: dict) -> pd.DataFrame:
    v = cfg["validation"]
    n0 = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="user_id", keep="first").copy()
    stats["users_duplicates_removed"] = n0 - len(df)
    bad_gender = ~df["gender"].isin(v["valid_genders"])
    df.loc[bad_gender, "gender"] = "U"
    bad_age = ~df["age"].isin(v["valid_ages"])
    mode_age = int(df.loc[~bad_age, "age"].mode().iloc[0])
    df.loc[bad_age, "age"] = mode_age
    bad_occ = df["occupation"].isna() | ~df["occupation"].between(0, 20)
    df.loc[bad_occ, "occupation"] = 0  # 0 = "other / not specified" in MovieLens
    df["zipcode"] = df["zipcode"].fillna("00000").astype(str).str.extract(r"(\d{1,5})", expand=False).fillna("00000").str.zfill(5)
    stats.update(users_gender_imputed=int(bad_gender.sum()), users_age_imputed=int(bad_age.sum()),
                 users_occupation_imputed=int(bad_occ.sum()), users_age_mode=mode_age)
    df = df.astype({"user_id": "int32", "age": "int16", "occupation": "int16", "gender": "string"})
    df["age_group"] = df["age"].map(AGE_DESC)
    return df.sort_values("user_id").reset_index(drop=True)


def clean_items(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="item_id", keep="first").copy()
    stats["items_duplicates_removed"] = n0 - len(df)
    miss = df["genres"].isna()
    df.loc[miss, "genres"] = "Unknown"
    stats["items_genres_imputed"] = int(miss.sum())
    df["genres"] = df["genres"].map(lambda g: "|".join(sorted(GENRE_FIX.get(x, x) for x in str(g).split("|"))))
    df["release_year"] = pd.to_numeric(df["title"].str.extract(r"\((\d{4})\)\s*$", expand=False), errors="coerce").astype("Int16")
    df["clean_title"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
    # "Matrix, The" -> "The Matrix"
    df["clean_title"] = df["clean_title"].str.replace(r"^(.*), (The|A|An)$", r"\2 \1", regex=True)
    return df.astype({"item_id": "int32"}).sort_values("item_id").reset_index(drop=True)


def clean_interactions(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates()
    stats["interactions_exact_duplicates_removed"] = n0 - len(df)
    n1 = len(df)
    df = df.sort_values("timestamp").drop_duplicates(subset=["user_id", "item_id"], keep="last")
    stats["interactions_rerating_duplicates_resolved"] = n1 - len(df)
    df = df.astype({"user_id": "int32", "item_id": "int32", "rating": "float32", "timestamp": "int64"})
    df["event_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, cfg: dict, log) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    p = cfg["preparation"]
    frac = p["test_fraction"]
    if p["split_strategy"] == "global_time":
        cutoff = int(df["timestamp"].quantile(1 - frac))
        train, test = df[df["timestamp"] < cutoff], df[df["timestamp"] >= cutoff]
    else:  # per_user_time – last `frac` of each user's history goes to test
        rank = df.groupby("user_id")["timestamp"].rank(method="first", pct=True)
        train, test = df[rank <= 1 - frac], df[rank > 1 - frac]
        cutoff = int(train["timestamp"].max()) + 1
    # keep train users with enough history to learn from
    counts = train["user_id"].value_counts()
    keep = counts[counts >= p["min_user_train_interactions"]].index
    train = train[train["user_id"].isin(keep)]
    raw_test = len(test)
    cold_users = ~test["user_id"].isin(train["user_id"].unique())
    cold_items = ~test["item_id"].isin(train["item_id"].unique())
    test_warm = test[~cold_users & ~cold_items]
    meta = {"strategy": p["split_strategy"], "cutoff_timestamp": cutoff,
            "cutoff_utc": str(pd.to_datetime(cutoff, unit="s", utc=True)),
            "train_rows": len(train), "test_rows_raw": raw_test, "test_rows_warm": len(test_warm),
            "test_cold_start_user_rows": int(cold_users.sum()), "test_cold_start_item_rows": int((cold_items & ~cold_users).sum()),
            "train_users": int(train["user_id"].nunique()), "train_items": int(train["item_id"].nunique()),
            "test_users_warm": int(test_warm["user_id"].nunique()), "test_items_warm": int(test_warm["item_id"].nunique()),
            "train_period": [str(train["event_time"].min()), str(train["event_time"].max())],
            "test_period": [str(test_warm["event_time"].min()), str(test_warm["event_time"].max())]}
    log.info("Time-based split (%s): cutoff=%s", meta["strategy"], meta["cutoff_utc"])
    log.info("  train: %s rows | %s users | %s items | %s → %s", f"{len(train):,}", f"{meta['train_users']:,}",
             f"{meta['train_items']:,}", meta["train_period"][0][:10], meta["train_period"][1][:10])
    log.info("  test : %s rows (warm) | %s users | %s items | cold-start rows excluded: users=%s items=%s",
             f"{len(test_warm):,}", f"{meta['test_users_warm']:,}", f"{meta['test_items_warm']:,}",
             f"{meta['test_cold_start_user_rows']:,}", f"{meta['test_cold_start_item_rows']:,}")
    return train, test_warm, meta


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger(STAGE, cfg)
    banner(log, "3. DATA PREPARATION  (cleaning, standardisation, time-based split)")
    stats: dict = {}
    with timed(log, "Preparation"):
        vdir = path(cfg, "validated")
        users = clean_users(pd.read_parquet(vdir / "users.parquet"), cfg, stats)
        items = clean_items(pd.read_parquet(vdir / "items.parquet"), stats)
        inter = clean_interactions(pd.read_parquet(vdir / "interactions.parquet"), stats)
        for k, v in stats.items():
            log.info("  %-45s %s", k, v)
        log.info("Clean tables: users=%s items=%s interactions=%s", f"{len(users):,}", f"{len(items):,}", f"{len(inter):,}")
        train, test, meta = time_split(inter, cfg, log)
        assert train["timestamp"].max() < test["timestamp"].min() or meta["strategy"] != "global_time", "temporal leakage!"
        out = path(cfg, "processed")
        users.to_parquet(out / "users.parquet", index=False)
        items.to_parquet(out / "items.parquet", index=False)
        inter.to_parquet(out / "interactions.parquet", index=False)
        train.to_parquet(out / "train.parquet", index=False)
        test.to_parquet(out / "test.parquet", index=False)
        summary = {"cleaning": stats, "split": meta}
        write_json(summary, out / "preparation_summary.json")
        log.info("Wrote users/items/interactions/train/test parquet to data/processed/")
    return summary


if __name__ == "__main__":
    run()
