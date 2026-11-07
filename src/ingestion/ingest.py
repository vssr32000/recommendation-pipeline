"""Stage 1 – Data Ingestion.

* Downloads the public MovieLens-1M dataset (users, movies, ratings) with retries,
  or reads it from a local folder when ``RECSYS_SOURCE_DIR`` is set (offline mode).
* Standardises column names to a common contract: ``users(user_id, ...)``,
  ``items(item_id, ...)``, ``interactions(user_id, item_id, rating, timestamp)``.
* Optionally injects a seeded, documented set of data-quality defects so the
  validation layer has something real to catch (see ``configs/pipeline.yaml``).
* Writes raw CSVs + an ingestion manifest (row counts, MD5 checksums, source,
  injected-defect log) to ``data/raw``.
"""
from __future__ import annotations

import os
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.utils import banner, get_logger, load_config, md5, path, set_seed, timed, write_json

STAGE = "ingestion"


# --------------------------------------------------------------------------- download
def _download(url: str, dest: Path, retries: int, log) -> None:
    for attempt in range(1, retries + 1):
        try:
            log.info("Downloading %s (attempt %d/%d)", url, attempt, retries)
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            return
        except Exception as exc:  # noqa: BLE001 – network errors vary by platform
            log.warning("Download failed: %s", exc)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def fetch_source(cfg: dict, log) -> dict[str, Path]:
    """Return local paths of the three source files (download once, then cached)."""
    src_cfg = cfg["ingestion"]
    cache = path(cfg, "raw", "source")
    cache.mkdir(parents=True, exist_ok=True)
    local_dir = os.environ.get("RECSYS_SOURCE_DIR")
    out = {}
    for entity, fname in src_cfg["files"].items():
        dest = cache / fname
        if local_dir:
            shutil.copy(Path(local_dir) / fname, dest)
            log.info("Copied %s from local source dir %s", fname, local_dir)
        elif dest.exists() and dest.stat().st_size > 0:
            log.info("Using cached source file %s", dest.relative_to(cfg["_root"]))
        else:
            _download(f"{src_cfg['source_url']}/{fname}", dest, src_cfg["download_retries"], log)
        out[entity] = dest
    return out


# --------------------------------------------------------------------------- standardise
def standardise(files: dict[str, Path]) -> dict[str, pd.DataFrame]:
    users = pd.read_csv(files["users"], sep="\t", index_col=0)
    items = pd.read_csv(files["items"], sep="\t", index_col=0, encoding="latin-1")
    inter = pd.read_csv(files["interactions"], sep="\t", index_col=0)
    users = users[["user_id", "gender", "age", "occupation", "zipcode"]].copy()
    items = items.rename(columns={"movie_id": "item_id"})[["item_id", "title", "genres"]].copy()
    inter = inter.rename(columns={"movie_id": "item_id"})[["user_id", "item_id", "rating", "timestamp"]].copy()
    return {"users": users, "items": items, "interactions": inter}


# --------------------------------------------------------------------------- defects
def inject_anomalies(dfs: dict[str, pd.DataFrame], spec: dict, rng: np.random.Generator) -> tuple[dict, list]:
    """Inject realistic defects. Returns modified frames + a log of what was injected."""
    log_rows: list[dict] = []

    def pick(df, n, used):
        avail = np.setdiff1d(np.arange(len(df)), np.fromiter(used, dtype=int, count=len(used)))
        idx = rng.choice(avail, size=n, replace=False)
        used.update(idx.tolist())
        return idx

    # ---- interactions ------------------------------------------------------
    df = dfs["interactions"].astype(object).reset_index(drop=True)
    s, used = spec["interactions"], set()
    for i in pick(df, s["missing_values"], used):
        col = rng.choice(["user_id", "item_id", "rating", "timestamp"])
        df.at[i, col] = np.nan
        log_rows.append({"entity": "interactions", "defect": "missing_value", "column": col, "row": int(i)})
    bad_type_values = {"rating": ["five", "4 stars", "***"], "timestamp": ["2000-13-45", "yesterday"], "user_id": ["u_12", "abc"]}
    for i in pick(df, s["wrong_types"], used):
        col = rng.choice(list(bad_type_values))
        df.at[i, col] = rng.choice(bad_type_values[col])
        log_rows.append({"entity": "interactions", "defect": "wrong_type", "column": col, "row": int(i)})
    for i in pick(df, s["invalid_ratings"], used):
        df.at[i, "rating"] = rng.choice([0, 6, 7, -1, 10, 2.5])
        log_rows.append({"entity": "interactions", "defect": "invalid_rating", "column": "rating", "row": int(i)})
    future = int(datetime(2031, 1, 1, tzinfo=timezone.utc).timestamp())
    for i in pick(df, s["invalid_timestamps"], used):
        df.at[i, "timestamp"] = rng.choice([future, 0, 315532800, -86400])
        log_rows.append({"entity": "interactions", "defect": "invalid_timestamp", "column": "timestamp", "row": int(i)})
    for i in pick(df, s["orphan_users"], used):
        df.at[i, "user_id"] = int(90000 + rng.integers(0, 999))
        log_rows.append({"entity": "interactions", "defect": "orphan_user", "column": "user_id", "row": int(i)})
    for i in pick(df, s["orphan_items"], used):
        df.at[i, "item_id"] = int(99000 + rng.integers(0, 999))
        log_rows.append({"entity": "interactions", "defect": "orphan_item", "column": "item_id", "row": int(i)})
    # duplicates: half exact copies, half re-ratings of the same (user, item) later in time
    dup_src = pick(df, s["duplicates"], used)
    dups = df.loc[dup_src].copy()
    half = len(dups) // 2
    rerate = dups.index[half:]
    dups.loc[rerate, "timestamp"] = dups.loc[rerate, "timestamp"].astype(int) + 3600
    dups.loc[rerate, "rating"] = rng.integers(1, 6, size=len(rerate))
    for k, i in enumerate(dup_src):
        log_rows.append({"entity": "interactions", "defect": "exact_duplicate" if k < half else "re_rating_duplicate", "column": "*", "row": int(i)})
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    dfs["interactions"] = df

    # ---- users -------------------------------------------------------------
    u, s, used = dfs["users"].astype(object).reset_index(drop=True), spec["users"], set()
    for i in pick(u, s["missing_values"], used):
        col = rng.choice(["gender", "age", "occupation", "zipcode"])
        u.at[i, col] = np.nan
        log_rows.append({"entity": "users", "defect": "missing_value", "column": col, "row": int(i)})
    for i in pick(u, s["invalid_age"], used):
        u.at[i, "age"] = rng.choice([0, 99, -5, 150])
        log_rows.append({"entity": "users", "defect": "invalid_age", "column": "age", "row": int(i)})
    dup_idx = pick(u, s["duplicates"], used)
    u = pd.concat([u, u.loc[dup_idx]], ignore_index=True)
    log_rows += [{"entity": "users", "defect": "exact_duplicate", "column": "*", "row": int(i)} for i in dup_idx]
    dfs["users"] = u

    # ---- items -------------------------------------------------------------
    it, s, used = dfs["items"].astype(object).reset_index(drop=True), spec["items"], set()
    for i in pick(it, s["missing_values"], used):
        it.at[i, "genres"] = np.nan
        log_rows.append({"entity": "items", "defect": "missing_value", "column": "genres", "row": int(i)})
    dup_idx = pick(it, s["duplicates"], used)
    it = pd.concat([it, it.loc[dup_idx]], ignore_index=True)
    log_rows += [{"entity": "items", "defect": "exact_duplicate", "column": "*", "row": int(i)} for i in dup_idx]
    dfs["items"] = it
    return dfs, log_rows


# --------------------------------------------------------------------------- entrypoint
def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger(STAGE, cfg)
    banner(log, "1. DATA INGESTION")
    set_seed(cfg["project"]["seed"])
    rng = np.random.default_rng(cfg["project"]["seed"])

    with timed(log, "Ingestion"):
        files = fetch_source(cfg, log)
        dfs = standardise(files)
        for name, df in dfs.items():
            log.info("Loaded source %-12s rows=%9s cols=%s", name, f"{len(df):,}", list(df.columns))

        injected = []
        if cfg["ingestion"]["inject_anomalies"]:
            dfs, injected = inject_anomalies(dfs, cfg["ingestion"]["anomalies"], rng)
            log.info("Injected %d documented data-quality defects (seed=%d) for validation testing",
                     len(injected), cfg["project"]["seed"])
            pd.DataFrame(injected).to_csv(path(cfg, "raw", "injected_defects_log.csv"), index=False)

        manifest = {"ingested_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source": cfg["ingestion"]["source_url"], "dataset": "MovieLens-1M (GroupLens)",
                    "anomalies_injected": len(injected), "tables": {}}
        for name, df in dfs.items():
            out = path(cfg, "raw", f"{name}.csv")
            df.to_csv(out, index=False)
            manifest["tables"][name] = {"file": str(out.relative_to(cfg["_root"])), "rows": len(df),
                                        "columns": list(df.columns), "md5": md5(out)}
            log.info("Wrote raw %-12s -> %s (%s rows, md5=%s)", name, out.relative_to(cfg["_root"]),
                     f"{len(df):,}", manifest["tables"][name]["md5"][:10])
        write_json(manifest, path(cfg, "raw", "ingestion_manifest.json"))
        log.info("Ingestion manifest written to data/raw/ingestion_manifest.json")
    return manifest


if __name__ == "__main__":
    run()
