"""Stage 6 – Model Training (with MLflow experiment tracking).

1. Hyper-parameter tuning on a validation split carved from the END of the
   train period (time-respecting; the test set is never touched). Every trial is
   logged as a nested MLflow run under ``hyperparameter_tuning``.
2. Final models are refit on the full train split with the selected parameters:
     popularity (baseline) · item_knn · als_mf · hybrid
   Each gets its own MLflow run with parameters, data lineage tags, training
   time, the ALS loss curve and the pickled model as an artifact.
3. ``models/model_registry.json`` records model path + MLflow run id per model.
"""
from __future__ import annotations

import itertools
import time

import mlflow
import numpy as np
import pandas as pd

from src.common.utils import banner, get_logger, load_config, path, read_json, set_seed, timed, write_json
from src.evaluation.metrics import ranking_metrics, rating_metrics
from src.training.context import context_from_frame, load_context, save_model
from src.training.models import (ALSMatrixFactorization, HybridRecommender, ItemKNNRecommender,
                                 PopularityRecommender)

STAGE = "training"


def setup_mlflow(cfg: dict) -> None:
    root = cfg["_root"]
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{root}/{uri.removeprefix('sqlite:///')}"
    mlflow.set_tracking_uri(uri)
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(path(cfg, "models").parent / "mlruns").as_uri())
    mlflow.set_experiment(name)


# --------------------------------------------------------------------------- tuning
def _val_split(cfg, train: pd.DataFrame):
    frac = cfg["tuning"]["validation_fraction"]
    cut = train["timestamp"].quantile(1 - frac)
    sub, val = train[train["timestamp"] < cut], train[train["timestamp"] >= cut]
    val = val[val["user_id"].isin(sub["user_id"]) & val["item_id"].isin(sub["item_id"])]
    return sub, val


def tune(cfg, train, fs, log) -> dict:
    t = cfg["tuning"]
    k, thr, metric = cfg["recommendation"]["k"], cfg["recommendation"]["relevance_threshold"], t["metric"]
    sub, val = _val_split(cfg, train)
    ctx = context_from_frame(sub, fs, cfg["project"]["seed"])
    rel = val[val["rating"] >= thr].groupby("user_id")["item_id"].apply(set).to_dict()
    users = np.array(sorted(rel))
    log.info("Tuning on validation split: sub-train=%s rows, validation=%s rows, %d users",
             f"{len(sub):,}", f"{len(val):,}", len(users))
    trials = []
    with mlflow.start_run(run_name="hyperparameter_tuning") as parent:
        mlflow.set_tags({"stage": "tuning", "selection_metric": metric})
        # ---- item-kNN grid
        best_knn, best_knn_score, knn_models = None, -1, {}
        for nb, sh in itertools.product(t["item_knn_grid"]["neighbours"], t["item_knn_grid"]["shrinkage"]):
            m = ItemKNNRecommender(neighbours=nb, shrinkage=sh).fit(ctx)
            score = ranking_metrics(m.recommend(users, k), rel, k)[metric]
            trials.append({"model": "item_knn", "params": f"neighbours={nb}, shrinkage={sh}", metric: score})
            with mlflow.start_run(run_name=f"item_knn_nb{nb}_sh{sh}", nested=True):
                mlflow.log_params({"model": "item_knn", "neighbours": nb, "shrinkage": sh})
                mlflow.log_metric(f"val_{metric}", score)
            log.info("  item_knn  neighbours=%-4d shrinkage=%-4d val_%s=%.4f", nb, sh, metric, score)
            knn_models[(nb, sh)] = m
            if score > best_knn_score:
                best_knn, best_knn_score = {"neighbours": nb, "shrinkage": sh}, score
        # ---- ALS regularisation grid (selected on validation RMSE)
        base = cfg["models"]["als_mf"]
        best_reg, best_rmse, als_models = None, np.inf, {}
        val_als = val[val["user_id"].isin(ctx.user_ids)]
        for reg in t["als_reg_grid"]:
            m = ALSMatrixFactorization(**{**base, "reg": reg}).fit(ctx)
            rmse = rating_metrics(val_als["rating"].to_numpy(), m.predict_rating(val_als))["rmse"]
            trials.append({"model": "als_mf", "params": f"reg={reg}", "val_rmse": rmse})
            with mlflow.start_run(run_name=f"als_mf_reg{reg}", nested=True):
                mlflow.log_params({"model": "als_mf", **base, "reg": reg})
                mlflow.log_metric("val_rmse", rmse)
            log.info("  als_mf    reg=%-5s                val_rmse=%.4f", reg, rmse)
            als_models[reg] = m
            if rmse < best_rmse:
                best_reg, best_rmse = reg, rmse
        # ---- hybrid weight grid (re-uses the best components)
        best_w, best_w_score = None, -1
        comps = {"item_knn": knn_models[(best_knn["neighbours"], best_knn["shrinkage"])], "als_mf": als_models[best_reg]}
        for i, w in enumerate(t["hybrid_weight_grid"]):
            m = HybridRecommender(comps, weights=w).fit(ctx)
            score = ranking_metrics(m.recommend(users, k), rel, k)[metric]
            trials.append({"model": "hybrid", "params": ", ".join(f"{a}={b}" for a, b in w.items()), metric: score})
            with mlflow.start_run(run_name=f"hybrid_w{i}", nested=True):
                mlflow.log_params({"model": "hybrid", **{f"w_{a}": b for a, b in w.items()}})
                mlflow.log_metric(f"val_{metric}", score)
            log.info("  hybrid    %s val_%s=%.4f", w, metric, score)
            if score > best_w_score:
                best_w, best_w_score = w, score
        best = {"item_knn": best_knn, "als_mf": {**base, "reg": best_reg}, "hybrid": {"weights": best_w}}
        mlflow.log_dict(best, "best_params.json")
        tr = pd.DataFrame(trials)
        tr.to_csv(path(cfg, "reports", "tuning_results.csv"), index=False)
        mlflow.log_artifact(str(path(cfg, "reports", "tuning_results.csv")))
        log.info("Selected: item_knn=%s | als_mf reg=%s | hybrid weights=%s", best_knn, best_reg, best_w)
        best["_tuning_run_id"] = parent.info.run_id
    return best


# --------------------------------------------------------------------------- final fit
def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger(STAGE, cfg)
    banner(log, "6. MODEL TRAINING  (tuning on validation split + MLflow tracking)")
    set_seed(cfg["project"]["seed"])
    setup_mlflow(cfg)
    ctx = load_context(cfg)
    train = pd.read_parquet(path(cfg, "processed", "train.parquet"))
    split = read_json(path(cfg, "processed", "preparation_summary.json"))["split"]
    fmeta = ctx.fs.metadata
    lineage = {"data.train_rows": len(train), "data.split_cutoff": split["cutoff_utc"],
               "data.n_users": ctx.R.shape[0], "data.n_items": ctx.R.shape[1],
               "features.registry_version": fmeta["registry_version"],
               "features.user_md5": fmeta["user_features"]["md5"], "features.item_md5": fmeta["item_features"]["md5"]}

    params = {"popularity": dict(cfg["models"]["popularity"]), "item_knn": dict(cfg["models"]["item_knn"]),
              "als_mf": dict(cfg["models"]["als_mf"]), "hybrid": dict(cfg["models"]["hybrid"])}
    tuning_run = None
    if cfg["tuning"]["enabled"]:
        with timed(log, "Hyper-parameter tuning"):
            best = tune(cfg, train, ctx.fs, log)
        tuning_run = best.pop("_tuning_run_id")
        for m in best:
            params[m].update(best[m])

    registry, fitted = {}, {}
    builders = {"popularity": lambda p: PopularityRecommender(**p),
                "item_knn": lambda p: ItemKNNRecommender(**p),
                "als_mf": lambda p: ALSMatrixFactorization(**p),
                "hybrid": lambda p: HybridRecommender({"item_knn": fitted["item_knn"], "als_mf": fitted["als_mf"]}, **p)}
    with timed(log, "Final model training"):
        for name, build in builders.items():
            with mlflow.start_run(run_name=f"train_{name}") as run:
                mlflow.set_tags({"stage": "training", "model": name, "dataset": "MovieLens-1M",
                                 "model_type": "baseline" if name == "popularity" else "personalised",
                                 **({"tuning_run_id": tuning_run} if tuning_run else {})})
                flat = {f"w_{k}": v for k, v in params[name].get("weights", {}).items()} if name == "hybrid" else params[name]
                mlflow.log_params({**flat, **lineage})
                t0 = time.perf_counter()
                model = build(params[name]).fit(ctx)
                dur = time.perf_counter() - t0
                mlflow.log_metric("train_seconds", dur)
                if name == "als_mf":
                    for step, rmse in enumerate(model.history):
                        mlflow.log_metric("train_rmse", rmse, step=step)
                p = path(cfg, "models", f"{name}.pkl")
                save_model(model, p)
                mlflow.log_artifact(str(p), artifact_path="model")
                mlflow.log_artifact(str(path(cfg, "features", "feature_metadata.json")), artifact_path="lineage")
                fitted[name] = model
                registry[name] = {"path": str(p.relative_to(cfg["_root"])), "run_id": run.info.run_id,
                                  "params": params[name], "train_seconds": round(dur, 2)}
                extra = f" final_train_rmse={model.history[-1]:.4f}" if name == "als_mf" else ""
                log.info("Trained %-10s in %6.2fs  params=%s  run_id=%s%s", name, dur, params[name],
                         run.info.run_id[:12], extra)
    write_json(registry, path(cfg, "models", "model_registry.json"))
    log.info("Model registry -> models/model_registry.json ; MLflow experiment '%s'", cfg["mlflow"]["experiment"])
    return registry


if __name__ == "__main__":
    run()
