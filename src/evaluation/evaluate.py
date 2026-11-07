"""Stage 7 – Model Evaluation.

Evaluates every trained model on the held-out (future) test period:

* Ranking:   Precision@K, Recall@K, NDCG@K, HitRate@K, MAP@K   (K = 5 and 10)
* Catalogue: Coverage@K, intra-list Diversity@K (genre), Novelty@K
* Rating:    RMSE, MAE  (models that predict explicit ratings)

Metrics are logged back to each model's MLflow training run, and a comparison
table + charts are written to ``reports/``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.common.utils import banner, df_to_markdown, get_logger, load_config, path, read_json, timed, write_json
from src.evaluation.metrics import catalog_coverage, intra_list_diversity, novelty, ranking_metrics, rating_metrics
from src.training.context import load_context, load_model
from src.training.train import setup_mlflow

STAGE = "evaluation"
LABELS = {"popularity": "Popularity (baseline)", "item_knn": "Item-kNN CF", "als_mf": "ALS Matrix Factorisation",
          "hybrid": "Hybrid (CF + MF + features)"}


PRETTY = {"precision": "Precision", "recall": "Recall", "ndcg": "NDCG", "hit_rate": "HitRate", "map": "MAP",
          "coverage": "Coverage", "diversity": "Diversity", "novelty": "Novelty", "rmse": "RMSE", "mae": "MAE"}


def pretty(c: str) -> str:
    if c == "model":
        return "Model"
    base, _, k = c.partition("_at_")
    return PRETTY.get(base, base) + (f"@{k}" if k else "")


def evaluate_model(model, users, rel, test, G, pop, n_users, catalog, ks) -> tuple[dict, pd.DataFrame]:
    kmax = max(ks)
    recs = model.recommend(users, kmax)
    res = {}
    for k in ks:
        res.update(ranking_metrics(recs, rel, k))
        res[f"coverage_at_{k}"] = catalog_coverage(recs, catalog, k)
        res[f"diversity_at_{k}"] = intra_list_diversity(recs, G, k)
        res[f"novelty_at_{k}"] = novelty(recs, pop, n_users, k)
    if model.rating_model:
        res.update(rating_metrics(test["rating"].to_numpy(), model.predict_rating(test)))
    else:
        res.update({"rmse": np.nan, "mae": np.nan})
    return res, recs


def plot_comparison(df: pd.DataFrame, k: int, out) -> None:
    metrics = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", f"coverage_at_{k}", f"diversity_at_{k}"]
    colors = ["#9AA5B1", "#3B82F6", "#F59E0B", "#10B981"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.6))
    for ax, m in zip(axes, metrics):
        vals = df[m].to_numpy()
        ax.bar(range(len(df)), vals, color=colors[: len(df)])
        ax.set_title(pretty(m), fontsize=11)
        ax.set_xticks(range(len(df)), [x.split(" (")[0] for x in df["model"]], rotation=25, ha="right", fontsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle(f"Model comparison on the future test period (K = {k})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    log = get_logger(STAGE, cfg)
    banner(log, "7. MODEL EVALUATION  (P/R/NDCG@K, coverage, diversity, RMSE/MAE)")
    setup_mlflow(cfg)
    ctx = load_context(cfg)
    test = pd.read_parquet(path(cfg, "processed", "test.parquet"))
    thr, K = cfg["recommendation"]["relevance_threshold"], cfg["recommendation"]["k"]
    ks = sorted({5, K})
    rel = test[test["rating"] >= thr].groupby("user_id")["item_id"].apply(set).to_dict()
    users = np.array(sorted(test["user_id"].unique()))
    G = ctx.fs.get_item_features(None, ctx.fs.genre_cols)
    pop = ctx.fs.get_item_features(None, ["item_popularity"])["item_popularity"]
    registry = read_json(path(cfg, "models", "model_registry.json"))
    log.info("Test period: %s ratings | %d users (%d with >=1 relevant item, rating >= %.0f) | catalogue=%d items",
             f"{len(test):,}", len(users), len(rel), thr, len(ctx.item_ids))

    rows, all_recs = [], []
    with timed(log, "Evaluation"):
        for name, entry in registry.items():
            model = load_model(path(cfg, "models", f"{name}.pkl"), ctx)
            res, recs = evaluate_model(model, users, rel, test, G, pop, ctx.R.shape[0], len(ctx.item_ids), ks)
            with mlflow.start_run(run_id=entry["run_id"]):
                mlflow.log_metrics({k: v for k, v in res.items() if not (isinstance(v, float) and np.isnan(v))})
            rows.append({"model": LABELS[name], "key": name, **res})
            recs.insert(0, "model", name)
            all_recs.append(recs)
            log.info("%-30s P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f Cov=%.4f Div=%.4f RMSE=%s MAE=%s", LABELS[name],
                     K, res[f"precision_at_{K}"], K, res[f"recall_at_{K}"], K, res[f"ndcg_at_{K}"],
                     res[f"coverage_at_{K}"], res[f"diversity_at_{K}"],
                     "–" if np.isnan(res["rmse"]) else f"{res['rmse']:.4f}",
                     "–" if np.isnan(res["mae"]) else f"{res['mae']:.4f}")

    df = pd.DataFrame(rows)
    rep = path(cfg, "reports")
    df.drop(columns="key").to_csv(rep / "model_comparison.csv", index=False)
    write_json(df.to_dict(orient="records"), rep / "model_comparison.json")
    items = pd.read_parquet(path(cfg, "processed", "items.parquet"))[["item_id", "clean_title", "genres"]]
    sample_users = users[:3]
    sample = pd.concat(all_recs).query("user_id in @sample_users").merge(items, on="item_id")
    sample.sort_values(["model", "user_id", "rank"]).to_csv(rep / "sample_recommendations.csv", index=False)
    plot_comparison(df, K, rep / "figures" / "model_comparison.png")

    main = [f"precision_at_{K}", f"recall_at_{K}", f"ndcg_at_{K}", f"hit_rate_at_{K}", f"coverage_at_{K}",
            f"diversity_at_{K}", f"novelty_at_{K}", "rmse", "mae"]
    tbl = df[["model"] + main].rename(columns=pretty)
    tbl5 = df[["model"] + [f"precision_at_5", "recall_at_5", "ndcg_at_5", "coverage_at_5", "diversity_at_5"]].rename(columns=pretty)
    md = ["# Model Evaluation Results", "",
          f"Test set = ratings after the time cut-off ({read_json(path(cfg, 'processed', 'preparation_summary.json'))['split']['cutoff_utc'][:10]}), "
          f"warm users/items only: {len(test):,} ratings, {len(users):,} users. Relevant = rating ≥ {thr:.0f}. "
          f"Already-seen items are excluded from recommendations.", "",
          f"## Main results (K = {K})", "", df_to_markdown(tbl), "", "## K = 5", "", df_to_markdown(tbl5), "",
          "RMSE/MAE are reported for models that predict explicit ratings (popularity predicts the item's Bayesian-average rating).",
          "", "![comparison](figures/model_comparison.png)", ""]
    (rep / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")

    with mlflow.start_run(run_name="evaluation_summary"):
        mlflow.set_tags({"stage": "evaluation"})
        mlflow.log_artifact(str(rep / "model_comparison.csv"))
        mlflow.log_artifact(str(rep / "model_comparison.md"))
        mlflow.log_artifact(str(rep / "figures" / "model_comparison.png"))
        best = df.loc[df[f"ndcg_at_{K}"].idxmax()]
        mlflow.set_tag("best_model_by_ndcg", best["key"])
    log.info("Comparison table (K=%d):\n%s", K, tbl.set_index("Model").round(4).to_string())
    log.info("Best model by NDCG@%d: %s", K, best["model"])
    log.info("Reports -> reports/model_comparison.{md,csv,json}, reports/figures/model_comparison.png")
    return df


if __name__ == "__main__":
    run()
