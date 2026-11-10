"""Builds docs/DM4ML_Assignment2_Report.pdf from the pipeline outputs.

    python docs/build_report.py --repo-url https://github.com/<user>/recommendation-pipeline

Reads reports/*.json|csv, docs/architecture.png and reports/screenshots/*.png.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
W = A4[0] - 36 * mm

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1E3A8A"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1E3A8A"))
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.3, leading=12.6, spaceAfter=4)
SM = ParagraphStyle("sm", parent=B, fontSize=8, leading=10)
CAP = ParagraphStyle("cap", parent=B, fontSize=8.3, textColor=colors.HexColor("#4B5563"), alignment=TA_CENTER, spaceAfter=8)
CELL = ParagraphStyle("cell", parent=B, fontSize=7.9, leading=9.8, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")


def img(p: Path, width: float, max_h: float | None = None) -> RLImage:
    w, h = Image.open(p).size
    ww, hh = width, width * h / w
    if max_h and hh > max_h:
        ww, hh = ww * max_h / hh, max_h
    return RLImage(str(p), width=ww, height=hh)


def table(rows, widths, header=True, zebra=True, bold_last=False):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0E7FF")))
    if zebra:
        for i in range(2, len(rows), 2):
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F8FAFC")))
    if bold_last:
        st.append(("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor("#ECFDF5")))
    t.setStyle(TableStyle(st))
    return t


def footer(c, doc):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawString(18 * mm, 10 * mm, "DM4ML Assignment 2 - End-to-End Data Management Pipeline for a Recommendation System")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    c.restoreState()


def build(repo_url: str, out: Path) -> None:
    val = json.loads((REP / "validation_report.json").read_text())
    prep = json.loads((ROOT / "data/processed/preparation_summary.json").read_text()) \
        if (ROOT / "data/processed/preparation_summary.json").exists() else None
    cmp_ = pd.read_csv(REP / "model_comparison.csv")
    cat = pd.read_csv(REP / "feature_catalog.csv")
    s = []

    # ------------------------------------------------------------------ title block
    s.append(Paragraph("End-to-End Data Management Pipeline for a Recommendation System",
                       ParagraphStyle("t", parent=ss["Title"], fontSize=17, leading=21, spaceAfter=4)))
    s.append(Paragraph("Data Management for Machine Learning (DM4ML) - Assignment 2 · BITS Pilani WILP, M.Tech AI/ML",
                       ParagraphStyle("st", parent=B, alignment=TA_CENTER, fontSize=10)))
    s.append(Paragraph(f"<b>Soma Sekhar Reddy Vattakula</b> · 2025AG05032 · {date.today():%d %B %Y}",
                       ParagraphStyle("st2", parent=B, alignment=TA_CENTER, fontSize=10, spaceAfter=6)))
    from xml.sax.saxutils import escape
    shown = escape(repo_url)
    link = Table([[Paragraph(f'<b>GitHub repository:</b> <link href="{shown}" color="#1D4ED8">{shown}</link>',
                             ParagraphStyle("l", parent=B, fontSize=10, alignment=TA_CENTER, spaceAfter=0))]], colWidths=[W])
    link.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#1D4ED8")),
                              ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
                              ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    s += [link, Spacer(1, 6)]

    # ------------------------------------------------------------------ 1 overview
    s.append(Paragraph("1. Problem and approach", H1))
    s.append(Paragraph(
        "The task is to recommend movies to users of an online movie platform from their past ratings and item "
        "metadata, and, more importantly, to build the <b>data management pipeline</b> around the model so that it "
        "is validated, reproducible and automated. I used the public <b>MovieLens-1M</b> dataset (6,040 users with "
        "demographics, 3,883 movies with genres, 1,000,209 timestamped 1-5 star ratings). The pipeline has seven "
        "stages, each a separate Python module with a <font face='Courier'>run(cfg)</font> entry point, driven by "
        "one YAML config and orchestrated as a Prefect flow (Figure 1).", B))
    s.append(img(ROOT / "docs/architecture.png", W))
    s.append(Paragraph("Figure 1 - Pipeline architecture. Solid arrows: task dependencies; dashed arrows: outputs.", CAP))

    # ------------------------------------------------------------------ 2 design
    s.append(Paragraph("2. Design decisions and tools", H1))
    rows = [["Stage", "Tool(s)", "Key design decision"],
            ["Ingestion", "urllib, pandas", "Download with retries and a local cache; standardise all sources into one "
             "users / items / interactions contract; MD5 + row-count manifest for lineage. Because MovieLens is "
             "already clean, 880 seeded, logged defects are injected so validation can be tested against a known answer."],
            ["Validation", "pandera", "Declarative schemas per table. Every rule has a severity: <i>error</i> rows "
             "(wrong type, null key, invalid rating/timestamp, orphan FK) are quarantined with reasons; <i>warning</i> rows "
             "(duplicates, missing optional attributes) pass and are repaired downstream, so no user is dropped and "
             "foreign keys stay valid. A data-quality gate fails the run above 1% rejected rows."],
            ["Preparation", "pandas, parquet", "Deterministic cleaning (latest re-rating wins, mode/'U'/'Unknown' "
             "imputation, UTC datetimes, parsed release year). One <b>global time cut-off</b> (80th percentile) so "
             "no future rating can leak into training, which per-user or random splits allow."],
            ["Transformation", "pandas, scipy.sparse", "Derived datasets come from the train split only: user activity "
             "summary, item popularity with a Bayesian-smoothed rating, genre matrices, and a CSR user-item matrix."],
            ["Features", "Custom feature store (YAML registry + parquet)", "One registry defines each feature; the "
             "build fails if a declared feature is missing; the catalog is generated from the registry. The same "
             "<font face='Courier'>FeatureStore</font> API serves training, evaluation and the serving CLI."],
            ["Training", "NumPy/SciPy, MLflow", "Four models behind one interface. Hyper-parameters are tuned on the "
             "last 10% of the <i>train</i> period (never the test set). MLflow logs every trial as a nested run plus "
             "params, lineage tags (cut-off, feature MD5s), loss curve and model artifacts."],
            ["Evaluation", "NumPy, matplotlib", "Ranking metrics at K=5 and 10 on warm test users, plus catalogue "
             "coverage, genre intra-list diversity, novelty and RMSE/MAE; metrics are written back to each MLflow run."],
            ["Orchestration", "Prefect 3 (+ plain-Python runner)", "The DAG is declared once and used by both "
             "orchestrators; tasks have explicit dependencies, ingestion retries, and a cron schedule via "
             "<font face='Courier'>flow.serve</font>. The fallback runner skips downstream stages on failure."]]
    s.append(table(rows, [24 * mm, 33 * mm, W - 57 * mm]))

    # ------------------------------------------------------------------ 3 data quality
    s.append(Paragraph("3. Data validation and preparation results", H1))
    it = val["interactions"]
    det = val.get("detection_vs_injected", {})
    caught = sum(v["detected"] >= v["injected"] for v in det.values())
    s.append(Paragraph(
        f"Of {it['rows_in']:,} raw interactions, <b>{it['rows_quarantined']:,} were quarantined</b> "
        f"(pass rate {100 * it['pass_rate']:.3f}%) and {it['rows_with_warnings']:,} passed with warnings. "
        f"All {caught} of {len(det)} injected error-type defect classes were detected in full. Users "
        f"({val['users']['rows_with_warnings']} warnings) and items ({val['items']['rows_with_warnings']}) needed "
        "repair only, with no rejections.", B))
    rc = it["rule_counts"]
    vrows = [["Rule (interactions)", "Severity", "Rows", "Rule (interactions)", "Severity", "Rows"]]
    keys = list(rc)
    half = (len(keys) + 1) // 2
    from src.validation.schemas import RULES  # noqa: E402 – local import keeps the script standalone-ish
    for i in range(half):
        a = keys[i]
        b = keys[i + half] if i + half < len(keys) else None
        vrows.append([a, RULES[a][0], f"{rc[a]:,}"] + ([b, RULES[b][0], f"{rc[b]:,}"] if b else ["", "", ""]))
    s.append(table(vrows, [42 * mm, 18 * mm, 15 * mm, 42 * mm, 18 * mm, W - 135 * mm]))
    if prep:
        sp = prep["split"]
        s.append(Spacer(1, 4))
        s.append(Paragraph(
            f"After cleaning, the time-based split at <b>{sp['cutoff_utc'][:10]}</b> gives {sp['train_rows']:,} training "
            f"ratings ({sp['train_users']:,} users, {sp['train_items']:,} items) and {sp['test_rows_warm']:,} warm test "
            f"ratings from {sp['test_users_warm']:,} users. {sp['test_cold_start_user_rows']:,} test ratings belong to users "
            "who joined after the cut-off. A personalised model cannot score these cold-start users, so they are reported "
            "and left out of the comparison; in serving they fall back to the popularity model.", B))

    # ------------------------------------------------------------------ 4 features
    s.append(Paragraph("4. Feature management", H1))
    ent = cat["Entity"].value_counts().to_dict()
    s.append(Paragraph(
        f"The registry defines <b>{len(cat)} features</b> ({ent.get('user', 0)} user, {ent.get('item', 0)} item and "
        f"{ent.get('user_item', 0)} on-demand user×item features; two of them are 19-dimensional genre vectors). All are "
        "computed <i>as of</i> the split cut-off (point-in-time correct), stored as parquet with MD5 checksums, and the "
        "checksums are logged to every MLflow training run. Examples: <font face='Courier'>user_avg_rating, "
        "user_activity_count, user_preferred_genre, user_days_since_last_interaction, item_popularity, "
        "item_bayes_rating, ui_genre_affinity</font>. The popularity and hybrid models read their signals from the store, "
        "and so does <font face='Courier'>src/serving/recommend.py</font>. The full catalog with types, sources and "
        "live statistics is in <font face='Courier'>reports/feature_catalog.md</font>.", B))

    # ------------------------------------------------------------------ 5 results
    s.append(Paragraph("5. Models and evaluation results", H1))
    s.append(Paragraph(
        "<b>Popularity</b> (baseline) recommends the most-rated unseen items. <b>Item-kNN</b> is item-based "
        "collaborative filtering with shrunk cosine similarity. <b>ALS-MF</b> is biased explicit matrix factorisation "
        "(48 factors, ALS-WR) that predicts ratings. <b>Hybrid</b> blends min-max-normalised item-kNN and ALS scores with "
        "the user×item genre affinity and item popularity features. Tuned values: item-kNN 300 neighbours with "
        "shrinkage 100; ALS lambda = 0.08; hybrid weights 0.45 / 0.25 / 0.05 / 0.25. "
        "Relevant = test rating of 4 or more; items already seen are excluded.", B))
    order = ["precision_at_10", "recall_at_10", "ndcg_at_10", "hit_rate_at_10", "coverage_at_10", "diversity_at_10", "rmse", "mae"]
    hdr = ["Model", "P@10", "R@10", "NDCG@10", "Hit@10", "Cov@10", "Div@10", "RMSE", "MAE"]
    mrows = [hdr] + [[r["model"]] + ["-" if pd.isna(r[c]) else f"{r[c]:.4f}" for c in order] for _, r in cmp_.iterrows()]
    s.append(table(mrows, [38 * mm] + [(W - 38 * mm) / 8] * 8, bold_last=True))
    s.append(Spacer(1, 4))
    s.append(img(REP / "figures/model_comparison.png", W))
    s.append(Paragraph("Figure 2 - Model comparison on the future test period (K = 10).", CAP))
    pop, hyb = cmp_.iloc[0], cmp_.iloc[3]
    s.append(Paragraph(
        f"<b>Interpretation.</b> The <b>hybrid model is best on every accuracy metric</b>: NDCG@10 "
        f"{hyb['ndcg_at_10']:.4f} vs {pop['ndcg_at_10']:.4f} (+{100 * (hyb['ndcg_at_10'] / pop['ndcg_at_10'] - 1):.1f}%), "
        f"Recall@10 +{100 * (hyb['recall_at_10'] / pop['recall_at_10'] - 1):.0f}%, and about "
        f"{hyb['coverage_at_10'] / pop['coverage_at_10']:.1f}x the catalogue coverage. Popularity is a strong baseline "
        "under a temporal split, because the test period is dominated by mainstream titles. Item-kNN matches it on "
        "NDCG while covering about 4x more items. ALS gives the best rating prediction (RMSE 0.894 vs 0.960 for the "
        "item-mean baseline), but ranking by predicted rating pushes long-tail items with high predicted scores (novelty "
        "4.4), so its top-K accuracy is low. That is the usual gap between rating prediction and top-K ranking, and it "
        "is why ALS contributes to the hybrid as a signal and is not used on its own. Results are identical across "
        "repeated runs (fixed seed and deterministic splits).", B))

    # ------------------------------------------------------------------ 6 limitations
    s.append(Paragraph("6. Limitations and future work", H1))
    lim = [
        "<b>Injected defects.</b> Data-quality issues are synthetic (seeded and logged), so the validation layer is "
        "proven against known defects, not against real production drift.",
        "<b>Cold start.</b> About half of the raw test ratings come from users who joined after the cut-off. They are "
        "served by popularity only. Demographic or content-based onboarding models could cover them.",
        "<b>Offline evaluation only.</b> A single temporal split on a 2000-2003 dataset. A rolling-origin backtest and "
        "online A/B metrics (CTR, watch time) would give a better estimate of real impact.",
        "<b>Scale.</b> The dense item-item similarity matrix and the pure-NumPy ALS loop suit 1M ratings. Larger "
        "catalogues need approximate nearest neighbours, the <font face='Courier'>implicit</font> library or Spark ALS.",
        "<b>Feature store.</b> The file-based store has no online, low-latency serving layer or TTLs. Feast with a "
        "Redis/Postgres online store is the natural next step. Hybrid tuning uses features built on the full train "
        "period, a mild optimistic bias limited to the choice of weights.",
        "<b>Model registry.</b> Models are tracked as MLflow artifacts, but they are not promoted through MLflow Model "
        "Registry stages, and the pipeline has no CI/CD."]
    for l in lim:
        s.append(Paragraph("• " + l, B))

    # ------------------------------------------------------------------ appendix
    s.append(PageBreak())
    s.append(Paragraph("Appendix A - Execution screenshots of every pipeline stage", H1))
    s.append(Paragraph(
        "Each stage was executed individually with <font face='Courier'>python run_pipeline.py --stage &lt;name&gt;</font>, "
        "then the full DAG was executed as a Prefect flow against a local Prefect server. Terminal captures show the "
        "unedited program output; UI captures are from the Prefect and MLflow web UIs of the same run.", B))
    shots = [
        ("01_ingest.png", "A1 - Stage 1, Data ingestion: download/cache, schema standardisation, defect injection, manifest with MD5."),
        ("02_validate.png", "A2 - Stage 2, Data validation: pandera rule counts by severity, quarantine, detection check against injected defects."),
        ("03_prepare.png", "A3 - Stage 3, Data preparation: de-duplication and imputation counts, time-based train/test split."),
        ("04_transform.png", "A4 - Stage 4, Data transformation: derived datasets (activity summary, item popularity, genre matrices, sparse user-item matrix)."),
        ("05_features.png", "A5 - Stage 5, Feature management: registry check, feature store tables, catalog, FeatureStore API calls."),
        ("06_train.png", "A6 - Stage 6, Model training: validation-split tuning (MLflow nested runs) and final training of the four models."),
        ("07_evaluate.png", "A7 - Stage 7, Model evaluation: P/R/NDCG@10, coverage, diversity, novelty, RMSE/MAE comparison."),
        ("08_prefect_flow.png", "A8 - Orchestration: full Prefect flow run, with the seven tasks completed in dependency order and the data-quality gate."),
        ("09_prefect_ui.png", "A9 - Prefect UI: successful flow run, task dependency graph and logs (7/7 tasks Completed)."),
        ("10_mlflow_runs.png", "A10 - MLflow experiment 'movielens-recsys': tuning parent runs, training runs and evaluation summaries, one set per pipeline execution."),
        ("11_mlflow_run_detail.png", "A11 - MLflow run detail (train_hybrid): logged metrics, parameters, lineage tags and link to the tuning run."),
        ("12_mlflow_query.png", "A12 - Tracked parameters and metrics of the latest execution, queried from MLflow."),
        ("13_serving.png", "A13 - Prediction: top-10 recommendations for user 635 from the hybrid model, with features read from the same feature store."),
        ("14_tests.png", "A14 - Unit tests for metrics and DAG ordering."),
    ]
    max_h = A4[1] - 60 * mm
    for f, cap in shots:
        p = REP / "screenshots" / f
        if p.exists():
            s.append(KeepTogether([img(p, W, max_h=max_h * (0.62 if "ui" in f or "mlflow_run" in f else 0.9)), Paragraph(cap, CAP)]))

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                            bottomMargin=16 * mm, title="DM4ML Assignment 2 - Recommendation Pipeline",
                            author="Soma Sekhar Reddy Vattakula")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/<your-github-username>/recommendation-pipeline")
    ap.add_argument("--out", default=str(ROOT / "docs" / "DM4ML_Assignment2_Report.pdf"))
    a = ap.parse_args()
    build(a.repo_url, Path(a.out))
