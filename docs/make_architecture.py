"""Draws docs/architecture.png (run: python docs/make_architecture.py)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STAGES = [
    ("1 · Ingestion", "download + retry\nstandardise schema\ninject documented\ndefects · MD5 manifest"),
    ("2 · Validation", "pandera contracts\ntypes · nulls · ranges\ntimestamps · FK\nintegrity · duplicates"),
    ("3 · Preparation", "de-duplicate · impute\nstandardise formats\nglobal time-based\nsplit (80 / 20)"),
    ("4 · Transformation", "user activity summary\nitem popularity\ngenre matrices\nsparse user×item matrix"),
    ("5 · Features", "registry YAML →\nfeature store (parquet)\nuser · item · user×item\npoint-in-time"),
    ("6 · Training", "tuning on validation\nsplit · Popularity\nItem-kNN · ALS-MF\nHybrid"),
    ("7 · Evaluation", "P / R / NDCG @5,10\nCoverage · Diversity\nNovelty\nRMSE / MAE"),
]
STAGE_C, STAGE_E = "#EEF2FF", "#4F46E5"
STORE_C, STORE_E = "#ECFDF5", "#059669"

fig, ax = plt.subplots(figsize=(17, 7.2))
ax.set_xlim(0, 17)
ax.set_ylim(0, 7.2)
ax.axis("off")


def box(x, y, w, h, title, body, fc, ec, round_=0.12, fs=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={round_}", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10.5, weight="bold", color="#111827")
    if body:
        ax.text(x + w / 2, y + h - 0.7, body, ha="center", va="top", fontsize=fs, color="#374151", linespacing=1.35)


def arrow(p, q, style="-|>", ls="-", color="#374151", rad=0.0, label=None, lpos=0.5, loff=(0, 0)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=1.3, ls=ls, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        x = p[0] + (q[0] - p[0]) * lpos + loff[0]
        y = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(x, y, label, fontsize=8.5, color="#4B5563", ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=1.5))


# orchestration frame
ax.add_patch(FancyBboxPatch((0.25, 2.55), 16.5, 3.05, boxstyle="round,pad=0.02,rounding_size=0.2",
                            fc="#FFFBEB", ec="#D97706", lw=1.2, ls="--"))
ax.text(2.0, 5.42, "Orchestration — Prefect flow  dags/recommendation_flow.py   (dependency-free fallback: run_pipeline.py · cron / flow.serve schedule)",
        fontsize=10, color="#92400E", va="center", weight="bold")

w, h, gap, x0, y0 = 2.05, 2.25, 0.30, 0.45, 2.85
centers = []
for i, (t, b) in enumerate(STAGES):
    x = x0 + i * (w + gap)
    box(x, y0, w, h, t, b, STAGE_C, STAGE_E)
    centers.append(x + w / 2)
    if i:
        arrow((x - gap, y0 + h / 2), (x, y0 + h / 2))

# source
box(0.45, 6.15, 2.6, 0.85, "MovieLens-1M (public)", None, STORE_C, STORE_E)
ax.text(1.75, 6.38, "users · movies · ratings", ha="center", fontsize=9, color="#374151")
arrow((centers[0], 6.15), (centers[0], y0 + h))

# stores below
stores = [("data/errors/", "quarantined rows", 1.0, [1]),
          ("reports/", "validation report · feature catalog\ncomparison table · charts", 4.6, [1, 4, 6]),
          ("MLflow tracking (SQLite)", "params · metrics · artifacts", 9.4, [5, 6]),
          ("Serving  src/serving/recommend.py", "FeatureStore + models/*.pkl", 13.1, [4, 5])]
for title, sub, sx, srcs in stores:
    sw = 3.4
    box(sx, 0.3, sw, 1.25, title, sub, STORE_C if "Serving" not in title else "#F5F3FF",
        STORE_E if "Serving" not in title else "#7C3AED", fs=8.6)
    for j, s in enumerate(srcs):
        ex = sx + sw * (j + 1) / (len(srcs) + 1)
        arrow((centers[s], y0), (ex, 1.55), ls=(0, (3, 2)), color="#6B7280")

fig.tight_layout()
out = Path(__file__).with_name("architecture.png")
fig.savefig(out, dpi=170, facecolor="white")
print("wrote", out)
