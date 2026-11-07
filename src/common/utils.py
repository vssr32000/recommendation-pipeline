"""Shared helpers: config loading, logging, paths, JSON IO, timing."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


def path(cfg: dict, key: str, *parts: str) -> Path:
    """Resolve a configured directory (relative to project root) and create it."""
    p = PROJECT_ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p.joinpath(*parts) if parts else p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str, cfg: dict | None = None) -> logging.Logger:
    """Console + file logger. One log file per stage under logs/."""
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_dir = PROJECT_ROOT / ((cfg or {}).get("paths", {}).get("logs", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    _LOGGERS[name] = logger
    return logger


def write_json(obj: Any, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def read_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def timed(logger: logging.Logger, label: str):
    t0 = time.perf_counter()
    yield
    logger.info("%s finished in %.2fs", label, time.perf_counter() - t0)


def banner(logger: logging.Logger, title: str) -> None:
    line = "=" * 78
    logger.info(line)
    logger.info("STAGE: %s", title)
    logger.info(line)


def df_to_markdown(df, floatfmt: str = ".4f") -> str:
    """Small dependency-free markdown table renderer (avoids needing `tabulate`)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, float):
                cells.append("–" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
