"""Stage 2 – Data Validation.

Runs automated checks on the raw tables:
  missing values · duplicates · incorrect data types · invalid ratings ·
  invalid timestamps · referential integrity (user/item FK)

Rows violating an *error* rule are quarantined to ``data/errors/<table>_errors.csv``
with the reason(s). Rows with only *warning* issues pass through (logged to
``data/errors/<table>_warnings.csv``) and are repaired in the preparation stage.
A machine-readable JSON and a human-readable Markdown report go to ``reports/``.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from src.common.utils import banner, df_to_markdown, get_logger, load_config, path, timed, write_json
from src.validation.schemas import RULES, interactions_schema, items_schema, users_schema

STAGE = "validation"
KEYS = {"users": ["user_id"], "items": ["item_id", "title"], "interactions": ["user_id", "item_id", "rating", "timestamp"]}
NUMERIC = {"users": {"user_id": "Int64", "age": "Int64", "occupation": "Int64"},
           "items": {"item_id": "Int64"},
           "interactions": {"user_id": "Int64", "item_id": "Int64", "rating": "Float64", "timestamp": "Int64"}}


class Issues:
    """Collects (row, rule, column, value) findings for one table."""

    def __init__(self):
        self.by_row: dict[int, list[str]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, rows, rule: str, column: str):
        for r in rows:
            tag = f"{rule}({column})"
            if tag not in self.by_row[int(r)]:
                self.by_row[int(r)].append(tag)
                self.counts[rule] += 1

    def error_rows(self) -> set[int]:
        return {r for r, tags in self.by_row.items() if any(RULES[t.split("(")[0]][0] == "error" for t in tags)}

    def warning_rows(self) -> set[int]:
        return set(self.by_row) - self.error_rows()


def _coerce(raw: pd.DataFrame, table: str, issues: Issues) -> pd.DataFrame:
    """Type coercion – anything present-but-unparseable is a wrong_type error."""
    df = raw.copy()
    for col in KEYS[table]:
        issues.add(df.index[df[col].isna()], "missing_key", col)
    for col, dtype in NUMERIC[table].items():
        num = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & num.isna()
        if dtype == "Int64":
            frac = num.notna() & (num != num.round())
            bad |= frac
            num = num.where(~frac)
        issues.add(df.index[bad], "wrong_type", col)
        df[col] = num.astype(dtype)
    return df


def _run_pandera(df: pd.DataFrame, schema: pa.DataFrameSchema, issues: Issues, skip: set[int]) -> None:
    """Run the pandera contract lazily on rows that survived type/null checks."""
    sub = df.loc[~df.index.isin(skip)]
    try:
        schema.validate(sub, lazy=True)
    except SchemaErrors as err:
        fc = err.failure_cases
        fc = fc[fc["index"].notna() & fc["check_number"].notna()]
        for (column, num), grp in fc.groupby(["column", "check_number"]):
            # pandera prints built-in checks by their expression, so resolve the
            # rule name we attached to the check in schemas.py
            rule = schema.columns[column].checks[int(num)].name
            issues.add(grp["index"].astype(int).unique(), rule, str(column))


def _duplicates(df: pd.DataFrame, table: str, issues: Issues) -> None:
    exact = df.duplicated(keep="first")
    issues.add(df.index[exact], "duplicate_row", "*")
    if table == "interactions":
        key = df.duplicated(subset=["user_id", "item_id"], keep=False) & ~df.duplicated(keep=False)
        issues.add(df.index[key], "duplicate_interaction", "user_id,item_id")


def _write_outputs(cfg, table, raw, typed, issues, log) -> dict:
    err_rows, warn_rows = issues.error_rows(), issues.warning_rows()
    reasons = pd.Series({r: "; ".join(t) for r, t in issues.by_row.items()}, dtype=object)
    for name, rows in (("errors", err_rows), ("warnings", warn_rows)):
        out = raw.loc[sorted(rows)].copy()
        out.insert(0, "raw_row", out.index)
        out["severity"] = "error" if name == "errors" else "warning"
        out["reasons"] = reasons.reindex(out.index).values
        out.to_csv(path(cfg, "errors", f"{table}_{name}.csv"), index=False)
    valid = typed.loc[~typed.index.isin(err_rows)]
    valid.to_parquet(path(cfg, "validated", f"{table}.parquet"), index=False)
    summary = {"rows_in": len(raw), "rows_passed": len(valid), "rows_quarantined": len(err_rows),
               "rows_with_warnings": len(warn_rows), "pass_rate": round(len(valid) / len(raw), 6),
               "rule_counts": dict(sorted(issues.counts.items()))}
    log.info("[%s] in=%s passed=%s quarantined=%s warnings=%s pass_rate=%.4f%%", table, f"{len(raw):,}",
             f"{len(valid):,}", f"{len(err_rows):,}", f"{len(warn_rows):,}", 100 * summary["pass_rate"])
    for rule, n in sorted(issues.counts.items()):
        log.info("    %-22s %-8s %6d  – %s", rule, RULES[rule][0].upper(), n, RULES[rule][1])
    return summary


def validate_table(cfg, table, schema_fn, log, **kw) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path(cfg, "raw", f"{table}.csv"), dtype=str, keep_default_na=True)
    issues = Issues()
    typed = _coerce(raw, table, issues)
    if table == "users":
        typed["gender"] = typed["gender"].astype("string").str.strip().str.upper()
        for col in ["gender", "age", "occupation", "zipcode"]:
            issues.add(typed.index[typed[col].isna()], "missing_attribute", col)
    if table == "items":
        issues.add(typed.index[typed["genres"].isna()], "missing_attribute", "genres")
    _run_pandera(typed.astype({c: "object" for c in typed.columns if typed[c].dtype == "string"}),
                 schema_fn(cfg, **kw), issues, skip=issues.error_rows())
    _duplicates(typed, table, issues)
    return typed, _write_outputs(cfg, table, raw, typed, issues, log)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger(STAGE, cfg)
    banner(log, "2. DATA VALIDATION  (pandera contracts + custom checks)")
    report = {}
    with timed(log, "Validation"):
        users, report["users"] = validate_table(cfg, "users", users_schema, log)
        items, report["items"] = validate_table(cfg, "items", items_schema, log)
        valid_users = set(users["user_id"].dropna().astype(int))
        valid_items = set(items["item_id"].dropna().astype(int))
        _, report["interactions"] = validate_table(cfg, "interactions", interactions_schema, log,
                                                   user_ids=valid_users, item_ids=valid_items)
        report["detection_vs_injected"] = _detection_check(cfg, report, log)
        write_json(report, path(cfg, "reports", "validation_report.json"))
        _write_markdown(cfg, report)
        log.info("Validation report -> reports/validation_report.md ; quarantined rows -> data/errors/")
    return report


def _detection_check(cfg, report, log) -> dict:
    """Cross-check: every injected *error-type* defect must be caught."""
    p = path(cfg, "raw", "injected_defects_log.csv")
    if not p.exists():
        return {}
    inj = pd.read_csv(p)
    inj = inj[inj["entity"] == "interactions"]
    mapping = {"missing_value": "missing_key", "wrong_type": "wrong_type", "invalid_rating": "invalid_rating",
               "invalid_timestamp": "invalid_timestamp", "orphan_user": "orphan_user", "orphan_item": "orphan_item"}
    out = {}
    for defect, rule in mapping.items():
        injected = int((inj["defect"] == defect).sum())
        detected = int(report["interactions"]["rule_counts"].get(rule, 0))
        out[defect] = {"injected": injected, "detected": detected}
        log.info("    detection check %-18s injected=%4d detected=%4d %s", defect, injected, detected,
                 "OK" if detected >= injected else "MISSED")
    return out


def _write_markdown(cfg, report) -> None:
    lines = ["# Data Validation Report", "", "Generated by `src/validation/validate.py` (pandera + custom checks).", "",
             "## Summary", ""]
    summ = pd.DataFrame([{"table": t, **{k: v for k, v in s.items() if k != "rule_counts"}}
                         for t, s in report.items() if t != "detection_vs_injected"])
    summ["pass_rate"] = (summ["pass_rate"] * 100).map(lambda x: f"{x:.3f}%")
    lines += [df_to_markdown(summ), "", "## Rule violations", ""]
    rows = [{"table": t, "rule": r, "severity": RULES[r][0], "count": n, "description": RULES[r][1]}
            for t, s in report.items() if t != "detection_vs_injected" for r, n in s["rule_counts"].items()]
    lines += [df_to_markdown(pd.DataFrame(rows)), ""]
    if report.get("detection_vs_injected"):
        lines += ["## Detection check against injected defects (interactions)", ""]
        det = pd.DataFrame([{"defect": k, **v} for k, v in report["detection_vs_injected"].items()])
        lines += [df_to_markdown(det), "",
                  "Detected ≥ injected is expected: a re-rating duplicate or a defect on a row may trigger more than one rule.", ""]
    lines += ["## Outputs", "", "* Valid rows → `data/processed/validated/*.parquet`",
              "* Quarantined rows (severity = error) → `data/errors/<table>_errors.csv`",
              "* Rows passed with warnings → `data/errors/<table>_warnings.csv` (fixed in preparation)"]
    path(cfg, "reports", "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
