"""Pandera schemas – the data contract for each raw table.

Each check is tagged with a *severity*:
  * ``error``   → the row is quarantined to ``data/errors`` and never reaches training
  * ``warning`` → the row passes, the issue is logged and fixed in the preparation stage
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pandera.pandas as pa

# (check-name → (severity, human-readable rule)) – used to build the validation report
RULES: dict[str, tuple[str, str]] = {
    "wrong_type": ("error", "Value cannot be parsed into the expected data type"),
    "missing_key": ("error", "Required key / measure is null"),
    "invalid_rating": ("error", "Rating must be an integer in [rating_min, rating_max]"),
    "invalid_timestamp": ("error", "Timestamp must lie between min_timestamp and ingestion time"),
    "orphan_user": ("error", "user_id must exist in the users table (referential integrity)"),
    "orphan_item": ("error", "item_id must exist in the items table (referential integrity)"),
    "duplicate_row": ("warning", "Exact duplicate record – de-duplicated in preparation"),
    "duplicate_interaction": ("warning", "Same (user_id, item_id) rated more than once – latest kept in preparation"),
    "missing_attribute": ("warning", "Optional attribute is null – imputed in preparation"),
    "invalid_age": ("warning", "Age not in the MovieLens age-bucket domain – imputed in preparation"),
    "invalid_gender": ("warning", "Gender not in {M, F} – mapped to 'U' in preparation"),
    "invalid_occupation": ("warning", "Occupation code outside 0–20 – imputed in preparation"),
}


def interactions_schema(cfg: dict, user_ids, item_ids) -> pa.DataFrameSchema:
    v = cfg["validation"]
    min_ts = int(pd.Timestamp(v["min_timestamp"], tz="UTC").timestamp())
    max_ts = int(datetime.now(timezone.utc).timestamp())
    return pa.DataFrameSchema(
        {
            "user_id": pa.Column("Int64", nullable=False, checks=[
                pa.Check.isin(set(user_ids), name="orphan_user")]),
            "item_id": pa.Column("Int64", nullable=False, checks=[
                pa.Check.isin(set(item_ids), name="orphan_item")]),
            "rating": pa.Column("Float64", nullable=False, checks=[
                pa.Check.in_range(v["rating_min"], v["rating_max"], name="invalid_rating"),
                pa.Check(lambda s: s == s.round(), element_wise=False, name="invalid_rating")]),
            "timestamp": pa.Column("Int64", nullable=False, checks=[
                pa.Check.in_range(min_ts, max_ts, name="invalid_timestamp")]),
        },
        strict=True, coerce=False, name="interactions",
    )


def users_schema(cfg: dict) -> pa.DataFrameSchema:
    v = cfg["validation"]
    return pa.DataFrameSchema(
        {
            "user_id": pa.Column("Int64", nullable=False, unique=False),
            "gender": pa.Column(str, nullable=True, checks=pa.Check.isin(v["valid_genders"], name="invalid_gender")),
            "age": pa.Column("Int64", nullable=True, checks=pa.Check.isin(v["valid_ages"], name="invalid_age")),
            "occupation": pa.Column("Int64", nullable=True, checks=pa.Check.in_range(0, 20, name="invalid_occupation")),
            "zipcode": pa.Column(str, nullable=True),
        },
        strict=True, name="users",
    )


def items_schema(cfg: dict) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "item_id": pa.Column("Int64", nullable=False),
            "title": pa.Column(str, nullable=False),
            "genres": pa.Column(str, nullable=True),
        },
        strict=True, name="items",
    )
