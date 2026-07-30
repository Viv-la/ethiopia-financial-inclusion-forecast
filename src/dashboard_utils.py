"""Pure helper functions shared by the Streamlit dashboard and tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def clean_text(value: Any) -> str:
    """Return whitespace-normalised text, or an empty string for missing data."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def normalise_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase snake_case-style column names."""
    normalised = dataframe.copy()
    normalised.columns = [
        clean_text(column)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        for column in normalised.columns
    ]
    return normalised


def find_first_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    """Return the first candidate present in a dataframe, if any."""
    return next(
        (candidate for candidate in candidates if candidate in dataframe.columns),
        None,
    )


def format_number(value: Any, decimals: int = 1) -> str:
    """Format a dashboard value using compact thousands and millions suffixes."""
    if pd.isna(value):
        return "N/A"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return clean_text(value)

    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.1f}K"
    return f"{number:,.{decimals}f}"


def shorten_label(value: Any, maximum: int = 48) -> str:
    """Truncate a label to a safe display length."""
    text = clean_text(value)
    if len(text) <= maximum:
        return text
    return text[: maximum - 3] + "..."


def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Encode a dataframe as a UTF-8 CSV download."""
    return dataframe.to_csv(index=False).encode("utf-8")
