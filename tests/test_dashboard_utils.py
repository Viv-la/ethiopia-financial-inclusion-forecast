"""Unit tests for reusable dashboard helpers."""

import pandas as pd

from src.dashboard_utils import (
    clean_text,
    find_first_column,
    format_number,
    normalise_columns,
    shorten_label,
    to_csv_bytes,
)


def test_clean_text_normalises_whitespace_and_missing_values() -> None:
    assert clean_text("  financial   inclusion ") == "financial inclusion"
    assert clean_text(pd.NA) == ""


def test_normalise_columns_returns_copy_with_consistent_names() -> None:
    original = pd.DataFrame({"Fiscal Year": [2025], "Account-Rate/%": [46]})
    result = normalise_columns(original)

    assert list(result.columns) == ["fiscal_year", "account_rate_%"]
    assert list(original.columns) == ["Fiscal Year", "Account-Rate/%"]


def test_find_first_column_respects_candidate_priority() -> None:
    frame = pd.DataFrame({"year": [2025], "fiscal_year": [2024]})
    assert find_first_column(frame, ["fiscal_year", "year"]) == "fiscal_year"
    assert find_first_column(frame, ["quarter"]) is None


def test_format_number_handles_missing_and_non_numeric_values() -> None:
    assert format_number(None) == "N/A"
    assert format_number("not reported") == "not reported"


def test_format_number_uses_compact_suffixes() -> None:
    assert format_number(1_250) == "1.2K"
    assert format_number(2_500_000) == "2.5M"
    assert format_number(46.25, decimals=2) == "46.25"


def test_shorten_label_preserves_short_and_truncates_long_text() -> None:
    assert shorten_label("Account ownership", maximum=20) == "Account ownership"
    assert shorten_label("Mobile money account ownership", maximum=15) == (
        "Mobile money..."
    )


def test_to_csv_bytes_produces_utf8_csv_without_index() -> None:
    payload = to_csv_bytes(pd.DataFrame({"indicator": ["Account ownership"]}))
    assert payload.decode("utf-8") == "indicator\nAccount ownership\n"
