"""
Task 1: Data preparation and validation
Forecasting Financial Inclusion in Ethiopia

This script:
1. Reads all sheets from the unified dataset workbook.
2. Cleans column names and text fields.
3. Removes completely empty rows and columns.
4. Checks duplicate IDs and missing values.
5. Preserves main records and impact links separately.
6. Exports clean CSV files to data/processed.
7. Generates a data-quality summary.
"""

from pathlib import Path
import re
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_FILE = RAW_DIR / "ethiopia_fi_unified_data.xlsx"
REFERENCE_FILE = RAW_DIR / "reference_codes.xlsx"
GUIDE_FILE = RAW_DIR / "Additional Data Points Guide.xlsx"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def clean_column_name(column):
    """Convert a column name to lowercase snake_case."""
    column = str(column).strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")


def clean_dataframe(df):
    """Apply basic, non-destructive cleaning to a dataframe."""
    df = df.copy()

    # Remove completely empty rows and columns
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Standardise column names
    df.columns = [clean_column_name(column) for column in df.columns]

    # Clean text without changing numeric values
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].apply(
            lambda value: re.sub(r"\s+", " ", value).strip()
            if isinstance(value, str)
            else value
        )

        # Convert common missing-value strings to proper null values
        df[column] = df[column].replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "N/A": pd.NA,
                "n/a": pd.NA,
                "NULL": pd.NA,
                "null": pd.NA,
                "-": pd.NA,
            }
        )

    return df.reset_index(drop=True)


def safe_filename(name):
    """Convert a worksheet name into a safe filename."""
    return clean_column_name(name) or "sheet"


def find_id_columns(df):
    """Return columns that appear to contain record identifiers."""
    return [
        column
        for column in df.columns
        if column == "id" or column.endswith("_id")
    ]


def dataframe_quality_report(file_name, sheet_name, df):
    """Create data-quality statistics for one worksheet."""
    id_columns = find_id_columns(df)

    duplicate_count = 0
    duplicate_column = ""

    for column in id_columns:
        non_null_values = df[column].dropna()

        if not non_null_values.empty:
            duplicates = int(non_null_values.duplicated().sum())

            if duplicates > duplicate_count:
                duplicate_count = duplicates
                duplicate_column = column

    return {
        "source_file": file_name,
        "sheet_name": sheet_name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "id_column_checked": duplicate_column,
        "duplicate_ids": duplicate_count,
    }


def process_workbook(workbook_path, output_prefix):
    """Clean and export every worksheet in an Excel workbook."""
    if not workbook_path.exists():
        raise FileNotFoundError(f"File not found: {workbook_path}")

    workbook = pd.ExcelFile(workbook_path)
    cleaned_sheets = {}
    quality_results = []

    print(f"\nProcessing: {workbook_path.name}")
    print(f"Worksheets found: {workbook.sheet_names}")

    for sheet_name in workbook.sheet_names:
        raw_df = pd.read_excel(workbook_path, sheet_name=sheet_name)
        clean_df = clean_dataframe(raw_df)

        if clean_df.empty:
            print(f"Skipped empty worksheet: {sheet_name}")
            continue

        cleaned_sheets[sheet_name] = clean_df

        output_name = (
            f"{output_prefix}_{safe_filename(sheet_name)}.csv"
        )
        output_path = PROCESSED_DIR / output_name
        clean_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        quality_results.append(
            dataframe_quality_report(
                workbook_path.name,
                sheet_name,
                clean_df,
            )
        )

        print(
            f"Saved {output_name}: "
            f"{len(clean_df)} rows × {len(clean_df.columns)} columns"
        )

    return cleaned_sheets, quality_results


# ---------------------------------------------------------
# Main processing workflow
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("WEEK 11 — TASK 1 DATA PREPARATION")
    print("=" * 60)

    all_quality_results = []

    # Main financial-inclusion dataset
    data_sheets, quality_results = process_workbook(
        DATA_FILE,
        "ethiopia_fi",
    )
    all_quality_results.extend(quality_results)

    # Reference codes
    reference_sheets, quality_results = process_workbook(
        REFERENCE_FILE,
        "reference_codes",
    )
    all_quality_results.extend(quality_results)

    # Additional data points guide
    guide_sheets, quality_results = process_workbook(
        GUIDE_FILE,
        "additional_data_points_guide",
    )
    all_quality_results.extend(quality_results)

    # Produce a convenient primary records file
    if data_sheets:
        main_sheet_name = max(
            data_sheets,
            key=lambda name: len(data_sheets[name]),
        )
        main_records = data_sheets[main_sheet_name]

        main_output = (
            PROCESSED_DIR
            / "ethiopia_fi_cleaned_records.csv"
        )
        main_records.to_csv(
            main_output,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\nPrimary dataset selected: {main_sheet_name}"
            f"\nSaved: {main_output.name}"
        )

    # Create data-quality summary
    quality_df = pd.DataFrame(all_quality_results)
    quality_output = PROCESSED_DIR / "data_quality_summary.csv"
    quality_df.to_csv(
        quality_output,
        index=False,
        encoding="utf-8-sig",
    )

    # Create a column inventory for documentation
    inventory_rows = []

    workbook_groups = {
        "ethiopia_fi_unified_data.xlsx": data_sheets,
        "reference_codes.xlsx": reference_sheets,
        "Additional Data Points Guide.xlsx": guide_sheets,
    }

    for source_file, sheets in workbook_groups.items():
        for sheet_name, df in sheets.items():
            for column in df.columns:
                inventory_rows.append(
                    {
                        "source_file": source_file,
                        "sheet_name": sheet_name,
                        "column_name": column,
                        "data_type": str(df[column].dtype),
                        "non_null_values": int(
                            df[column].notna().sum()
                        ),
                        "missing_values": int(
                            df[column].isna().sum()
                        ),
                        "unique_values": int(
                            df[column].nunique(dropna=True)
                        ),
                    }
                )

    inventory_df = pd.DataFrame(inventory_rows)
    inventory_output = PROCESSED_DIR / "column_inventory.csv"
    inventory_df.to_csv(
        inventory_output,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("TASK 1 PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Output folder: {PROCESSED_DIR}")
    print(f"Quality report: {quality_output.name}")
    print(f"Column inventory: {inventory_output.name}")
    print("\nData-quality summary:")
    print(quality_df.to_string(index=False))


if __name__ == "__main__":
    main()