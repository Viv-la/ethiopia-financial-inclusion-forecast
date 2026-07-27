"""
Task 2: Exploratory Data Analysis
Forecasting Financial Inclusion in Ethiopia

Outputs:
- Clean analytical tables
- Descriptive summaries
- Five professional charts
- Evidence-based findings
- Data limitations report
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ethiopia_fi_cleaned_records.csv"
)
IMPACT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ethiopia_fi_impact_sheet.csv"
)
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Visual style
# ---------------------------------------------------------
AU_GREEN = "#00843D"
AU_GOLD = "#F2A900"
DARK_GREEN = "#174A3A"
LIGHT_GREEN = "#8CC63F"
RED = "#C44E52"
BLUE = "#3572A5"
GREY = "#6B7280"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.figsize": (11, 6),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def normalise_text(value):
    """Return a clean lowercase string for matching."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def first_existing_column(df, candidates):
    """Return the first candidate column available in the dataframe."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def save_figure(filename):
    """Apply common layout settings and save the current figure."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {filename}")


def extract_year(row):
    """Create a comparable analytical year from date or fiscal-year fields."""
    fiscal_year = str(row.get("fiscal_year", "")).strip()

    # Use the opening year of an Ethiopian/project fiscal-year label
    if fiscal_year and fiscal_year.lower() not in {"nan", "none", "<na>"}:
        digits = "".join(
            character if character.isdigit() else " "
            for character in fiscal_year
        ).split()

        four_digit_years = [
            int(value)
            for value in digits
            if len(value) == 4
        ]

        if four_digit_years:
            return four_digit_years[0]

    observation_date = row.get("observation_date")

    if pd.notna(observation_date):
        parsed_date = pd.to_datetime(observation_date, errors="coerce")

        if pd.notna(parsed_date):
            return int(parsed_date.year)

        text = str(observation_date)

        for token in text.replace("/", " ").replace("-", " ").split():
            if token.isdigit() and len(token) == 4:
                return int(token)

    return np.nan


def classify_indicator(row):
    """Assign broad analytical groups using pillar, code and indicator text."""
    combined_text = " ".join(
        [
            normalise_text(row.get("pillar")),
            normalise_text(row.get("indicator")),
            normalise_text(row.get("indicator_code")),
        ]
    )

    if any(
        phrase in combined_text
        for phrase in [
            "digital payment",
            "digital-payment",
            "made or received digital",
            "mobile money",
            "electronic payment",
            "usage",
        ]
    ):
        return "Digital payment usage"

    if any(
        phrase in combined_text
        for phrase in [
            "account ownership",
            "account owner",
            "financial account",
            "formal account",
            "access",
        ]
    ):
        return "Account ownership"

    if "gender" in combined_text or "female" in combined_text:
        return "Gender inclusion"

    if "afford" in combined_text or "cost" in combined_text:
        return "Affordability"

    return "Other"


def choose_percent_value(row):
    """
    Convert proportion values into percentage points where appropriate.
    Values already above 1 are retained.
    """
    value = row.get("value_numeric")
    unit = normalise_text(row.get("unit"))
    value_type = normalise_text(row.get("value_type"))

    if pd.isna(value):
        return np.nan

    value = float(value)

    looks_like_percentage = (
        "%" in unit
        or "percent" in unit
        or "percentage" in value_type
        or "rate" in normalise_text(row.get("indicator"))
        or "share" in normalise_text(row.get("indicator"))
    )

    if looks_like_percentage and 0 <= value <= 1:
        return value * 100

    return value


# ---------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------
def load_and_prepare_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "record_type",
        "pillar",
        "indicator",
        "value_numeric",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing_columns)
        )

    df["record_type_clean"] = (
        df["record_type"].astype(str).str.strip().str.lower()
    )

    df["value_numeric"] = pd.to_numeric(
        df["value_numeric"],
        errors="coerce",
    )

    df["analytical_year"] = df.apply(extract_year, axis=1)
    df["indicator_group"] = df.apply(classify_indicator, axis=1)
    df["analysis_value"] = df.apply(choose_percent_value, axis=1)

    observations = df[
        df["record_type_clean"].eq("observation")
    ].copy()

    events = df[
        df["record_type_clean"].eq("event")
    ].copy()

    targets = df[
        df["record_type_clean"].eq("target")
    ].copy()

    observations = observations[
        observations["value_numeric"].notna()
    ].copy()

    observations["analytical_year"] = pd.to_numeric(
        observations["analytical_year"],
        errors="coerce",
    )

    observations.to_csv(
        PROCESSED_DIR / "eda_observations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return df, observations, events, targets


# ---------------------------------------------------------
# Summary tables
# ---------------------------------------------------------
def create_summary_tables(df, observations):
    record_summary = (
        df.groupby("record_type_clean", dropna=False)
        .size()
        .reset_index(name="record_count")
        .rename(columns={"record_type_clean": "record_type"})
    )

    pillar_summary = (
        df.assign(pillar=df["pillar"].fillna("Not assigned"))
        .groupby("pillar")
        .size()
        .reset_index(name="record_count")
        .sort_values("record_count", ascending=False)
    )

    indicator_summary = (
        observations.groupby(
            ["indicator_group", "indicator"],
            dropna=False,
        )
        .agg(
            observations=("analysis_value", "count"),
            minimum=("analysis_value", "min"),
            maximum=("analysis_value", "max"),
            mean=("analysis_value", "mean"),
            first_year=("analytical_year", "min"),
            last_year=("analytical_year", "max"),
        )
        .reset_index()
        .sort_values(
            ["indicator_group", "observations"],
            ascending=[True, False],
        )
    )

    missingness = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_values"})
    )
    missingness["missing_percent"] = (
        missingness["missing_values"] / len(df) * 100
    ).round(1)
    missingness = missingness.sort_values(
        "missing_percent",
        ascending=False,
    )

    record_summary.to_csv(
        PROCESSED_DIR / "eda_record_summary.csv",
        index=False,
    )
    pillar_summary.to_csv(
        PROCESSED_DIR / "eda_pillar_summary.csv",
        index=False,
    )
    indicator_summary.to_csv(
        PROCESSED_DIR / "eda_indicator_summary.csv",
        index=False,
    )
    missingness.to_csv(
        PROCESSED_DIR / "eda_missingness_summary.csv",
        index=False,
    )

    return (
        record_summary,
        pillar_summary,
        indicator_summary,
        missingness,
    )


# ---------------------------------------------------------
# Visualisations
# ---------------------------------------------------------
def plot_record_types(df):
    counts = (
        df["record_type_clean"]
        .replace({"nan": "Unknown"})
        .value_counts()
        .sort_values(ascending=True)
    )

    colors = [AU_GREEN, AU_GOLD, BLUE, GREY][: len(counts)]

    plt.figure(figsize=(9, 5))
    bars = plt.barh(
        counts.index.str.title(),
        counts.values,
        color=colors,
    )

    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontweight="bold",
        )

    plt.title("Composition of the Unified Dataset")
    plt.xlabel("Number of records")
    plt.ylabel("")
    sns.despine(left=True, bottom=True)
    save_figure("01_dataset_composition.png")


def plot_pillar_coverage(df):
    pillar_counts = (
        df.assign(pillar=df["pillar"].fillna("Event / multi-pillar"))
        ["pillar"]
        .value_counts()
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        pillar_counts.index,
        pillar_counts.values,
        color=AU_GREEN,
        alpha=0.88,
    )

    for bar, value in zip(bars, pillar_counts.values):
        plt.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
        )

    plt.title("Financial-Inclusion Records by Pillar")
    plt.xlabel("Number of records")
    plt.ylabel("")
    sns.despine(left=True, bottom=True)
    save_figure("02_pillar_coverage.png")


def plot_observation_timeline(observations):
    timeline = observations.dropna(
        subset=["analytical_year", "analysis_value"]
    ).copy()

    if timeline.empty:
        return

    timeline["analytical_year"] = timeline[
        "analytical_year"
    ].astype(int)

    priority = timeline[
        timeline["indicator_group"].isin(
            ["Account ownership", "Digital payment usage"]
        )
    ].copy()

    if priority.empty:
        priority = timeline.copy()

    grouped = (
        priority.groupby(
            ["analytical_year", "indicator_group"],
            as_index=False,
        )["analysis_value"]
        .mean()
    )

    plt.figure(figsize=(11, 6))

    palette = {
        "Account ownership": AU_GREEN,
        "Digital payment usage": AU_GOLD,
        "Gender inclusion": RED,
        "Affordability": BLUE,
        "Other": GREY,
    }

    for group, group_data in grouped.groupby("indicator_group"):
        group_data = group_data.sort_values("analytical_year")

        plt.plot(
            group_data["analytical_year"],
            group_data["analysis_value"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=group,
            color=palette.get(group, GREY),
        )

    plt.title("Historical Financial-Inclusion Indicator Trends")
    plt.xlabel("Analytical year")
    plt.ylabel("Reported value")
    plt.legend(frameon=False)
    sns.despine()
    save_figure("03_access_usage_trends.png")


def plot_gender_comparison(observations):
    gender_column = first_existing_column(
        observations,
        ["gender", "sex"],
    )

    if gender_column is None:
        return

    gender_data = observations[
        observations[gender_column].notna()
        & observations["analysis_value"].notna()
    ].copy()

    gender_data = gender_data[
        ~gender_data[gender_column]
        .astype(str)
        .str.lower()
        .isin(["all", "total", "national", "not applicable"])
    ]

    if gender_data.empty:
        return

    gender_summary = (
        gender_data.groupby(
            [gender_column, "indicator_group"],
            as_index=False,
        )["analysis_value"]
        .mean()
    )

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=gender_summary,
        x=gender_column,
        y="analysis_value",
        hue="indicator_group",
        palette=[AU_GREEN, AU_GOLD, RED, BLUE, GREY],
    )

    plt.title("Financial-Inclusion Indicators by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Average reported value")
    plt.legend(
        title="Indicator group",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    sns.despine()
    save_figure("04_gender_comparison.png")


def plot_location_comparison(observations):
    location_column = first_existing_column(
        observations,
        ["location", "residence", "area"],
    )

    if location_column is None:
        return

    location_data = observations[
        observations[location_column].notna()
        & observations["analysis_value"].notna()
    ].copy()

    excluded = {
        "national",
        "ethiopia",
        "all",
        "total",
        "not applicable",
    }

    location_data = location_data[
        ~location_data[location_column]
        .astype(str)
        .str.lower()
        .isin(excluded)
    ]

    if location_data.empty:
        return

    summary = (
        location_data.groupby(
            [location_column, "indicator_group"],
            as_index=False,
        )["analysis_value"]
        .mean()
    )

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=summary,
        x=location_column,
        y="analysis_value",
        hue="indicator_group",
        palette=[AU_GREEN, AU_GOLD, RED, BLUE, GREY],
    )

    plt.title("Financial-Inclusion Indicators by Location")
    plt.xlabel("Location category")
    plt.ylabel("Average reported value")
    plt.xticks(rotation=25, ha="right")
    plt.legend(
        title="Indicator group",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    sns.despine()
    save_figure("05_location_comparison.png")


def plot_observations_by_year(observations):
    coverage = (
        observations.dropna(subset=["analytical_year"])
        .assign(
            analytical_year=lambda data: data[
                "analytical_year"
            ].astype(int)
        )
        .groupby("analytical_year")
        .size()
        .reset_index(name="observations")
    )

    if coverage.empty:
        return

    plt.figure(figsize=(11, 5))
    bars = plt.bar(
        coverage["analytical_year"].astype(str),
        coverage["observations"],
        color=AU_GREEN,
    )

    for bar, value in zip(bars, coverage["observations"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(value),
            ha="center",
            fontweight="bold",
        )

    plt.title("Historical Observation Coverage by Year")
    plt.xlabel("Analytical year")
    plt.ylabel("Number of observations")
    sns.despine()
    save_figure("06_observation_coverage_by_year.png")


def plot_missingness(missingness):
    chart_data = missingness[
        missingness["missing_percent"] > 0
    ].head(15).sort_values("missing_percent")

    if chart_data.empty:
        return

    plt.figure(figsize=(10, 7))
    plt.barh(
        chart_data["column"],
        chart_data["missing_percent"],
        color=AU_GOLD,
    )
    plt.title("Variables with the Highest Missingness")
    plt.xlabel("Missing values (%)")
    plt.ylabel("")
    sns.despine(left=True, bottom=True)
    save_figure("07_missingness.png")


# ---------------------------------------------------------
# Findings and limitations
# ---------------------------------------------------------
def create_findings(
    df,
    observations,
    events,
    targets,
    indicator_summary,
):
    year_values = observations["analytical_year"].dropna()

    if not year_values.empty:
        first_year = int(year_values.min())
        last_year = int(year_values.max())
        time_statement = (
            f"Historical observations cover approximately {first_year} "
            f"to {last_year}, but the distribution is irregular."
        )
    else:
        time_statement = (
            "Historical year coverage could not be determined consistently."
        )

    group_counts = observations["indicator_group"].value_counts()
    access_count = int(group_counts.get("Account ownership", 0))
    usage_count = int(group_counts.get("Digital payment usage", 0))

    gender_count = 0
    if "gender" in observations.columns:
        gender_count = int(observations["gender"].notna().sum())

    location_count = 0
    if "location" in observations.columns:
        location_count = int(observations["location"].notna().sum())

    findings = f"""# Task 2 Exploratory Data Analysis Findings

## Dataset overview

The unified dataset contains **{len(df)} primary records**, comprising
**{len(observations)} historical observations**, **{len(events)} events** and
**{len(targets)} targets**.

## Evidence-based findings

1. **Access has stronger coverage than several other dimensions.**  
   The Access pillar contains the largest number of pillar-specific records.
   This makes account ownership more suitable for quantitative trend analysis
   than indicators with only one or two historical measurements.

2. **The principal forecasting indicators have limited time-series depth.**  
   The automated classification identified approximately **{access_count}
   account-ownership observations** and **{usage_count} digital-payment usage
   observations**. These are useful for directional analysis but insufficient
   for highly parameterised forecasting models.

3. **Historical observations are concentrated in recent years.**  
   {time_statement} This concentration means recent developments may dominate
   the analysis, while long-term structural change is less precisely measured.

4. **The dataset combines access, usage, gender and affordability evidence.**  
   This supports a multidimensional interpretation of financial inclusion.
   However, coverage is uneven, with affordability substantially less
   represented than Access and Usage.

5. **Gender evidence exists but is not consistently available.**  
   Approximately **{gender_count} observation records** contain a populated
   gender field. Gender comparisons should therefore be presented as partial
   evidence rather than a complete national time series.

6. **Location disaggregation is limited.**  
   Approximately **{location_count} observation records** contain location
   information, and much of the dataset is national. Regional and urban-rural
   differences cannot be estimated consistently from the current data alone.

7. **Events offer plausible context but do not establish causality.**  
   The dataset contains **{len(events)} policy, infrastructure or market events.
   These can be compared with indicator movements, but observed timing alone
   cannot prove that an event caused a change.

8. **Targets must remain separate from historical outcomes.**  
   The **{len(targets)} target records** represent intended future performance,
   not realised measurements. They will be used for scenario benchmarking and
   progress assessment, not for training the baseline forecast.

## Modelling implications

Because the time series are short and irregular, Task 4 should prioritise:

- Transparent and parsimonious forecasting methods
- Linear or trend-based baselines
- Scenario adjustments informed by documented events
- Wide uncertainty intervals
- Clear separation of observations, estimates and targets
- Sensitivity testing instead of overly complex machine-learning models

## Generated outputs

The EDA generated:

- Dataset composition chart
- Pillar coverage chart
- Access and usage trend chart
- Gender comparison chart, where data permit
- Location comparison chart, where data permit
- Observation coverage chart
- Missingness chart
- Record, pillar, indicator and missingness summary tables
"""

    findings_path = REPORTS_DIR / "task_2_eda_findings.md"
    findings_path.write_text(findings, encoding="utf-8")
    print(f"Saved report: {findings_path.name}")

    limitations = """# Task 2 Data Limitations

1. The historical time series are short and irregularly spaced.
2. Some indicators contain only one or two observations.
3. Calendar years and fiscal years are mixed in the source data.
4. Indicator definitions may differ between institutions and survey rounds.
5. Gender-disaggregated data are incomplete.
6. Regional and urban-rural coverage is limited.
7. Affordability is underrepresented relative to Access and Usage.
8. Missing fields often reflect non-applicability across different record types.
9. Events and indicator movements show association, not proven causation.
10. Targets and projections must not be interpreted as actual historical outcomes.
11. Small samples limit the validity of complex forecasting techniques.
12. Confidence intervals should reflect both model and data uncertainty.
"""

    limitations_path = REPORTS_DIR / "task_2_data_limitations.md"
    limitations_path.write_text(limitations, encoding="utf-8")
    print(f"Saved report: {limitations_path.name}")


# ---------------------------------------------------------
# Main workflow
# ---------------------------------------------------------
def main():
    print("=" * 65)
    print("WEEK 11 — TASK 2 EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    df, observations, events, targets = load_and_prepare_data()

    (
        record_summary,
        pillar_summary,
        indicator_summary,
        missingness,
    ) = create_summary_tables(df, observations)

    plot_record_types(df)
    plot_pillar_coverage(df)
    plot_observation_timeline(observations)
    plot_gender_comparison(observations)
    plot_location_comparison(observations)
    plot_observations_by_year(observations)
    plot_missingness(missingness)

    create_findings(
        df,
        observations,
        events,
        targets,
        indicator_summary,
    )

    print("\nDataset summary")
    print("-" * 65)
    print(f"Primary records: {len(df)}")
    print(f"Historical observations: {len(observations)}")
    print(f"Events: {len(events)}")
    print(f"Targets: {len(targets)}")
    print(f"Figures folder: {FIGURES_DIR}")
    print(f"Reports folder: {REPORTS_DIR}")

    print("\n" + "=" * 65)
    print("TASK 2 COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()