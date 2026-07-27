"""
Task 3: Event-to-Indicator Association and Impact Analysis
Forecasting Financial Inclusion in Ethiopia
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
IMPACT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ethiopia_fi_impact_sheet.csv"
)
RECORDS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ethiopia_fi_cleaned_records.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
def clean_text(value):
    """Standardise text without changing its substantive meaning."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def normalise_key(value):
    """Create a lowercase matching key."""
    return clean_text(value).lower()


def find_column(dataframe, candidates):
    """Return the first available column from a list of candidates."""
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def save_figure(filename):
    """Save a chart using consistent formatting."""
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / filename,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print(f"Saved figure: {filename}")


def map_ordered_score(value, mapping, default=np.nan):
    """Convert a text classification into a numeric analytical score."""
    key = normalise_key(value)
    return mapping.get(key, default)


def classify_indicator(value):
    """Assign related indicators to broad financial-inclusion groups."""
    text = normalise_key(value)

    if any(
        phrase in text
        for phrase in [
            "digital payment",
            "digital-payment",
            "mobile money",
            "electronic payment",
            "digital transaction",
            "made or received digital",
        ]
    ):
        return "Digital payment usage"

    if any(
        phrase in text
        for phrase in [
            "account ownership",
            "account owner",
            "financial account",
            "formal account",
            "bank account",
        ]
    ):
        return "Account ownership"

    if any(
        phrase in text
        for phrase in [
            "gender",
            "female",
            "women",
        ]
    ):
        return "Gender inclusion"

    if any(
        phrase in text
        for phrase in [
            "affordability",
            "affordable",
            "cost",
            "fee",
        ]
    ):
        return "Affordability"

    if any(
        phrase in text
        for phrase in [
            "credit",
            "loan",
            "finance",
            "saving",
        ]
    ):
        return "Financial services"

    return clean_text(value) or "Other"


def shorten_label(value, maximum=48):
    """Shorten long chart labels while retaining readability."""
    text = clean_text(value)

    if len(text) <= maximum:
        return text

    return text[: maximum - 3] + "..."


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
def load_data():
    if not IMPACT_FILE.exists():
        raise FileNotFoundError(
            f"Impact-link dataset not found: {IMPACT_FILE}"
        )

    impacts = pd.read_csv(IMPACT_FILE)

    if RECORDS_FILE.exists():
        records = pd.read_csv(RECORDS_FILE)
    else:
        records = pd.DataFrame()

    impacts.columns = [
        normalise_key(column).replace(" ", "_")
        for column in impacts.columns
    ]

    if not records.empty:
        records.columns = [
            normalise_key(column).replace(" ", "_")
            for column in records.columns
        ]

    impacts = impacts.dropna(how="all").copy()

    text_columns = impacts.select_dtypes(include="object").columns

    for column in text_columns:
        impacts[column] = impacts[column].apply(
            lambda value: clean_text(value) if pd.notna(value) else np.nan
        )

    print(f"Loaded {len(impacts)} event-impact relationships")

    return impacts, records


# ---------------------------------------------------------
# Preparation and scoring
# ---------------------------------------------------------
def prepare_impact_links(impacts, records):
    parent_column = find_column(
        impacts,
        ["parent_id", "event_id", "record_id"],
    )
    indicator_column = find_column(
        impacts,
        [
            "related_indicator",
            "indicator",
            "indicator_code",
            "affected_indicator",
        ],
    )
    direction_column = find_column(
        impacts,
        ["impact_direction", "direction"],
    )
    magnitude_column = find_column(
        impacts,
        ["impact_magnitude", "magnitude"],
    )
    confidence_column = find_column(
        impacts,
        ["confidence", "confidence_level"],
    )
    lag_column = find_column(
        impacts,
        ["lag_months", "impact_lag_months", "lag"],
    )
    estimate_column = find_column(
        impacts,
        ["impact_estimate", "estimated_impact"],
    )
    relationship_column = find_column(
        impacts,
        ["relationship_type", "relationship"],
    )
    evidence_column = find_column(
        impacts,
        ["evidence_basis", "evidence", "evidence_type"],
    )

    if parent_column is None:
        raise ValueError(
            "The impact-link file does not contain a parent/event ID column."
        )

    if indicator_column is None:
        raise ValueError(
            "The impact-link file does not contain a related-indicator column."
        )

    prepared = impacts.copy()

    prepared["event_id"] = prepared[parent_column].apply(clean_text)
    prepared["related_indicator_clean"] = prepared[
        indicator_column
    ].apply(clean_text)

    prepared["indicator_group"] = prepared[
        "related_indicator_clean"
    ].apply(classify_indicator)

    direction_mapping = {
        "positive": 1,
        "increase": 1,
        "increasing": 1,
        "upward": 1,
        "negative": -1,
        "decrease": -1,
        "decreasing": -1,
        "downward": -1,
        "mixed": 0,
        "neutral": 0,
        "uncertain": 0,
        "unknown": 0,
    }

    magnitude_mapping = {
        "very low": 0.25,
        "low": 0.50,
        "small": 0.50,
        "medium": 0.75,
        "moderate": 0.75,
        "high": 1.00,
        "large": 1.00,
        "very high": 1.25,
    }

    confidence_mapping = {
        "very low": 0.25,
        "low": 0.50,
        "medium": 0.75,
        "moderate": 0.75,
        "high": 1.00,
        "very high": 1.00,
    }

    if direction_column:
        prepared["direction_score"] = prepared[
            direction_column
        ].apply(
            lambda value: map_ordered_score(
                value,
                direction_mapping,
                default=0,
            )
        )
        prepared["impact_direction_clean"] = prepared[
            direction_column
        ].apply(clean_text)
    else:
        prepared["direction_score"] = 0
        prepared["impact_direction_clean"] = "Unknown"

    if magnitude_column:
        prepared["magnitude_score"] = prepared[
            magnitude_column
        ].apply(
            lambda value: map_ordered_score(
                value,
                magnitude_mapping,
                default=0.50,
            )
        )
        prepared["impact_magnitude_clean"] = prepared[
            magnitude_column
        ].apply(clean_text)
    else:
        prepared["magnitude_score"] = 0.50
        prepared["impact_magnitude_clean"] = "Not specified"

    if confidence_column:
        prepared["confidence_score"] = prepared[
            confidence_column
        ].apply(
            lambda value: map_ordered_score(
                value,
                confidence_mapping,
                default=0.50,
            )
        )
        prepared["confidence_clean"] = prepared[
            confidence_column
        ].apply(clean_text)
    else:
        prepared["confidence_score"] = 0.50
        prepared["confidence_clean"] = "Not specified"

    if lag_column:
        prepared["lag_months_numeric"] = pd.to_numeric(
            prepared[lag_column],
            errors="coerce",
        )
    else:
        prepared["lag_months_numeric"] = np.nan

    if estimate_column:
        prepared["impact_estimate_numeric"] = pd.to_numeric(
            prepared[estimate_column],
            errors="coerce",
        )
    else:
        prepared["impact_estimate_numeric"] = np.nan

    if relationship_column:
        prepared["relationship_type_clean"] = prepared[
            relationship_column
        ].apply(clean_text)
    else:
        prepared["relationship_type_clean"] = "Not specified"

    if evidence_column:
        prepared["evidence_basis_clean"] = prepared[
            evidence_column
        ].apply(clean_text)
    else:
        prepared["evidence_basis_clean"] = "Not specified"

    # Direction × magnitude × confidence
    prepared["weighted_impact_score"] = (
        prepared["direction_score"]
        * prepared["magnitude_score"]
        * prepared["confidence_score"]
    )

    # Add event names/descriptions from the primary dataset where available
    prepared["event_name"] = prepared["event_id"]

    if not records.empty and "record_id" in records.columns:
        event_records = records.copy()

        if "record_type" in event_records.columns:
            event_records = event_records[
                event_records["record_type"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("event")
            ].copy()

        event_label_column = find_column(
            event_records,
            [
                "indicator",
                "event_name",
                "event",
                "title",
                "description",
                "notes",
            ],
        )

        if event_label_column:
            event_lookup = (
                event_records[["record_id", event_label_column]]
                .dropna(subset=["record_id"])
                .drop_duplicates(subset=["record_id"])
                .rename(
                    columns={
                        "record_id": "event_id",
                        event_label_column: "event_label",
                    }
                )
            )

            event_lookup["event_id"] = event_lookup[
                "event_id"
            ].apply(clean_text)

            prepared = prepared.merge(
                event_lookup,
                on="event_id",
                how="left",
            )

            prepared["event_name"] = prepared[
                "event_label"
            ].fillna(prepared["event_id"])

    prepared["event_label_short"] = prepared[
        "event_name"
    ].apply(shorten_label)

    prepared.to_csv(
        PROCESSED_DIR / "event_impact_scored_links.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return prepared


# ---------------------------------------------------------
# Association matrix and summaries
# ---------------------------------------------------------
def create_analysis_tables(prepared):
    association_matrix = pd.pivot_table(
        prepared,
        index="event_label_short",
        columns="indicator_group",
        values="weighted_impact_score",
        aggfunc="sum",
        fill_value=0,
    )

    association_matrix.to_csv(
        PROCESSED_DIR / "event_indicator_association_matrix.csv",
        encoding="utf-8-sig",
    )

    event_summary = (
        prepared.groupby(
            ["event_id", "event_name", "event_label_short"],
            dropna=False,
        )
        .agg(
            linked_indicators=(
                "related_indicator_clean",
                "nunique",
            ),
            total_links=(
                "related_indicator_clean",
                "size",
            ),
            net_weighted_impact=(
                "weighted_impact_score",
                "sum",
            ),
            mean_confidence=(
                "confidence_score",
                "mean",
            ),
            mean_lag_months=(
                "lag_months_numeric",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "net_weighted_impact",
            ascending=False,
        )
    )

    indicator_summary = (
        prepared.groupby(
            ["indicator_group", "related_indicator_clean"],
            dropna=False,
        )
        .agg(
            linked_events=("event_id", "nunique"),
            total_links=("event_id", "size"),
            net_weighted_impact=(
                "weighted_impact_score",
                "sum",
            ),
            average_weighted_impact=(
                "weighted_impact_score",
                "mean",
            ),
            average_confidence=(
                "confidence_score",
                "mean",
            ),
            average_lag_months=(
                "lag_months_numeric",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "net_weighted_impact",
            ascending=False,
        )
    )

    direction_summary = (
        prepared.groupby(
            "impact_direction_clean",
            dropna=False,
        )
        .size()
        .reset_index(name="relationship_count")
        .sort_values(
            "relationship_count",
            ascending=False,
        )
    )

    evidence_summary = (
        prepared.groupby(
            "evidence_basis_clean",
            dropna=False,
        )
        .size()
        .reset_index(name="relationship_count")
        .sort_values(
            "relationship_count",
            ascending=False,
        )
    )

    event_summary.to_csv(
        PROCESSED_DIR / "event_impact_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    indicator_summary.to_csv(
        PROCESSED_DIR / "indicator_impact_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    direction_summary.to_csv(
        PROCESSED_DIR / "impact_direction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evidence_summary.to_csv(
        PROCESSED_DIR / "impact_evidence_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        association_matrix,
        event_summary,
        indicator_summary,
        direction_summary,
        evidence_summary,
    )


# ---------------------------------------------------------
# Visualisations
# ---------------------------------------------------------
def plot_association_matrix(association_matrix):
    if association_matrix.empty:
        return

    figure_height = max(6, len(association_matrix) * 0.65)

    plt.figure(
        figsize=(12, figure_height)
    )

    sns.heatmap(
        association_matrix,
        annot=True,
        fmt=".2f",
        cmap=sns.diverging_palette(
            10,
            140,
            as_cmap=True,
        ),
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": "Weighted impact score"
        },
    )

    plt.title("Event-to-Indicator Association Matrix")
    plt.xlabel("Financial-inclusion indicator group")
    plt.ylabel("Event")
    save_figure("08_event_indicator_association_matrix.png")


def plot_event_impact_ranking(event_summary):
    if event_summary.empty:
        return

    chart_data = (
        event_summary
        .sort_values(
            "net_weighted_impact",
            ascending=True,
        )
        .tail(15)
    )

    colors = [
        AU_GREEN if value >= 0 else RED
        for value in chart_data["net_weighted_impact"]
    ]

    plt.figure(
        figsize=(11, max(6, len(chart_data) * 0.55))
    )

    bars = plt.barh(
        chart_data["event_label_short"],
        chart_data["net_weighted_impact"],
        color=colors,
    )

    plt.axvline(
        0,
        color=GREY,
        linewidth=1,
    )

    for bar, value in zip(
        bars,
        chart_data["net_weighted_impact"],
    ):
        alignment = "left" if value >= 0 else "right"
        offset = 0.02 if value >= 0 else -0.02

        plt.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha=alignment,
            fontsize=9,
        )

    plt.title("Events Ranked by Net Weighted Impact")
    plt.xlabel("Net weighted impact score")
    plt.ylabel("Event")
    sns.despine(left=True, bottom=True)
    save_figure("09_event_impact_ranking.png")


def plot_indicator_exposure(indicator_summary):
    if indicator_summary.empty:
        return

    grouped = (
        indicator_summary.groupby(
            "indicator_group",
            as_index=False,
        )
        .agg(
            linked_events=("linked_events", "sum"),
            net_weighted_impact=(
                "net_weighted_impact",
                "sum",
            ),
        )
        .sort_values(
            "net_weighted_impact",
            ascending=True,
        )
    )

    colors = [
        AU_GREEN if value >= 0 else RED
        for value in grouped["net_weighted_impact"]
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.barh(
        grouped["indicator_group"],
        grouped["net_weighted_impact"],
        color=colors,
    )

    plt.axvline(
        0,
        color=GREY,
        linewidth=1,
    )

    for bar, value in zip(
        bars,
        grouped["net_weighted_impact"],
    ):
        plt.text(
            value + (0.02 if value >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
        )

    plt.title("Cumulative Event Impact by Indicator Group")
    plt.xlabel("Net weighted impact score")
    plt.ylabel("")
    sns.despine(left=True, bottom=True)
    save_figure("10_indicator_impact_exposure.png")


def plot_impact_directions(direction_summary):
    if direction_summary.empty:
        return

    chart_data = direction_summary.copy()

    colour_map = {
        "positive": AU_GREEN,
        "negative": RED,
        "mixed": AU_GOLD,
        "neutral": GREY,
        "uncertain": BLUE,
        "unknown": GREY,
    }

    colors = [
        colour_map.get(
            normalise_key(value),
            BLUE,
        )
        for value in chart_data["impact_direction_clean"]
    ]

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        chart_data["impact_direction_clean"],
        chart_data["relationship_count"],
        color=colors,
    )

    for bar, value in zip(
        bars,
        chart_data["relationship_count"],
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            str(value),
            ha="center",
            fontweight="bold",
        )

    plt.title("Distribution of Expected Impact Directions")
    plt.xlabel("Impact direction")
    plt.ylabel("Number of event-indicator links")
    plt.xticks(rotation=20, ha="right")
    sns.despine()
    save_figure("11_impact_direction_distribution.png")


def plot_lag_distribution(prepared):
    lag_data = prepared.dropna(
        subset=["lag_months_numeric"]
    ).copy()

    if lag_data.empty:
        return

    lag_summary = (
        lag_data.groupby(
            "indicator_group",
            as_index=False,
        )["lag_months_numeric"]
        .mean()
        .sort_values(
            "lag_months_numeric",
            ascending=True,
        )
    )

    plt.figure(figsize=(10, 6))

    bars = plt.barh(
        lag_summary["indicator_group"],
        lag_summary["lag_months_numeric"],
        color=AU_GOLD,
    )

    for bar, value in zip(
        bars,
        lag_summary["lag_months_numeric"],
    ):
        plt.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
        )

    plt.title("Average Expected Impact Lag by Indicator Group")
    plt.xlabel("Average lag in months")
    plt.ylabel("")
    sns.despine(left=True, bottom=True)
    save_figure("12_average_impact_lag.png")


def plot_confidence_distribution(prepared):
    confidence_counts = (
        prepared["confidence_clean"]
        .fillna("Not specified")
        .replace("", "Not specified")
        .value_counts()
    )

    if confidence_counts.empty:
        return

    order = [
        level
        for level in [
            "Very High",
            "High",
            "Medium",
            "Moderate",
            "Low",
            "Very Low",
            "Not specified",
        ]
        if level in confidence_counts.index
    ]

    remaining = [
        level
        for level in confidence_counts.index
        if level not in order
    ]

    display_order = order + remaining
    confidence_counts = confidence_counts.reindex(
        display_order
    )

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        confidence_counts.index,
        confidence_counts.values,
        color=BLUE,
    )

    for bar, value in zip(
        bars,
        confidence_counts.values,
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            str(value),
            ha="center",
            fontweight="bold",
        )

    plt.title("Confidence Levels of Event-Impact Relationships")
    plt.xlabel("Confidence classification")
    plt.ylabel("Number of relationships")
    plt.xticks(rotation=20, ha="right")
    sns.despine()
    save_figure("13_impact_confidence_distribution.png")


# ---------------------------------------------------------
# Findings and limitations
# ---------------------------------------------------------
def create_reports(
    prepared,
    event_summary,
    indicator_summary,
    direction_summary,
):
    total_links = len(prepared)
    total_events = prepared["event_id"].nunique()
    total_indicators = prepared[
        "related_indicator_clean"
    ].nunique()
    total_groups = prepared["indicator_group"].nunique()

    positive_links = int(
        (
            prepared["direction_score"] > 0
        ).sum()
    )
    negative_links = int(
        (
            prepared["direction_score"] < 0
        ).sum()
    )
    neutral_links = int(
        (
            prepared["direction_score"] == 0
        ).sum()
    )

    lag_values = prepared[
        "lag_months_numeric"
    ].dropna()

    if lag_values.empty:
        lag_statement = (
            "Impact lags were not sufficiently quantified "
            "for a reliable average."
        )
    else:
        lag_statement = (
            f"The documented average impact lag is "
            f"approximately **{lag_values.mean():.1f} months**, "
            f"with recorded values ranging from "
            f"**{lag_values.min():.0f} to {lag_values.max():.0f} months**."
        )

    if event_summary.empty:
        leading_event = "Not available"
        leading_event_score = 0
    else:
        leading_event = clean_text(
            event_summary.iloc[0]["event_name"]
        )
        leading_event_score = event_summary.iloc[0][
            "net_weighted_impact"
        ]

    if indicator_summary.empty:
        leading_indicator = "Not available"
        leading_indicator_score = 0
    else:
        grouped_indicators = (
            indicator_summary.groupby(
                "indicator_group",
                as_index=False,
            )["net_weighted_impact"]
            .sum()
            .sort_values(
                "net_weighted_impact",
                ascending=False,
            )
        )

        leading_indicator = grouped_indicators.iloc[0][
            "indicator_group"
        ]
        leading_indicator_score = grouped_indicators.iloc[0][
            "net_weighted_impact"
        ]

    findings = f"""# Task 3 Event-to-Indicator Impact Analysis

## Analysis overview

Task 3 examined **{total_links} documented event-impact relationships**
linking **{total_events} events** to **{total_indicators} distinct indicators**
across **{total_groups} analytical indicator groups**.

The analysis converted qualitative direction, magnitude and confidence
classifications into transparent weighted scores. The score is calculated as:

`direction × magnitude × confidence`

Positive values indicate an expected favourable contribution to financial inclusion,
negative values indicate a possible adverse effect, and zero values represent mixed,
neutral or uncertain relationships.

## Key findings

1. **Most documented relationships anticipate positive influence.**  
   The impact-link dataset contains **{positive_links} positive**,
   **{negative_links} negative** and **{neutral_links} neutral, mixed or uncertain**
   relationships. This reflects expectations recorded in the source data and does
   not independently prove that the anticipated effects occurred.

2. **The events affect multiple dimensions of financial inclusion.**  
   The association matrix shows that policy, infrastructure and market events may
   influence account ownership, digital-payment usage and other financial-service
   indicators simultaneously. Events should therefore not be assigned mechanically
   to only one pillar.

3. **The highest-ranked event is `{leading_event}`.**  
   Its net weighted impact score is **{leading_event_score:.2f}**. This ranking is
   useful for scenario design, but it depends on the source magnitude and confidence
   classifications.

4. **The most strongly exposed indicator group is `{leading_indicator}`.**  
   Its cumulative weighted score is **{leading_indicator_score:.2f}**, indicating
   that it has the strongest documented connection to the recorded events.

5. **Impacts may occur with a delay.**  
   {lag_statement} Forecast scenarios should therefore apply event effects after
   the stated lag rather than assuming immediate changes in indicator values.

6. **Confidence weighting materially affects interpretation.**  
   High-confidence relationships receive more influence than low-confidence links.
   This prevents weak or speculative evidence from having the same scenario weight
   as better-supported relationships.

7. **The association matrix is a scenario tool, not a causal model.**  
   The links document plausible pathways between events and indicators. They do not
   isolate events from economic conditions, demographic change, measurement changes
   or other concurrent reforms.

## Forecasting implications

The event-impact results should inform the forecasting stage through:

- A historical baseline based only on observed indicator values
- Positive and conservative event-adjusted scenarios
- Explicit use of documented impact lags
- Reduced weights for low-confidence relationships
- Sensitivity tests for magnitude assumptions
- Separate treatment of official targets
- Clear uncertainty intervals around all projections

## Generated outputs

Task 3 generated:

- A scored event-impact link dataset
- An event-to-indicator association matrix
- Event and indicator impact summaries
- Direction and evidence summaries
- A heatmap of event-indicator relationships
- Event impact rankings
- Indicator exposure analysis
- Impact direction, lag and confidence charts
"""

    findings_path = (
        REPORTS_DIR
        / "task_3_event_impact_findings.md"
    )
    findings_path.write_text(
        findings,
        encoding="utf-8",
    )
    print(f"Saved report: {findings_path.name}")

    limitations = """# Task 3 Event-Impact Analysis Limitations

1. The association matrix is based on documented relationships and expert-informed
   assumptions rather than experimental causal evidence.
2. An event occurring before an indicator change does not prove that it caused the
   change.
3. Several events may occur simultaneously, making individual effects difficult to
   isolate.
4. Direction, magnitude and confidence classifications are partly qualitative.
5. Numeric impact scores are analytical weights, not measured percentage-point
   changes.
6. Impact lags may differ between population groups and locations.
7. National-level indicators may conceal regional, gender and income disparities.
8. Some events may affect more than one indicator or operate through indirect
   pathways.
9. External economic, political, regulatory and technological developments are not
   fully represented.
10. Low-confidence relationships should not drive the baseline forecast.
11. Targets must remain separate from historical observations and realised outcomes.
12. Scenario conclusions should be tested against alternative weights and lag
   assumptions.
"""

    limitations_path = (
        REPORTS_DIR
        / "task_3_event_impact_limitations.md"
    )
    limitations_path.write_text(
        limitations,
        encoding="utf-8",
    )
    print(f"Saved report: {limitations_path.name}")


# ---------------------------------------------------------
# Main workflow
# ---------------------------------------------------------
def main():
    print("=" * 68)
    print("WEEK 11 — TASK 3 EVENT-TO-INDICATOR IMPACT ANALYSIS")
    print("=" * 68)

    impacts, records = load_data()
    prepared = prepare_impact_links(
        impacts,
        records,
    )

    (
        association_matrix,
        event_summary,
        indicator_summary,
        direction_summary,
        evidence_summary,
    ) = create_analysis_tables(prepared)

    plot_association_matrix(
        association_matrix
    )
    plot_event_impact_ranking(
        event_summary
    )
    plot_indicator_exposure(
        indicator_summary
    )
    plot_impact_directions(
        direction_summary
    )
    plot_lag_distribution(
        prepared
    )
    plot_confidence_distribution(
        prepared
    )

    create_reports(
        prepared,
        event_summary,
        indicator_summary,
        direction_summary,
    )

    print("\nTask 3 summary")
    print("-" * 68)
    print(
        f"Impact relationships analysed: {len(prepared)}"
    )
    print(
        f"Unique events: {prepared['event_id'].nunique()}"
    )
    print(
        "Distinct related indicators: "
        f"{prepared['related_indicator_clean'].nunique()}"
    )
    print(
        f"Association matrix dimensions: "
        f"{association_matrix.shape[0]} events × "
        f"{association_matrix.shape[1]} indicator groups"
    )
    print(f"Processed outputs: {PROCESSED_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Reports: {REPORTS_DIR}")

    print("\n" + "=" * 68)
    print("TASK 3 COMPLETED SUCCESSFULLY")
    print("=" * 68)


if __name__ == "__main__":
    main()