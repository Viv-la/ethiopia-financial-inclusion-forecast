"""
Task 4: Baseline Forecasting and Scenario Analysis
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

RECORDS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ethiopia_fi_cleaned_records.csv"
)

EVENT_LINKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "event_impact_scored_links.csv"
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
DARK_GREEN = "#174A3A"
AU_GOLD = "#F2A900"
BLUE = "#3572A5"
LIGHT_BLUE = "#8DB7D8"
RED = "#C44E52"
GREY = "#6B7280"
LIGHT_GREY = "#D1D5DB"

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
    """Clean text without changing its substantive meaning."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def normalise_key(value):
    """Create a lowercase matching key."""
    return clean_text(value).lower()


def normalise_columns(dataframe):
    """Standardise dataframe column names."""
    dataframe = dataframe.copy()
    dataframe.columns = [
        normalise_key(column).replace(" ", "_")
        for column in dataframe.columns
    ]
    return dataframe


def save_figure(filename):
    """Save and close a chart."""
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / filename,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print(f"Saved figure: {filename}")


def shorten_label(value, maximum=52):
    """Shorten long labels for charts."""
    text = clean_text(value)

    if len(text) <= maximum:
        return text

    return text[: maximum - 3] + "..."


def classify_indicator(value):
    """Assign indicators to broad analytical groups."""
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
            "saving",
            "financial service",
        ]
    ):
        return "Financial services"

    return clean_text(value) or "Other"


def safe_percentage_bound(values, value_type):
    """Apply logical percentage boundaries where appropriate."""
    values = np.asarray(values, dtype=float)
    unit_text = normalise_key(value_type)

    if "percent" in unit_text or "percentage" in unit_text:
        return np.clip(values, 0, 100)

    return np.maximum(values, 0)


# ---------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------
def load_data():
    if not RECORDS_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {RECORDS_FILE}"
        )

    records = pd.read_csv(RECORDS_FILE)
    records = normalise_columns(records)
    records = records.dropna(how="all").copy()

    required_columns = [
        "record_type",
        "indicator",
        "fiscal_year",
        "value_numeric",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in records.columns
    ]

    if missing_columns:
        raise ValueError(
            "The cleaned dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    records["record_type_clean"] = (
        records["record_type"]
        .apply(normalise_key)
    )

    records["indicator_clean"] = (
        records["indicator"]
        .apply(clean_text)
    )

    records["year_numeric"] = pd.to_numeric(
        records["fiscal_year"],
        errors="coerce",
    )

    records["value_numeric_clean"] = pd.to_numeric(
        records["value_numeric"],
        errors="coerce",
    )

    records["indicator_group"] = (
        records["indicator_clean"]
        .apply(classify_indicator)
    )

    if "value_type" not in records.columns:
        records["value_type"] = ""

    if "gender" not in records.columns:
        records["gender"] = ""

    if "location" not in records.columns:
        records["location"] = ""

    observations = records[
        records["record_type_clean"].eq("observation")
    ].copy()

    targets = records[
        records["record_type_clean"].eq("target")
    ].copy()

    observations = observations.dropna(
        subset=[
            "year_numeric",
            "value_numeric_clean",
            "indicator_clean",
        ]
    )

    targets = targets.dropna(
        subset=[
            "year_numeric",
            "value_numeric_clean",
            "indicator_clean",
        ]
    )

    observations["year_numeric"] = (
        observations["year_numeric"].astype(int)
    )

    if not targets.empty:
        targets["year_numeric"] = (
            targets["year_numeric"].astype(int)
        )

    if EVENT_LINKS_FILE.exists():
        event_links = pd.read_csv(EVENT_LINKS_FILE)
        event_links = normalise_columns(event_links)
    else:
        event_links = pd.DataFrame()

    print(f"Loaded {len(observations)} historical observations")
    print(f"Loaded {len(targets)} target records")
    print(f"Loaded {len(event_links)} event-impact relationships")

    return records, observations, targets, event_links


# ---------------------------------------------------------
# Select forecasting series
# ---------------------------------------------------------
def select_primary_series(observations):
    """
    Select comparable aggregate observations.

    Records without gender and location classifications are preferred
    because they generally represent the national aggregate series.
    """

    selected_groups = []

    for indicator, group in observations.groupby(
        "indicator_clean",
        dropna=False,
    ):
        group = group.copy()

        gender_blank = (
            group["gender"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        location_blank = (
            group["location"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        aggregate_group = group[
            gender_blank & location_blank
        ].copy()

        if aggregate_group["year_numeric"].nunique() >= 2:
            chosen = aggregate_group
        else:
            chosen = group

        chosen = (
            chosen.groupby(
                [
                    "indicator_clean",
                    "indicator_group",
                    "year_numeric",
                    "value_type",
                ],
                as_index=False,
                dropna=False,
            )["value_numeric_clean"]
            .mean()
        )

        if chosen["year_numeric"].nunique() >= 2:
            selected_groups.append(chosen)

    if not selected_groups:
        raise ValueError(
            "No indicator contains at least two usable historical observations."
        )

    selected = pd.concat(
        selected_groups,
        ignore_index=True,
    )

    selected = selected.sort_values(
        ["indicator_clean", "year_numeric"]
    )

    selected.to_csv(
        PROCESSED_DIR / "forecast_input_series.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Forecast-ready indicators: "
        f"{selected['indicator_clean'].nunique()}"
    )

    return selected


# ---------------------------------------------------------
# Event adjustment
# ---------------------------------------------------------
def calculate_event_adjustments(event_links):
    """
    Calculate conservative scenario adjustments.

    Event-impact scores are converted into modest annual scenario
    adjustments. They are not interpreted as measured percentage-point
    effects.
    """

    if event_links.empty:
        return {}

    if "indicator_group" not in event_links.columns:
        if "related_indicator_clean" in event_links.columns:
            event_links["indicator_group"] = (
                event_links["related_indicator_clean"]
                .apply(classify_indicator)
            )
        else:
            return {}

    if "weighted_impact_score" not in event_links.columns:
        return {}

    event_links["weighted_impact_score"] = pd.to_numeric(
        event_links["weighted_impact_score"],
        errors="coerce",
    ).fillna(0)

    group_scores = (
        event_links.groupby(
            "indicator_group",
            as_index=False,
        )["weighted_impact_score"]
        .sum()
    )

    adjustments = {}

    for _, row in group_scores.iterrows():
        group_name = clean_text(row["indicator_group"])
        raw_score = float(row["weighted_impact_score"])

        # Conservative cap: maximum ±2 percentage points annually.
        annual_adjustment = float(
            np.clip(raw_score * 0.35, -2.0, 2.0)
        )

        adjustments[group_name] = annual_adjustment

    return adjustments


# ---------------------------------------------------------
# Forecast modelling
# ---------------------------------------------------------
def fit_indicator_forecast(
    indicator_data,
    event_adjustment,
    forecast_end_year=2030,
):
    """Fit a parsimonious linear trend with uncertainty scenarios."""

    indicator_data = (
        indicator_data
        .sort_values("year_numeric")
        .copy()
    )

    indicator = indicator_data[
        "indicator_clean"
    ].iloc[0]

    indicator_group = indicator_data[
        "indicator_group"
    ].iloc[0]

    value_type = indicator_data[
        "value_type"
    ].iloc[0]

    historical = (
        indicator_data.groupby(
            "year_numeric",
            as_index=False,
        )["value_numeric_clean"]
        .mean()
        .sort_values("year_numeric")
    )

    years = historical[
        "year_numeric"
    ].to_numpy(dtype=float)

    values = historical[
        "value_numeric_clean"
    ].to_numpy(dtype=float)

    first_year = int(years.min())
    last_year = int(years.max())

    if len(years) < 2:
        return None

    centred_years = years - first_year

    slope, intercept = np.polyfit(
        centred_years,
        values,
        1,
    )

    fitted_historical = (
        intercept + slope * centred_years
    )

    residuals = values - fitted_historical

    if len(values) > 2:
        residual_standard_error = float(
            np.sqrt(
                np.sum(residuals ** 2)
                / max(len(values) - 2, 1)
            )
        )
    else:
        historical_change = abs(values[-1] - values[0])
        residual_standard_error = max(
            historical_change * 0.20,
            np.std(values) * 0.25,
            1.0,
        )

    forecast_start_year = last_year + 1

    if forecast_start_year > forecast_end_year:
        forecast_start_year = forecast_end_year

    forecast_years = np.arange(
        forecast_start_year,
        forecast_end_year + 1,
    )

    all_years = np.arange(
        first_year,
        forecast_end_year + 1,
    )

    baseline_all = (
        intercept
        + slope * (all_years - first_year)
    )

    baseline_all = safe_percentage_bound(
        baseline_all,
        value_type,
    )

    years_after_latest = np.maximum(
        all_years - last_year,
        0,
    )

    uncertainty_growth = (
        residual_standard_error
        * np.sqrt(
            1 + years_after_latest
        )
    )

    lower_all = safe_percentage_bound(
        baseline_all - 1.96 * uncertainty_growth,
        value_type,
    )

    upper_all = safe_percentage_bound(
        baseline_all + 1.96 * uncertainty_growth,
        value_type,
    )

    event_adjustment_path = (
        years_after_latest
        * event_adjustment
    )

    optimistic_all = safe_percentage_bound(
        baseline_all + event_adjustment_path,
        value_type,
    )

    conservative_all = safe_percentage_bound(
        baseline_all
        - np.abs(event_adjustment_path) * 0.50,
        value_type,
    )

    output = pd.DataFrame(
        {
            "indicator": indicator,
            "indicator_group": indicator_group,
            "year": all_years.astype(int),
            "baseline_forecast": baseline_all,
            "lower_95": lower_all,
            "upper_95": upper_all,
            "optimistic_scenario": optimistic_all,
            "conservative_scenario": conservative_all,
            "historical_last_year": last_year,
            "historical_slope": slope,
            "event_annual_adjustment": event_adjustment,
            "value_type": value_type,
        }
    )

    output["period_type"] = np.where(
        output["year"] <= last_year,
        "historical_fit",
        "forecast",
    )

    historical_actuals = historical.rename(
        columns={
            "year_numeric": "year",
            "value_numeric_clean": "actual_value",
        }
    )

    output = output.merge(
        historical_actuals,
        on="year",
        how="left",
    )

    model_summary = {
        "indicator": indicator,
        "indicator_group": indicator_group,
        "observations": len(values),
        "first_year": first_year,
        "last_year": last_year,
        "last_actual_value": float(values[-1]),
        "annual_historical_slope": float(slope),
        "residual_standard_error": residual_standard_error,
        "event_annual_adjustment": event_adjustment,
        "baseline_2030": float(
            output.loc[
                output["year"].eq(forecast_end_year),
                "baseline_forecast",
            ].iloc[0]
        ),
        "optimistic_2030": float(
            output.loc[
                output["year"].eq(forecast_end_year),
                "optimistic_scenario",
            ].iloc[0]
        ),
        "conservative_2030": float(
            output.loc[
                output["year"].eq(forecast_end_year),
                "conservative_scenario",
            ].iloc[0]
        ),
        "lower_95_2030": float(
            output.loc[
                output["year"].eq(forecast_end_year),
                "lower_95",
            ].iloc[0]
        ),
        "upper_95_2030": float(
            output.loc[
                output["year"].eq(forecast_end_year),
                "upper_95",
            ].iloc[0]
        ),
        "value_type": value_type,
    }

    return output, model_summary


def generate_forecasts(
    selected_series,
    event_adjustments,
):
    """Generate forecasts for every usable indicator."""

    forecast_frames = []
    summary_rows = []

    for indicator, indicator_data in selected_series.groupby(
        "indicator_clean"
    ):
        indicator_group = indicator_data[
            "indicator_group"
        ].iloc[0]

        event_adjustment = event_adjustments.get(
            indicator_group,
            0.0,
        )

        result = fit_indicator_forecast(
            indicator_data,
            event_adjustment,
            forecast_end_year=2030,
        )

        if result is None:
            continue

        forecast_frame, model_summary = result

        forecast_frames.append(forecast_frame)
        summary_rows.append(model_summary)

    if not forecast_frames:
        raise ValueError(
            "No forecasts could be generated from the available observations."
        )

    forecasts = pd.concat(
        forecast_frames,
        ignore_index=True,
    )

    model_summary = pd.DataFrame(summary_rows)

    forecasts.to_csv(
        PROCESSED_DIR / "financial_inclusion_forecasts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_summary.to_csv(
        PROCESSED_DIR / "forecast_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return forecasts, model_summary


# ---------------------------------------------------------
# Target comparison
# ---------------------------------------------------------
def create_target_comparison(
    forecasts,
    targets,
):
    """Compare forecast scenarios with official target records."""

    if targets.empty:
        comparison = pd.DataFrame(
            columns=[
                "target_indicator",
                "target_year",
                "target_value",
                "matched_forecast_indicator",
                "baseline_forecast",
                "optimistic_scenario",
                "conservative_scenario",
                "baseline_gap",
                "target_status",
            ]
        )

        comparison.to_csv(
            PROCESSED_DIR / "forecast_target_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )

        return comparison

    comparisons = []

    forecast_indicators = (
        forecasts["indicator"]
        .dropna()
        .unique()
        .tolist()
    )

    for _, target in targets.iterrows():
        target_indicator = clean_text(
            target["indicator_clean"]
        )

        target_group = classify_indicator(
            target_indicator
        )

        target_year = int(
            target["year_numeric"]
        )

        target_value = float(
            target["value_numeric_clean"]
        )

        exact_matches = [
            indicator
            for indicator in forecast_indicators
            if normalise_key(indicator)
            == normalise_key(target_indicator)
        ]

        if exact_matches:
            matched_indicator = exact_matches[0]
        else:
            group_matches = (
                forecasts[
                    forecasts["indicator_group"].eq(
                        target_group
                    )
                ]["indicator"]
                .dropna()
                .unique()
                .tolist()
            )

            if not group_matches:
                continue

            matched_indicator = group_matches[0]

        indicator_forecast = forecasts[
            forecasts["indicator"].eq(
                matched_indicator
            )
        ].copy()

        if target_year in indicator_forecast["year"].values:
            comparison_row = indicator_forecast[
                indicator_forecast["year"].eq(
                    target_year
                )
            ].iloc[0]
        else:
            comparison_row = indicator_forecast.iloc[-1]

        baseline_value = float(
            comparison_row["baseline_forecast"]
        )

        optimistic_value = float(
            comparison_row["optimistic_scenario"]
        )

        conservative_value = float(
            comparison_row["conservative_scenario"]
        )

        baseline_gap = baseline_value - target_value

        if baseline_value >= target_value:
            target_status = "Baseline meets or exceeds target"
        elif optimistic_value >= target_value:
            target_status = "Target possible under optimistic scenario"
        else:
            target_status = "Target remains above modelled scenarios"

        comparisons.append(
            {
                "target_indicator": target_indicator,
                "target_year": target_year,
                "target_value": target_value,
                "matched_forecast_indicator": matched_indicator,
                "comparison_year": int(
                    comparison_row["year"]
                ),
                "baseline_forecast": baseline_value,
                "optimistic_scenario": optimistic_value,
                "conservative_scenario": conservative_value,
                "baseline_gap": baseline_gap,
                "target_status": target_status,
            }
        )

    comparison = pd.DataFrame(comparisons)

    comparison.to_csv(
        PROCESSED_DIR / "forecast_target_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return comparison


# ---------------------------------------------------------
# Visualisations
# ---------------------------------------------------------
def plot_indicator_forecasts(forecasts):
    """Create individual forecast charts."""

    for chart_number, (
        indicator,
        data,
    ) in enumerate(
        forecasts.groupby("indicator"),
        start=14,
    ):
        data = data.sort_values("year")

        historical = data.dropna(
            subset=["actual_value"]
        )

        forecast_period = data[
            data["period_type"].eq("forecast")
        ]

        last_historical_year = int(
            data["historical_last_year"].iloc[0]
        )

        plt.figure(figsize=(11, 6))

        if not historical.empty:
            plt.plot(
                historical["year"],
                historical["actual_value"],
                marker="o",
                linewidth=2.5,
                color=DARK_GREEN,
                label="Historical observation",
                zorder=5,
            )

        plt.plot(
            data["year"],
            data["baseline_forecast"],
            linestyle="--",
            linewidth=2.2,
            color=BLUE,
            label="Baseline trend",
        )

        if not forecast_period.empty:
            plt.fill_between(
                forecast_period["year"],
                forecast_period["lower_95"],
                forecast_period["upper_95"],
                color=LIGHT_BLUE,
                alpha=0.28,
                label="95% uncertainty interval",
            )

            plt.plot(
                forecast_period["year"],
                forecast_period["optimistic_scenario"],
                linewidth=2,
                color=AU_GREEN,
                label="Event-adjusted optimistic",
            )

            plt.plot(
                forecast_period["year"],
                forecast_period["conservative_scenario"],
                linewidth=2,
                color=AU_GOLD,
                label="Conservative scenario",
            )

        plt.axvline(
            last_historical_year,
            color=GREY,
            linestyle=":",
            linewidth=1.5,
        )

        plt.text(
            last_historical_year + 0.1,
            plt.ylim()[1] * 0.96,
            "Forecast begins",
            color=GREY,
            va="top",
            fontsize=9,
        )

        plt.title(
            f"{indicator}: Historical Trend and Forecast to 2030"
        )
        plt.xlabel("Year")
        plt.ylabel(
            clean_text(data["value_type"].iloc[0])
            or "Indicator value"
        )
        plt.legend(
            loc="best",
            frameon=True,
        )
        sns.despine()

        safe_name = (
            normalise_key(indicator)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )

        safe_name = "".join(
            character
            for character in safe_name
            if character.isalnum() or character == "_"
        )

        save_figure(
            f"{chart_number:02d}_forecast_{safe_name[:45]}.png"
        )


def plot_2030_scenario_comparison(model_summary):
    """Compare 2030 forecast scenarios across indicators."""

    if model_summary.empty:
        return

    chart_data = (
        model_summary.sort_values(
            "baseline_2030",
            ascending=True,
        )
        .copy()
    )

    chart_data["label"] = (
        chart_data["indicator"]
        .apply(shorten_label)
    )

    y_positions = np.arange(
        len(chart_data)
    )

    height = max(
        6,
        len(chart_data) * 0.65,
    )

    plt.figure(figsize=(12, height))

    plt.scatter(
        chart_data["conservative_2030"],
        y_positions,
        color=AU_GOLD,
        s=75,
        label="Conservative",
        zorder=4,
    )

    plt.scatter(
        chart_data["baseline_2030"],
        y_positions,
        color=BLUE,
        s=75,
        label="Baseline",
        zorder=4,
    )

    plt.scatter(
        chart_data["optimistic_2030"],
        y_positions,
        color=AU_GREEN,
        s=75,
        label="Optimistic",
        zorder=4,
    )

    for position, (_, row) in zip(
        y_positions,
        chart_data.iterrows(),
    ):
        values = [
            row["conservative_2030"],
            row["baseline_2030"],
            row["optimistic_2030"],
        ]

        plt.plot(
            [min(values), max(values)],
            [position, position],
            color=LIGHT_GREY,
            linewidth=2,
            zorder=1,
        )

    plt.yticks(
        y_positions,
        chart_data["label"],
    )

    plt.title("Comparison of Financial-Inclusion Scenarios in 2030")
    plt.xlabel("Forecast indicator value")
    plt.ylabel("")
    plt.legend()
    sns.despine(left=True, bottom=True)
    save_figure("20_2030_scenario_comparison.png")


def plot_target_gaps(target_comparison):
    """Plot baseline gaps against official targets."""

    if target_comparison.empty:
        return

    chart_data = target_comparison.copy()

    chart_data["label"] = (
        chart_data["target_indicator"]
        .apply(shorten_label)
    )

    chart_data = chart_data.sort_values(
        "baseline_gap",
        ascending=True,
    )

    colors = [
        AU_GREEN if gap >= 0 else RED
        for gap in chart_data["baseline_gap"]
    ]

    plt.figure(
        figsize=(
            11,
            max(5, len(chart_data) * 0.7),
        )
    )

    bars = plt.barh(
        chart_data["label"],
        chart_data["baseline_gap"],
        color=colors,
    )

    plt.axvline(
        0,
        color=GREY,
        linewidth=1.2,
    )

    for bar, value in zip(
        bars,
        chart_data["baseline_gap"],
    ):
        plt.text(
            value + (0.2 if value >= 0 else -0.2),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left" if value >= 0 else "right",
        )

    plt.title("Baseline Forecast Gap Against Official Targets")
    plt.xlabel(
        "Forecast minus target value"
    )
    plt.ylabel("")
    sns.despine(left=True, bottom=True)
    save_figure("21_forecast_target_gap.png")


def plot_forecast_uncertainty(model_summary):
    """Visualise the width of 2030 uncertainty intervals."""

    if model_summary.empty:
        return

    chart_data = model_summary.copy()

    chart_data["uncertainty_width"] = (
        chart_data["upper_95_2030"]
        - chart_data["lower_95_2030"]
    )

    chart_data["label"] = (
        chart_data["indicator"]
        .apply(shorten_label)
    )

    chart_data = chart_data.sort_values(
        "uncertainty_width",
        ascending=True,
    )

    plt.figure(
        figsize=(
            11,
            max(6, len(chart_data) * 0.65),
        )
    )

    bars = plt.barh(
        chart_data["label"],
        chart_data["uncertainty_width"],
        color=LIGHT_BLUE,
    )

    for bar, value in zip(
        bars,
        chart_data["uncertainty_width"],
    ):
        plt.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
        )

    plt.title("Forecast Uncertainty Width in 2030")
    plt.xlabel(
        "Width of 95% uncertainty interval"
    )
    plt.ylabel("")
    sns.despine(left=True, bottom=True)
    save_figure("22_forecast_uncertainty.png")


# ---------------------------------------------------------
# Reports
# ---------------------------------------------------------
def create_reports(
    observations,
    selected_series,
    forecasts,
    model_summary,
    target_comparison,
):
    total_observations = len(observations)
    forecast_observations = len(selected_series)
    total_indicators = model_summary["indicator"].nunique()

    earliest_year = int(
        selected_series["year_numeric"].min()
    )

    latest_year = int(
        selected_series["year_numeric"].max()
    )

    strongest_growth = (
        model_summary.sort_values(
            "annual_historical_slope",
            ascending=False,
        ).iloc[0]
    )

    weakest_growth = (
        model_summary.sort_values(
            "annual_historical_slope",
            ascending=True,
        ).iloc[0]
    )

    widest_uncertainty = model_summary.copy()

    widest_uncertainty["uncertainty_width"] = (
        widest_uncertainty["upper_95_2030"]
        - widest_uncertainty["lower_95_2030"]
    )

    widest_uncertainty = (
        widest_uncertainty.sort_values(
            "uncertainty_width",
            ascending=False,
        ).iloc[0]
    )

    if target_comparison.empty:
        target_statement = (
            "Target records were retained separately, but no sufficiently "
            "comparable forecast-target matches were available."
        )
        targets_met = 0
        targets_possible = 0
        targets_missed = 0
    else:
        targets_met = int(
            target_comparison["target_status"]
            .eq("Baseline meets or exceeds target")
            .sum()
        )

        targets_possible = int(
            target_comparison["target_status"]
            .eq(
                "Target possible under optimistic scenario"
            )
            .sum()
        )

        targets_missed = int(
            target_comparison["target_status"]
            .eq(
                "Target remains above modelled scenarios"
            )
            .sum()
        )

        target_statement = (
            f"The target comparison contains **{len(target_comparison)} "
            f"matched records**: **{targets_met}** are met or exceeded by "
            f"the baseline, **{targets_possible}** may be reached under the "
            f"optimistic scenario, and **{targets_missed}** remain above "
            f"the modelled scenarios."
        )

    findings = f"""# Task 4 Forecasting and Scenario Analysis

## Analysis overview

Task 4 developed transparent baseline and event-adjusted forecasts for
Ethiopia's financial-inclusion indicators through 2030.

The forecasting process used **{forecast_observations} comparable historical
records** selected from **{total_observations} available observations**.
The final modelling dataset covers **{total_indicators} indicators** between
**{earliest_year} and {latest_year}**.

Because the available time series are short and irregular, the analysis uses
parsimonious linear trends instead of complex machine-learning models.

## Forecasting approach

For each indicator, the analysis:

1. Selected comparable historical observations.
2. Preferred national aggregate records without gender or location filters.
3. Estimated a linear historical trend.
4. Extended the baseline trend to 2030.
5. Calculated widening uncertainty intervals.
6. Applied confidence-weighted event adjustments conservatively.
7. Created baseline, optimistic and conservative scenarios.
8. Compared modelled results with target records where appropriate.

Event-impact scores are scenario weights. They are not interpreted as directly
measured percentage-point effects.

## Key findings

1. **Historical evidence remains limited.**  
   The modelling series cover {earliest_year}–{latest_year}, but observations
   are irregular and some indicators contain only a small number of measured
   years. Forecasts should therefore be treated as directional estimates.

2. **The strongest historical growth occurs in
   `{clean_text(strongest_growth["indicator"])}`.**  
   Its fitted historical trend is approximately
   **{strongest_growth["annual_historical_slope"]:.2f} units per year**.

3. **The weakest fitted trend occurs in
   `{clean_text(weakest_growth["indicator"])}`.**  
   Its estimated trend is approximately
   **{weakest_growth["annual_historical_slope"]:.2f} units per year**.
   A negative or weak slope should be interpreted cautiously because sparse
   observations may not represent a stable long-term pattern.

4. **Event-adjusted scenarios show the potential importance of reforms.**  
   The optimistic scenario applies documented event relationships gradually
   after the latest observation. The conservative scenario assumes weaker
   progress and does not treat policy announcements as guaranteed outcomes.

5. **Uncertainty increases with the forecast horizon.**  
   `{clean_text(widest_uncertainty["indicator"])}` has the widest modelled
   2030 interval, reflecting historical variability and limited observations.
   The width of its 95% interval is approximately
   **{widest_uncertainty["uncertainty_width"]:.2f} units**.

6. **Targets remain separate from forecasts.**  
   {target_statement}

7. **Forecast results are scenarios, not certainties.**  
   Macroeconomic conditions, regulation, infrastructure, affordability,
   digital adoption and programme implementation may cause actual outcomes
   to differ materially from the projected values.

## Policy implications

- Sustain investments in digital and physical financial infrastructure.
- Prioritise implementation quality rather than relying on policy adoption alone.
- Address affordability, trust, digital literacy and consumer-protection barriers.
- Monitor gender and regional gaps alongside national averages.
- Improve annual indicator collection to reduce forecast uncertainty.
- Establish regular progress reviews against official 2030 targets.
- Update forecasts whenever new nationally representative observations become available.

## Generated outputs

Task 4 generated:

- Forecast-ready historical series
- Indicator-level baseline forecasts to 2030
- Event-adjusted optimistic and conservative scenarios
- Widening 95% uncertainty intervals
- Model diagnostic and summary tables
- Forecast-to-target comparison table
- Individual historical and forecast charts
- 2030 scenario comparison chart
- Target-gap chart, where comparable targets exist
- Forecast-uncertainty chart
"""

    findings_path = (
        REPORTS_DIR
        / "task_4_forecasting_findings.md"
    )

    findings_path.write_text(
        findings,
        encoding="utf-8",
    )

    print(f"Saved report: {findings_path.name}")

    limitations = """# Task 4 Forecasting Limitations

1. Historical time series are short and irregularly spaced.
2. Some indicators contain only two or three usable observations.
3. Linear trends simplify relationships that may be nonlinear.
4. Forecasts are sensitive to unusually high or low historical observations.
5. Survey definitions and measurement approaches may differ across years.
6. Calendar-year and fiscal-year reporting may not always be fully comparable.
7. Gender and location disaggregation are incomplete.
8. National averages may conceal regional and population-group inequalities.
9. Event-impact scores are scenario weights, not causal effect estimates.
10. Event implementation quality cannot be determined from announcement dates alone.
11. Multiple reforms and external shocks may affect indicators simultaneously.
12. Macroeconomic, political and regulatory changes are not modelled independently.
13. Confidence intervals represent model uncertainty but cannot capture every future shock.
14. Target values represent intended outcomes, not realised historical measurements.
15. Forecasts should be updated when new observations become available.
16. Results should support planning and monitoring, not replace expert judgement.
"""

    limitations_path = (
        REPORTS_DIR
        / "task_4_forecasting_limitations.md"
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
    print("=" * 70)
    print("WEEK 11 — TASK 4 FORECASTING AND SCENARIO ANALYSIS")
    print("=" * 70)

    (
        records,
        observations,
        targets,
        event_links,
    ) = load_data()

    selected_series = select_primary_series(
        observations
    )

    event_adjustments = calculate_event_adjustments(
        event_links
    )

    forecasts, model_summary = generate_forecasts(
        selected_series,
        event_adjustments,
    )

    target_comparison = create_target_comparison(
        forecasts,
        targets,
    )

    plot_indicator_forecasts(
        forecasts
    )

    plot_2030_scenario_comparison(
        model_summary
    )

    plot_target_gaps(
        target_comparison
    )

    plot_forecast_uncertainty(
        model_summary
    )

    create_reports(
        observations,
        selected_series,
        forecasts,
        model_summary,
        target_comparison,
    )

    print("\nTask 4 summary")
    print("-" * 70)
    print(
        f"Historical observations available: {len(observations)}"
    )
    print(
        f"Forecast input observations: {len(selected_series)}"
    )
    print(
        f"Indicators forecast: {model_summary['indicator'].nunique()}"
    )
    print(
        f"Forecast records generated: {len(forecasts)}"
    )
    print(
        f"Target comparisons generated: {len(target_comparison)}"
    )
    print(f"Forecast horizon: 2030")
    print(f"Processed outputs: {PROCESSED_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Reports: {REPORTS_DIR}")

    print("\n" + "=" * 70)
    print("TASK 4 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()