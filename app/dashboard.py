"""
Week 11 Project: Forecasting Financial Inclusion in Ethiopia
Task 5: Interactive Streamlit Dashboard
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Ethiopia Financial Inclusion Forecast",
    page_icon="🇪🇹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# COLOURS
# ============================================================
AU_GREEN = "#00843D"
DARK_GREEN = "#174A3A"
ETHIOPIA_RED = "#DA121A"
AU_GOLD = "#F2A900"
BLUE = "#2F75B5"
LIGHT_BLUE = "#DCEAF7"
GREY = "#6B7280"
DARK_TEXT = "#1F2937"


# ============================================================
# FILE LOCATIONS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_csv(*keywords):
    """Find the first CSV whose filename contains the supplied keywords."""
    csv_files = list(PROJECT_ROOT.rglob("*.csv"))

    for csv_file in csv_files:
        filename = csv_file.name.lower()

        if all(keyword.lower() in filename for keyword in keywords):
            return csv_file

    return None


RECORDS_FILE = find_csv("cleaned", "record") or find_csv("record")
EVENTS_FILE = find_csv("event", "impact") or find_csv("event", "link")
EVENT_SUMMARY_FILE = find_csv("event", "summary")
INDICATOR_IMPACT_FILE = find_csv("indicator", "impact")
ASSOCIATION_FILE = find_csv("association")
FORECASTS_FILE = find_csv("forecast") or find_csv("scenario")
MODEL_SUMMARY_FILE = (
    find_csv("model", "summary")
    or find_csv("forecast", "summary")
)
TARGET_COMPARISON_FILE = (
    find_csv("target", "comparison")
    or find_csv("forecast", "target")
)


# ============================================================
# STYLING
# ============================================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #F7F9FB;
        color: {DARK_TEXT};
    }}

    .main-header {{
        background: linear-gradient(
            120deg,
            {DARK_GREEN},
            {AU_GREEN}
        );
        padding: 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
    }}

    .main-header h1 {{
        color: white;
        margin: 0;
        font-size: 2.2rem;
    }}

    .main-header p {{
        color: #EAF7EF;
        margin: 0.6rem 0 0 0;
        font-size: 1.02rem;
    }}

    .section-header {{
        color: {DARK_GREEN};
        font-size: 1.35rem;
        font-weight: 700;
        border-bottom: 3px solid {AU_GOLD};
        padding-bottom: 0.35rem;
        margin: 1.3rem 0 1rem 0;
    }}

    .insight-box {{
        background-color: white;
        color: {DARK_TEXT} !important;
        border-left: 5px solid {AU_GREEN};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }}

    .insight-box strong {{
        color: {DARK_GREEN} !important;
    }}

    .warning-box {{
        background-color: #FFF8E6;
        color: #374151 !important;
        border-left: 5px solid {AU_GOLD};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.75rem 0;
    }}

    .warning-box strong {{
        color: #7A5200 !important;
    }}

    .source-box {{
        background-color: #EEF4F8;
        color: #374151 !important;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.75rem 0;
    }}

    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }}

    div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {{
    color: #374151 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}}

    div[data-testid="stMetricValue"] {{
        color: {DARK_GREEN};
    }}

    .footer {{
        text-align: center;
        color: {GREY};
        padding: 2rem 0 0.5rem 0;
        font-size: 0.88rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def clean_text(value):
    if pd.isna(value):
        return ""

    return " ".join(str(value).strip().split())


def normalise_columns(dataframe):
    dataframe = dataframe.copy()
    dataframe.columns = [
        clean_text(column)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        for column in dataframe.columns
    ]
    return dataframe


def read_csv_safely(file_path):
    if file_path is None or not file_path.exists():
        return pd.DataFrame()

    try:
        return normalise_columns(pd.read_csv(file_path))
    except Exception as error:
        st.warning(f"Could not read {file_path.name}: {error}")
        return pd.DataFrame()


def find_first_column(dataframe, candidates):
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    return None


def format_number(value, decimals=1):
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


def shorten_label(value, maximum=48):
    text = clean_text(value)

    if len(text) <= maximum:
        return text

    return text[: maximum - 3] + "..."


def display_header(title, subtitle):
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_section(title):
    st.markdown(
        f'<div class="section-header">{title}</div>',
        unsafe_allow_html=True,
    )


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", linestyle="--", alpha=0.25)


def to_csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_dashboard_data():
    return {
        "records": read_csv_safely(RECORDS_FILE),
        "event_links": read_csv_safely(EVENTS_FILE),
        "event_summary": read_csv_safely(EVENT_SUMMARY_FILE),
        "indicator_impact": read_csv_safely(INDICATOR_IMPACT_FILE),
        "association_matrix": read_csv_safely(ASSOCIATION_FILE),
        "forecasts": read_csv_safely(FORECASTS_FILE),
        "model_summary": read_csv_safely(MODEL_SUMMARY_FILE),
        "target_comparison": read_csv_safely(
            TARGET_COMPARISON_FILE
        ),
    }


data = load_dashboard_data()

records = data["records"]
event_links = data["event_links"]
event_summary = data["event_summary"]
indicator_impact = data["indicator_impact"]
association_matrix = data["association_matrix"]
forecasts = data["forecasts"]
model_summary = data["model_summary"]
target_comparison = data["target_comparison"]


# ============================================================
# PREPARE HISTORICAL DATA
# ============================================================
historical_records = pd.DataFrame()
target_records = pd.DataFrame()

if not records.empty:
    record_type_column = find_first_column(
        records,
        ["record_type", "type"],
    )

    year_column = find_first_column(
        records,
        [
            "fiscal_year",
            "analytical_year",
            "year",
            "calendar_year",
        ],
    )

    value_column = find_first_column(
        records,
        [
            "value_numeric",
            "numeric_value",
            "value",
        ],
    )

    indicator_column = find_first_column(
        records,
        ["indicator", "indicator_name"],
    )

    if year_column:
        records["dashboard_year"] = pd.to_numeric(
            records[year_column],
            errors="coerce",
        )

    if value_column:
        records["dashboard_value"] = pd.to_numeric(
            records[value_column],
            errors="coerce",
        )

    if indicator_column:
        records["dashboard_indicator"] = records[
            indicator_column
        ].apply(clean_text)

    if record_type_column:
        records["dashboard_record_type"] = (
            records[record_type_column]
            .apply(clean_text)
            .str.lower()
        )
    else:
        records["dashboard_record_type"] = "observation"

    historical_records = records[
        records["dashboard_record_type"].eq("observation")
    ].copy()

    target_records = records[
        records["dashboard_record_type"].eq("target")
    ].copy()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🇪🇹 Dashboard Navigation")

    page = st.radio(
        "Select a page",
        [
            "Executive Overview",
            "Historical Trends",
            "Event Impact Analysis",
            "Forecasts to 2030",
            "Target Assessment",
            "Data Explorer",
            "Methodology",
        ],
    )

    st.divider()

    st.markdown("### Project scope")
    st.caption(
        "Historical evidence, policy events and financial-inclusion "
        "forecasts for Ethiopia."
    )

    available_files = sum(
        not dataframe.empty for dataframe in data.values()
    )

    st.metric("Analytical datasets", available_files)

    if st.button("Refresh dashboard data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Week 11 Financial-Inclusion Forecasting Project")


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================
if page == "Executive Overview":
    display_header(
        "Ethiopia Financial Inclusion Forecast",
        "Interactive evidence dashboard integrating historical "
        "indicators, policy events and forecast scenarios through 2030.",
    )

    event_name_column = find_first_column(
        event_links,
        ["event_name", "event", "event_title", "title"],
    )

    total_events = (
        event_links[event_name_column].nunique()
        if not event_links.empty and event_name_column
        else len(event_links)
    )

    historical_indicators = (
        historical_records["dashboard_indicator"].nunique()
        if (
            not historical_records.empty
            and "dashboard_indicator"
            in historical_records.columns
        )
        else 0
    )

    forecast_indicators = (
        model_summary["indicator"].nunique()
        if (
            not model_summary.empty
            and "indicator" in model_summary.columns
        )
        else (
            forecasts["indicator"].nunique()
            if (
                not forecasts.empty
                and "indicator" in forecasts.columns
            )
            else 0
        )
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Historical observations",
        len(historical_records),
    )
    metric_2.metric(
        "Indicators assessed",
        historical_indicators,
    )
    metric_3.metric(
        "Policy events analysed",
        total_events,
    )
    metric_4.metric(
        "Indicators forecast",
        forecast_indicators,
    )

    left_column, right_column = st.columns([1.4, 1])

    with left_column:
        display_section("2030 forecast outlook")

        required_columns = {
            "indicator",
            "last_actual_value",
            "baseline_2030",
            "optimistic_2030",
            "conservative_2030",
        }

        if (
            not model_summary.empty
            and required_columns.issubset(model_summary.columns)
        ):
            chart_data = model_summary.copy()

            numeric_columns = [
                "last_actual_value",
                "baseline_2030",
                "optimistic_2030",
                "conservative_2030",
            ]

            for column in numeric_columns:
                chart_data[column] = pd.to_numeric(
                    chart_data[column],
                    errors="coerce",
                )

            chart_data = chart_data[
                chart_data["last_actual_value"].notna()
                & chart_data["last_actual_value"].ne(0)
            ].copy()

            for scenario in [
                "baseline",
                "optimistic",
                "conservative",
            ]:
                chart_data[f"{scenario}_change"] = (
                    (
                        chart_data[f"{scenario}_2030"]
                        - chart_data["last_actual_value"]
                    )
                    / chart_data["last_actual_value"]
                    * 100
                )

            chart_data = chart_data.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna(subset=["baseline_change"])

            chart_data = chart_data.sort_values(
                "baseline_change"
            )

            chart_data["short_indicator"] = chart_data[
                "indicator"
            ].apply(shorten_label)

            figure, axis = plt.subplots(figsize=(10, 5.8))

            positions = np.arange(len(chart_data))
            bar_height = 0.23

            axis.barh(
                positions - bar_height,
                chart_data["conservative_change"],
                height=bar_height,
                color=AU_GOLD,
                label="Conservative",
            )

            axis.barh(
                positions,
                chart_data["baseline_change"],
                height=bar_height,
                color=BLUE,
                label="Baseline",
            )

            axis.barh(
                positions + bar_height,
                chart_data["optimistic_change"],
                height=bar_height,
                color=AU_GREEN,
                label="Optimistic",
            )

            axis.set_yticks(positions)
            axis.set_yticklabels(
                chart_data["short_indicator"]
            )
            axis.set_xlabel(
                "Change from latest observation (%)"
            )
            axis.set_title(
                "Projected Change to 2030",
                fontweight="bold",
            )
            axis.axvline(0, color=GREY, linewidth=1)
            axis.legend(frameon=False)

            style_axis(axis)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)

        else:
            st.info(
                "The complete forecast summary was not found. "
                "Use the Data Explorer to inspect the available files."
            )

    with right_column:
        display_section("Decision summary")

        if (
            not model_summary.empty
            and "annual_historical_slope"
            in model_summary.columns
        ):
            trend_data = model_summary.copy()
            trend_data["annual_historical_slope"] = pd.to_numeric(
                trend_data["annual_historical_slope"],
                errors="coerce",
            )
            trend_data = trend_data.dropna(
                subset=["annual_historical_slope"]
            )

            if not trend_data.empty:
                strongest = trend_data.loc[
                    trend_data["annual_historical_slope"].idxmax()
                ]

                weakest = trend_data.loc[
                    trend_data["annual_historical_slope"].idxmin()
                ]

                st.markdown(
                    f"""
                    <div class="insight-box">
                        <strong>Strongest historical growth</strong><br>
                        {clean_text(strongest.get("indicator", "N/A"))}<br>
                        Annual fitted change:
                        {format_number(
                            strongest.get("annual_historical_slope"),
                            2
                        )}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="insight-box">
                        <strong>Weakest historical trend</strong><br>
                        {clean_text(weakest.get("indicator", "N/A"))}<br>
                        Annual fitted change:
                        {format_number(
                            weakest.get("annual_historical_slope"),
                            2
                        )}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if not target_comparison.empty:
            st.markdown(
                f"""
                <div class="insight-box">
                    <strong>Target assessment</strong><br>
                    {len(target_comparison)} comparable target records
                    were assessed across the forecast scenarios.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="warning-box">
                <strong>Interpretation note</strong><br>
                These forecasts are transparent planning scenarios.
                They are not guaranteed outcomes or causal estimates.
            </div>
            """,
            unsafe_allow_html=True,
        )

    display_section("Historical evidence coverage")

    if {
        "dashboard_year",
        "dashboard_indicator",
        "dashboard_value",
    }.issubset(historical_records.columns):
        coverage = (
            historical_records.dropna(
                subset=[
                    "dashboard_year",
                    "dashboard_indicator",
                ]
            )
            .groupby("dashboard_indicator")
            .agg(
                observations=("dashboard_value", "count"),
                first_year=("dashboard_year", "min"),
                latest_year=("dashboard_year", "max"),
            )
            .reset_index()
            .sort_values("observations", ascending=False)
        )

        st.dataframe(
            coverage,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Historical coverage information is unavailable.")


# ============================================================
# HISTORICAL TRENDS
# ============================================================
elif page == "Historical Trends":
    display_header(
        "Historical Financial-Inclusion Trends",
        "Explore available indicators and compare changes over time.",
    )

    required_columns = {
        "dashboard_year",
        "dashboard_indicator",
        "dashboard_value",
    }

    if historical_records.empty or not required_columns.issubset(
        historical_records.columns
    ):
        st.error("No usable historical observation records were found.")
    else:
        indicators = sorted(
            historical_records["dashboard_indicator"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_indicators = st.multiselect(
            "Select indicators",
            options=indicators,
            default=indicators[: min(4, len(indicators))],
        )

        filtered = historical_records[
            historical_records["dashboard_indicator"].isin(
                selected_indicators
            )
        ].copy()

        if filtered.empty:
            st.info("Select at least one indicator.")
        else:
            display_section("Indicator trends")

            figure, axis = plt.subplots(figsize=(12, 6))

            for indicator, group in filtered.groupby(
                "dashboard_indicator"
            ):
                grouped = (
                    group.dropna(
                        subset=[
                            "dashboard_year",
                            "dashboard_value",
                        ]
                    )
                    .groupby(
                        "dashboard_year",
                        as_index=False,
                    )["dashboard_value"]
                    .mean()
                    .sort_values("dashboard_year")
                )

                axis.plot(
                    grouped["dashboard_year"],
                    grouped["dashboard_value"],
                    marker="o",
                    linewidth=2.2,
                    label=shorten_label(indicator, 40),
                )

            axis.set_title(
                "Historical Financial-Inclusion Indicators",
                fontweight="bold",
            )
            axis.set_xlabel("Year")
            axis.set_ylabel("Indicator value")
            axis.legend(frameon=False)

            style_axis(axis)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)

            display_section("Historical records")

            display_columns = [
                column
                for column in [
                    "dashboard_year",
                    "dashboard_indicator",
                    "dashboard_value",
                    "value_type",
                    "gender",
                    "location",
                    "source_name",
                    "source",
                ]
                if column in filtered.columns
            ]

            st.dataframe(
                filtered[display_columns],
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download filtered historical data",
                data=to_csv_bytes(filtered[display_columns]),
                file_name="ethiopia_historical_indicators.csv",
                mime="text/csv",
            )


# ============================================================
# EVENT IMPACT ANALYSIS
# ============================================================
elif page == "Event Impact Analysis":
    display_header(
        "Event Impact Analysis",
        "Examine documented relationships between major events, "
        "reforms and financial-inclusion indicators.",
    )

    if event_links.empty:
        st.error("The event-impact dataset was not found.")
    else:
        event_name_column = find_first_column(
            event_links,
            ["event_name", "event", "event_title", "title"],
        )

        weighted_score_column = find_first_column(
            event_links,
            [
                "weighted_impact_score",
                "impact_score",
                "weighted_score",
            ],
        )

        indicator_column = find_first_column(
            event_links,
            [
                "indicator_group",
                "related_indicator_clean",
                "related_indicator",
                "indicator",
            ],
        )

        direction_column = find_first_column(
            event_links,
            ["impact_direction", "direction"],
        )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        metric_1.metric("Event-indicator links", len(event_links))
        metric_2.metric(
            "Unique events",
            (
                event_links[event_name_column].nunique()
                if event_name_column
                else "N/A"
            ),
        )
        metric_3.metric(
            "Indicators affected",
            (
                event_links[indicator_column].nunique()
                if indicator_column
                else "N/A"
            ),
        )

        if weighted_score_column:
            event_links[weighted_score_column] = pd.to_numeric(
                event_links[weighted_score_column],
                errors="coerce",
            )
            average_score = event_links[
                weighted_score_column
            ].mean()
        else:
            average_score = np.nan

        metric_4.metric(
            "Average weighted score",
            format_number(average_score, 2),
        )

        left_column, right_column = st.columns(2)

        with left_column:
            display_section("Impact by event")

            if event_name_column and weighted_score_column:
                chart = (
                    event_links.groupby(
                        event_name_column,
                        as_index=False,
                    )[weighted_score_column]
                    .sum()
                    .sort_values(weighted_score_column)
                )

                chart["label"] = chart[
                    event_name_column
                ].apply(shorten_label)

                colours = [
                    AU_GREEN if value >= 0 else ETHIOPIA_RED
                    for value in chart[weighted_score_column]
                ]

                figure, axis = plt.subplots(figsize=(9, 6))
                axis.barh(
                    chart["label"],
                    chart[weighted_score_column],
                    color=colours,
                )
                axis.axvline(0, color=GREY, linewidth=1)
                axis.set_title(
                    "Confidence-Weighted Impact by Event",
                    fontweight="bold",
                )
                axis.set_xlabel("Cumulative weighted impact")

                style_axis(axis)
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

        with right_column:
            display_section("Impact by indicator")

            if indicator_column and weighted_score_column:
                chart = (
                    event_links.groupby(
                        indicator_column,
                        as_index=False,
                    )[weighted_score_column]
                    .sum()
                    .sort_values(weighted_score_column)
                )

                chart["label"] = chart[
                    indicator_column
                ].apply(shorten_label)

                colours = [
                    AU_GREEN if value >= 0 else ETHIOPIA_RED
                    for value in chart[weighted_score_column]
                ]

                figure, axis = plt.subplots(figsize=(9, 6))
                axis.barh(
                    chart["label"],
                    chart[weighted_score_column],
                    color=colours,
                )
                axis.axvline(0, color=GREY, linewidth=1)
                axis.set_title(
                    "Cumulative Impact by Indicator",
                    fontweight="bold",
                )
                axis.set_xlabel("Cumulative weighted impact")

                style_axis(axis)
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

        if direction_column:
            display_section("Impact direction distribution")

            direction_counts = (
                event_links[direction_column]
                .fillna("Not classified")
                .value_counts()
            )

            figure, axis = plt.subplots(figsize=(9, 4.5))
            axis.bar(
                direction_counts.index,
                direction_counts.values,
                color=[
                    AU_GREEN,
                    ETHIOPIA_RED,
                    AU_GOLD,
                    BLUE,
                    GREY,
                ][: len(direction_counts)],
            )
            axis.set_ylabel("Number of links")
            axis.tick_params(axis="x", rotation=20)
            axis.set_title(
                "Direction of Documented Event Impacts",
                fontweight="bold",
            )

            style_axis(axis)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)

        display_section("Event-impact evidence")

        st.dataframe(
            event_links,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download event-impact records",
            data=to_csv_bytes(event_links),
            file_name="event_impact_analysis.csv",
            mime="text/csv",
        )


# ============================================================
# FORECASTS TO 2030
# ============================================================
elif page == "Forecasts to 2030":
    display_header(
        "Financial-Inclusion Forecasts to 2030",
        "Compare historical observations with baseline, optimistic "
        "and conservative scenarios.",
    )

    required_columns = {"indicator", "year"}

    if forecasts.empty or not required_columns.issubset(
        forecasts.columns
    ):
        st.error("The Task 4 forecast dataset was not found.")
    else:
        indicators = sorted(
            forecasts["indicator"].dropna().unique().tolist()
        )

        selected_indicator = st.selectbox(
            "Select an indicator",
            indicators,
        )

        selected = forecasts[
            forecasts["indicator"].eq(selected_indicator)
        ].copy()

        selected["year"] = pd.to_numeric(
            selected["year"],
            errors="coerce",
        )
        selected = selected.sort_values("year")

        numeric_columns = [
            "actual_value",
            "baseline_forecast",
            "lower_95",
            "upper_95",
            "optimistic_scenario",
            "conservative_scenario",
        ]

        for column in numeric_columns:
            if column in selected.columns:
                selected[column] = pd.to_numeric(
                    selected[column],
                    errors="coerce",
                )

        summary = pd.Series(dtype=object)

        if (
            not model_summary.empty
            and "indicator" in model_summary.columns
        ):
            summary_rows = model_summary[
                model_summary["indicator"].eq(selected_indicator)
            ]

            if not summary_rows.empty:
                summary = summary_rows.iloc[0]

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        metric_1.metric(
            "Latest actual",
            format_number(summary.get("last_actual_value", np.nan)),
        )
        metric_2.metric(
            "Baseline 2030",
            format_number(summary.get("baseline_2030", np.nan)),
        )
        metric_3.metric(
            "Optimistic 2030",
            format_number(summary.get("optimistic_2030", np.nan)),
        )
        metric_4.metric(
            "Conservative 2030",
            format_number(
                summary.get("conservative_2030", np.nan)
            ),
        )

        display_section("Historical trend and forecast scenarios")

        figure, axis = plt.subplots(figsize=(12, 6.5))

        if "actual_value" in selected.columns:
            historical = selected.dropna(subset=["actual_value"])

            axis.plot(
                historical["year"],
                historical["actual_value"],
                color=DARK_GREEN,
                marker="o",
                linewidth=2.8,
                label="Historical observation",
            )

        if "baseline_forecast" in selected.columns:
            axis.plot(
                selected["year"],
                selected["baseline_forecast"],
                color=BLUE,
                linestyle="--",
                linewidth=2.3,
                label="Baseline forecast",
            )

        if {
            "lower_95",
            "upper_95",
        }.issubset(selected.columns):
            uncertainty = selected.dropna(
                subset=["year", "lower_95", "upper_95"]
            )

            axis.fill_between(
                uncertainty["year"],
                uncertainty["lower_95"],
                uncertainty["upper_95"],
                color=LIGHT_BLUE,
                alpha=0.4,
                label="95% uncertainty interval",
            )

        if "optimistic_scenario" in selected.columns:
            axis.plot(
                selected["year"],
                selected["optimistic_scenario"],
                color=AU_GREEN,
                linewidth=2.3,
                label="Optimistic scenario",
            )

        if "conservative_scenario" in selected.columns:
            axis.plot(
                selected["year"],
                selected["conservative_scenario"],
                color=AU_GOLD,
                linewidth=2.3,
                label="Conservative scenario",
            )

        axis.set_title(
            f"{selected_indicator}: Forecast to 2030",
            fontweight="bold",
        )
        axis.set_xlabel("Year")
        axis.set_ylabel("Indicator value")
        axis.legend(frameon=False)

        style_axis(axis)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        st.markdown(
            f"""
            <div class="insight-box">
                Under the baseline scenario,
                <strong>{selected_indicator}</strong> is projected to
                reach approximately
                <strong>{
                    format_number(
                        summary.get("baseline_2030", np.nan)
                    )
                }</strong>
                by 2030.
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_section("Forecast data")

        st.dataframe(
            selected,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download selected forecast",
            data=to_csv_bytes(selected),
            file_name="selected_indicator_forecast.csv",
            mime="text/csv",
        )


# ============================================================
# TARGET ASSESSMENT
# ============================================================
elif page == "Target Assessment":
    display_header(
        "Forecast-to-Target Assessment",
        "Assess whether forecast scenarios are sufficient to meet "
        "available official targets.",
    )

    if target_comparison.empty:
        st.warning(
            "No comparable forecast-target records were generated."
        )
    else:
        target_data = target_comparison.copy()

        numeric_columns = [
            "target_value",
            "baseline_forecast",
            "optimistic_scenario",
            "conservative_scenario",
            "baseline_gap",
        ]

        for column in numeric_columns:
            if column in target_data.columns:
                target_data[column] = pd.to_numeric(
                    target_data[column],
                    errors="coerce",
                )

        status_column = find_first_column(
            target_data,
            ["target_status", "status"],
        )

        targets_met = 0
        targets_possible = 0
        targets_above = 0

        if status_column:
            status_text = target_data[status_column].fillna("")

            targets_met = int(
                status_text.str.contains(
                    "baseline meets",
                    case=False,
                ).sum()
            )
            targets_possible = int(
                status_text.str.contains(
                    "optimistic",
                    case=False,
                ).sum()
            )
            targets_above = int(
                status_text.str.contains(
                    "above modelled",
                    case=False,
                ).sum()
            )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        metric_1.metric("Targets assessed", len(target_data))
        metric_2.metric("Met by baseline", targets_met)
        metric_3.metric(
            "Possible if optimistic",
            targets_possible,
        )
        metric_4.metric(
            "Above all scenarios",
            targets_above,
        )

        display_section("Baseline forecast gaps")

        if {
            "target_indicator",
            "baseline_gap",
        }.issubset(target_data.columns):
            chart = target_data.dropna(
                subset=["baseline_gap"]
            ).sort_values("baseline_gap")

            chart["label"] = chart[
                "target_indicator"
            ].apply(shorten_label)

            colours = [
                AU_GREEN if value >= 0 else ETHIOPIA_RED
                for value in chart["baseline_gap"]
            ]

            figure, axis = plt.subplots(figsize=(11, 5.5))
            axis.barh(
                chart["label"],
                chart["baseline_gap"],
                color=colours,
            )
            axis.axvline(0, color=GREY, linewidth=1.2)
            axis.set_title(
                "Baseline Forecast Minus Target",
                fontweight="bold",
            )
            axis.set_xlabel(
                "Positive values indicate the target is met"
            )

            style_axis(axis)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)

        display_section("Target comparison table")

        st.dataframe(
            target_data,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download target comparison",
            data=to_csv_bytes(target_data),
            file_name="forecast_target_comparison.csv",
            mime="text/csv",
        )


# ============================================================
# DATA EXPLORER
# ============================================================
elif page == "Data Explorer":
    display_header(
        "Project Data Explorer",
        "Inspect and download analytical datasets generated across "
        "Tasks 1–4.",
    )

    dataset_options = {
        "Cleaned records": records,
        "Event-impact links": event_links,
        "Event-impact summary": event_summary,
        "Indicator-impact summary": indicator_impact,
        "Association matrix": association_matrix,
        "Forecast records": forecasts,
        "Forecast model summary": model_summary,
        "Target comparison": target_comparison,
    }

    selected_name = st.selectbox(
        "Select a dataset",
        list(dataset_options.keys()),
    )

    selected_data = dataset_options[selected_name]

    if selected_data.empty:
        st.warning(f"{selected_name} is not available.")
    else:
        metric_1, metric_2 = st.columns(2)
        metric_1.metric("Rows", len(selected_data))
        metric_2.metric("Columns", len(selected_data.columns))

        search_term = st.text_input(
            "Search the selected dataset",
            placeholder="Enter a keyword",
        )

        displayed_data = selected_data.copy()

        if search_term:
            matches = (
                displayed_data.astype(str)
                .apply(
                    lambda column: column.str.contains(
                        search_term,
                        case=False,
                        na=False,
                    )
                )
                .any(axis=1)
            )

            displayed_data = displayed_data[matches]

        st.dataframe(
            displayed_data,
            width="stretch",
            hide_index=True,
            height=560,
        )

        filename = (
            selected_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        st.download_button(
            "Download selected dataset",
            data=to_csv_bytes(displayed_data),
            file_name=f"{filename}.csv",
            mime="text/csv",
        )


# ============================================================
# METHODOLOGY
# ============================================================
elif page == "Methodology":
    display_header(
        "Methodology and Limitations",
        "A transparent explanation of how the evidence, event-impact "
        "scores and forecasts were developed.",
    )

    display_section("Analytical workflow")

    st.markdown(
        """
        1. **Data collection:** Historical observations, policy targets
           and major financial-inclusion events were assembled.

        2. **Data preparation:** Years, values, indicator names, record
           types and classifications were cleaned and standardised.

        3. **Exploratory analysis:** Historical coverage, trends,
           differences and data gaps were assessed.

        4. **Event association:** Events were linked to relevant
           indicators using direction, magnitude, confidence and lag.

        5. **Baseline forecasting:** Transparent linear trends were
           fitted where sufficient historical observations existed.

        6. **Scenario development:** Event evidence informed
           conservative and optimistic scenario adjustments.

        7. **Uncertainty assessment:** Forecast intervals widen with
           the forecasting horizon.

        8. **Target comparison:** Forecast scenarios were compared
           with available targets where indicators were comparable.
        """
    )

    display_section("Why a parsimonious model was used")

    st.markdown(
        """
        Historical observations are short and irregular. Complex
        machine-learning models would create an unsupported appearance
        of precision. Linear models provide a transparent baseline that
        can be interpreted, challenged and updated when new data become
        available.
        """
    )

    display_section("Key limitations")

    st.markdown(
        """
        - Historical observations are limited and irregularly spaced.
        - Some forecasts rely on only two or three observations.
        - Indicator definitions may change between data sources.
        - National averages may conceal gender and regional inequalities.
        - Linear trends cannot capture every structural change.
        - Event-impact scores are analytical judgements, not causal proof.
        - Policy announcements do not guarantee implementation.
        - Macroeconomic and political shocks are not modelled separately.
        - Forecasts should support, not replace, expert judgement.
        """
    )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Responsible interpretation:</strong><br>
            Dashboard values should be treated as directional evidence.
            Forecasts are most useful for comparing possible pathways,
            identifying target gaps and prioritising better data
            collection.
        </div>

        <div class="source-box">
            <strong>Project:</strong> Forecasting Financial Inclusion
            in Ethiopia<br>
            <strong>Analytical horizon:</strong> 2030<br>
            <strong>Coverage:</strong> Historical trends, event-impact
            evidence, scenario forecasts and target assessment
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        Ethiopia Financial Inclusion Forecast Dashboard |
        Week 11 Project
    </div>
    """,
    unsafe_allow_html=True,
)