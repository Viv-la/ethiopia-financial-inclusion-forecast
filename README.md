# Forecasting Financial Inclusion in Ethiopia

[![CI](https://github.com/Viv-la/ethiopia-financial-inclusion-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/Viv-la/ethiopia-financial-inclusion-forecast/actions/workflows/ci.yml)

> Week 12 portfolio upgrade: modular utilities, automated tests, continuous
> integration, complete dependency management, and stronger reproducibility.

## Project Overview

This project analyses historical financial-inclusion trends in Ethiopia and develops transparent forecasts to support policy planning through 2030. It integrates historical indicators, major economic and policy events, scenario-based forecasting, target assessment, and an interactive Streamlit dashboard.

The analysis focuses particularly on:

- Access to formal financial services
- Account ownership
- Digital-payment usage
- Mobile-money and digital-finance expansion
- The influence of policy reforms and major economic events
- Progress towards national financial-inclusion targets

The original analysis was completed for the **10 Academy KAIM Week 11
Challenge** and professionally upgraded for the **Week 12 portfolio challenge**.

---

## Business Objective

The objective is to help policymakers and financial-sector stakeholders understand:

1. How financial inclusion in Ethiopia has changed over time.
2. Which major events and reforms may have influenced observed trends.
3. What financial-inclusion outcomes could look like under different scenarios.
4. Whether projected progress is sufficient to meet available policy targets.
5. Which interventions could accelerate inclusive financial-sector growth.

---

## Project Tasks

### Task 1: Data Exploration and Enrichment

Historical observations, targets, policy events, and supporting contextual evidence were collected and standardised.

The main preparation activities included:

- Cleaning indicator names and classifications
- Converting year and value fields into analytical formats
- Separating historical observations from policy targets
- Reviewing missing values and inconsistencies
- Standardising event and source descriptions
- Producing reusable cleaned datasets

### Task 2: Exploratory Data Analysis

Exploratory analysis was conducted to assess:

- Historical indicator coverage
- Trends in account ownership and digital-payment usage
- Differences between indicators
- Data gaps and irregular reporting periods
- Changes associated with the expansion of digital financial services
- Progress towards financial-inclusion targets

### Task 3: Event-Impact Modelling

Major policy, economic, technological, and institutional events were linked to relevant financial-inclusion indicators.

Each relationship was assessed using:

- Impact direction
- Expected magnitude
- Confidence level
- Time lag
- Weighted impact score

These scores represent structured analytical judgements and should not be interpreted as causal estimates.

### Task 4: Forecasting

Transparent baseline models were developed from the available historical observations.

Three scenarios were produced:

- **Baseline scenario:** continuation of the fitted historical trend
- **Optimistic scenario:** stronger progress supported by successful reforms and digital-finance expansion
- **Conservative scenario:** slower progress resulting from implementation constraints or adverse conditions

The forecasting outputs include:

- Historical values
- Baseline forecasts
- Optimistic forecasts
- Conservative forecasts
- 95% uncertainty intervals
- Forecast-to-target comparisons

The required 2025–2027 forecast period is included, while projections are extended to 2030 to support longer-term policy assessment.

### Task 5: Interactive Dashboard

A multipage Streamlit dashboard was developed to communicate the project findings.

The dashboard contains:

- Executive overview
- Historical trends
- Event-impact analysis
- Forecasts to 2030
- Target assessment
- Data explorer
- Methodology and limitations
- Downloadable analytical datasets

---

## Repository Structure

```text
.
├── app/
│   └── dashboard.py
├── .github/workflows/
│   └── ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   ├── week_12_gap_analysis.md
│   └── figures/
├── src/
│   ├── dashboard_utils.py
│   ├── event_impact_analysis.py
│   ├── exploratory_analysis.py
│   ├── forecasting_analysis.py
│   └── prepare_data.py
├── tests/
│   └── test_dashboard_utils.py
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
└── README.md
```

The exact contents of some folders may vary according to the analytical outputs generated during each task.

---

## Installation

### 1. Clone the repository

```powershell
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-PROJECT-FOLDER>
```

If the repository is already open in VS Code, continue to the next step.

### 2. Create a virtual environment

```powershell
py -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 4. Install the required packages

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

If a requirements file is unavailable, install the main packages directly:

```powershell
py -m pip install pandas numpy matplotlib streamlit
```

For development, testing, and quality checks:

```powershell
py -m pip install -r requirements-dev.txt
```

---

## Reproducing and Verifying the Project

Run the analytical pipeline from the repository root:

```powershell
py src\prepare_data.py
py src\exploratory_analysis.py
py src\event_impact_analysis.py
py src\forecasting_analysis.py
```

Run the automated test suite and lint checks:

```powershell
py -m pytest
py -m ruff check src/dashboard_utils.py tests
```

GitHub Actions repeats the lint and test checks automatically for every push
and pull request to `main`.

---

## Week 12 Engineering Upgrade

The portfolio upgrade strengthens the project without changing its analytical
purpose:

- Pure dashboard transformations were extracted into
  `src/dashboard_utils.py`.
- Type hints and focused docstrings make reusable code easier to understand.
- Seven automated tests cover missing values, column normalisation, candidate
  selection, compact numeric formatting, label truncation, and CSV exports.
- GitHub Actions provides repeatable continuous integration.
- Runtime and development dependencies are explicitly separated.
- The original data, analytical outputs, dashboard, responsible-use guidance,
  and policy narrative are preserved.

The detailed before-and-after assessment is available in
[`reports/week_12_gap_analysis.md`](reports/week_12_gap_analysis.md).

---

## Running the Dashboard

From the root of the project, run:

```powershell
py -m streamlit run app\dashboard.py
```

Streamlit should open the dashboard automatically in the browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

To stop the dashboard, return to PowerShell and press:

```text
Ctrl + C
```

---

## Analytical Outputs

The project generates outputs covering:

- Cleaned historical records
- Event-to-indicator relationships
- Event-impact summaries
- Indicator-impact summaries
- Association matrices
- Forecast records
- Forecast model summaries
- Forecast-to-target comparisons
- Dashboard-ready tables and figures

The dashboard automatically searches the repository for the relevant CSV outputs.

---

## Methodology

### Historical Analysis

Historical observations were cleaned, grouped by indicator and year, and reviewed for patterns, coverage, and reporting gaps.

### Event-Impact Assessment

Events were linked to indicators using structured evidence and assessed according to their direction, magnitude, confidence, and expected time lag.

### Forecasting Approach

Parsimonious linear-trend models were selected because the historical series are short and irregular. This approach provides a transparent baseline that can be interpreted and updated as new observations become available.

Scenario adjustments were then applied to demonstrate how stronger or weaker implementation conditions could affect future outcomes.

### Target Assessment

Forecast values were compared with available targets where indicator definitions and forecast years were sufficiently comparable.

---

## Key Limitations

- Historical observations are limited and irregularly spaced.
- Some forecasts rely on very few data points.
- Indicator definitions may differ between sources.
- National averages may conceal gender, income, and geographic inequalities.
- Linear models cannot represent every structural change.
- Event-impact scores are analytical judgements rather than causal proof.
- Policy announcements do not guarantee effective implementation.
- Future macroeconomic, political, and technological shocks are uncertain.
- Forecasts should support—not replace—expert and policy judgement.

---

## Responsible Interpretation

The forecasts are planning scenarios rather than guaranteed outcomes. They are most useful for:

- Comparing possible future pathways
- Identifying target gaps
- Supporting policy discussions
- Prioritising interventions
- Improving future data collection

Forecast results should always be interpreted alongside updated administrative data, household surveys, institutional evidence, and expert knowledge.

---

## Recommendations

The analysis supports the following priorities:

1. Expand affordable and accessible digital-financial services.
2. Strengthen consumer protection and financial literacy.
3. Address gender, rural, income, and connectivity gaps.
4. Improve interoperability between financial-service providers.
5. Strengthen implementation monitoring for national strategies.
6. Establish more regular and consistent indicator reporting.
7. Update the forecasts when new evidence becomes available.
8. Use scenario analysis as an ongoing planning tool rather than a one-time exercise.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Streamlit
- Visual Studio Code
- Git and GitHub

---

## Author

**E. Frida N.**

10 Academy – KAIM  
Week 11 Project  
Forecasting Financial Inclusion in Ethiopia

---

## License

This repository was developed for educational and analytical purposes as part of the 10 Academy KAIM programme.
