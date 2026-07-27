# Data Enrichment Log

## Forecasting Financial Inclusion in Ethiopia

**Project:** Week 11 – Forecasting Financial Inclusion in Ethiopia  
**Task:** Task 1 – Data Exploration and Enrichment  
**Dataset:** `ethiopia_fi_unified_data.xlsx`  
**Processed dataset:** `data/processed/ethiopia_fi_cleaned_records.csv`

---

## 1. Purpose

This log documents the preparation, validation and enrichment of the unified
financial-inclusion dataset used to forecast account ownership and digital-payment
adoption in Ethiopia.

The objective was to create an auditable analytical dataset containing:

- Historical financial-inclusion observations
- Relevant policy, infrastructure and market events
- National financial-inclusion targets
- Event-to-indicator impact relationships
- Source and confidence information
- Standardised fields for later analysis and forecasting

No original observation was overwritten during processing. The raw Excel files
remain unchanged under `data/raw`.

---

## 2. Source Files

The following source files were reviewed and processed:

| Source file | Purpose |
|---|---|
| `ethiopia_fi_unified_data.xlsx` | Main observations, events, targets and event-impact links |
| `reference_codes.xlsx` | Controlled codes and accepted category values |
| `Additional Data Points Guide.xlsx` | Guidance for identifying and documenting additional data |
| `Forecasting_Financial_Inclusion_Methodology.pdf` | Forecasting and scenario-development guidance |
| `Association_Matrix_Methodology.pdf` | Guidance for associating events with financial-inclusion indicators |

---

## 3. Dataset Structure

The primary cleaned dataset contains **43 records** and **25 variables**.

### Record types

| Record type | Number of records | Description |
|---|---:|---|
| Observation | 30 | Historical indicator measurements |
| Event | 10 | Policy, infrastructure, product or market developments |
| Target | 3 | Official or project-relevant financial-inclusion targets |
| **Total** | **43** | Primary analytical records |

A separate impact-link worksheet contains **14 relationships** connecting events
to indicators. It was retained separately to avoid duplicating events or observations
in the primary dataset.

---

## 4. Pillar Coverage

| Pillar | Number of records |
|---|---:|
| Access | 16 |
| Usage | 11 |
| Gender | 5 |
| Affordability | 1 |
| Not applicable to a single pillar | 10 |

The 10 records without a pillar are event records. This is intentional because an
event may affect several indicators or pillars through the separate impact-link table.

The strongest coverage is currently in the Access and Usage pillars, which correspond
to the project’s two principal forecasting outcomes:

1. Account ownership rate
2. Digital-payment adoption rate

Affordability has limited direct coverage and should therefore be interpreted
cautiously.

---

## 5. Time Coverage

The dataset includes calendar-year and fiscal-year observations.

### Calendar-year coverage

| Year | Records |
|---|---:|
| 2014 | 1 |
| 2017 | 1 |
| 2021 | 7 |
| 2022 | 1 |
| 2023 | 1 |
| 2024 | 12 |
| 2025 | 5 |
| 2028 | 1 |
| 2030 | 1 |

### Fiscal-year coverage

| Fiscal year | Records |
|---|---:|
| FY2022/23 | 1 |
| FY2023/24 | 1 |
| FY2024/25 | 9 |
| FY2025/26 | 2 |

Calendar and fiscal-year labels were preserved during Task 1 to maintain fidelity
to the source material. Before time-series modelling, a standard analytical-year
variable will be created while retaining the original `fiscal_year` value.

Future years, including 2028 and 2030, represent targets rather than historical
observations and must not be treated as actual outcomes.

---

## 6. Variables Retained

The cleaned dataset preserves the following analytical fields:

- `record_id`
- `record_type`
- `pillar`
- `indicator`
- `indicator_code`
- `value_numeric`
- `value_type`
- `unit`
- `observation_date`
- `fiscal_year`
- `gender`
- `location`
- Source and citation information
- Confidence and reliability classifications
- Collection and comparison fields
- Notes and contextual information

These variables support descriptive analysis, disaggregation, source assessment,
event-impact modelling and scenario forecasting.

---

## 7. Processing and Validation Actions

The processing script `src/prepare_data.py` performed the following actions:

1. Read every worksheet from the three Excel workbooks.
2. Removed completely empty rows and columns.
3. Converted column names to lowercase `snake_case`.
4. Removed unnecessary leading, trailing and repeated whitespace.
5. Converted common missing-value strings into proper null values.
6. Preserved numeric values without manual alteration.
7. Exported every non-empty worksheet as a separate CSV file.
8. Preserved primary records and impact links as separate tables.
9. Checked for duplicate records and duplicate identifiers.
10. Produced a data-quality summary and column inventory.

---

## 8. Data-Quality Results

The primary dataset contains:

- **43 records**
- **25 columns**
- **No duplicate rows**
- **No duplicate record IDs**
- A mixture of populated and intentionally non-applicable fields

Missing cells are expected because the unified structure contains different record
types. For example, event records may not contain an indicator value, while
observation records may not contain event-specific fields.

Missingness should therefore be evaluated by record type rather than interpreted
automatically as poor data quality.

---

## 9. Enrichment and Impact-Link Treatment

The dataset combines quantitative observations with contextual events and targets.
This enrichment enables later analysis of whether changes in financial inclusion
coincide with:

- Policy reforms
- Financial-sector initiatives
- Digital infrastructure expansion
- Mobile-money or payment-system developments
- Gender-focused inclusion programmes
- National financial-inclusion strategies

The 14 event-impact relationships are stored in:

`data/processed/ethiopia_fi_impact_sheet.csv`

These links will be used in Task 3 to construct an event-indicator association matrix.
They remain analytical hypotheses until compared with historical indicator movements
and supporting evidence.

---

## 10. Source and Confidence Controls

Source information and confidence classifications were retained to distinguish
official observations from estimates, projections and contextual records.

The following principles will guide subsequent analysis:

- Official statistical and institutional sources receive the greatest weight.
- Estimates must remain clearly identified as estimates.
- Targets must not be combined with historical observations.
- Events indicate possible influence, not automatic causation.
- Conflicting values must be investigated before modelling.
- Low-confidence records may support context but should not drive the baseline forecast.

---

## 11. Known Limitations

The dataset has several limitations:

1. Historical observations are sparse and irregularly spaced.
2. Some indicators have only one or two observations.
3. Calendar years and fiscal years are mixed.
4. Gender-disaggregated observations are not consistently available.
5. Location coverage is mainly national, limiting regional comparison.
6. Affordability has substantially fewer records than Access and Usage.
7. Event-impact relationships indicate association rather than proven causality.
8. Different sources may apply different definitions and survey methods.
9. Future targets cannot be used as historical training observations.
10. Short time series will limit the reliability of complex forecasting models.

These limitations will be reflected in model selection, uncertainty intervals and
scenario assumptions.

---

## 12. Files Generated

Task 1 generated the following outputs under `data/processed`:

- `ethiopia_fi_cleaned_records.csv`
- `ethiopia_fi_main_sheet.csv`
- `ethiopia_fi_impact_sheet.csv`
- `reference_codes_reference_codes.csv`
- `additional_data_points_guide_sheet1.csv`
- `additional_data_points_guide_detailed_field_guide.csv`
- `additional_data_points_guide_quick_reference.csv`
- `additional_data_points_guide_blank_enrichment_template.csv`
- `data_quality_summary.csv`
- `column_inventory.csv`

---

## 13. Reproducibility

The complete data-preparation process can be reproduced with:

```powershell
py src\prepare_data.py