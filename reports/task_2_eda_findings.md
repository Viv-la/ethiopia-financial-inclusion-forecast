# Task 2 Exploratory Data Analysis Findings

## Dataset overview

The unified dataset contains **43 primary records**, comprising
**30 historical observations**, **10 events** and
**3 targets**.

## Evidence-based findings

1. **Access has stronger coverage than several other dimensions.**  
   The Access pillar contains the largest number of pillar-specific records.
   This makes account ownership more suitable for quantitative trend analysis
   than indicators with only one or two historical measurements.

2. **The principal forecasting indicators have limited time-series depth.**  
   The automated classification identified approximately **14
   account-ownership observations** and **14 digital-payment usage
   observations**. These are useful for directional analysis but insufficient
   for highly parameterised forecasting models.

3. **Historical observations are concentrated in recent years.**  
   Historical observations cover approximately 2014 to 2025, but the distribution is irregular. This concentration means recent developments may dominate
   the analysis, while long-term structural change is less precisely measured.

4. **The dataset combines access, usage, gender and affordability evidence.**  
   This supports a multidimensional interpretation of financial inclusion.
   However, coverage is uneven, with affordability substantially less
   represented than Access and Usage.

5. **Gender evidence exists but is not consistently available.**  
   Approximately **30 observation records** contain a populated
   gender field. Gender comparisons should therefore be presented as partial
   evidence rather than a complete national time series.

6. **Location disaggregation is limited.**  
   Approximately **30 observation records** contain location
   information, and much of the dataset is national. Regional and urban-rural
   differences cannot be estimated consistently from the current data alone.

7. **Events offer plausible context but do not establish causality.**  
   The dataset contains **10 policy, infrastructure or market events.
   These can be compared with indicator movements, but observed timing alone
   cannot prove that an event caused a change.

8. **Targets must remain separate from historical outcomes.**  
   The **3 target records** represent intended future performance,
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
