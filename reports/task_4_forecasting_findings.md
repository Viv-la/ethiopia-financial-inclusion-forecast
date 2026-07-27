# Task 4 Forecasting and Scenario Analysis

## Analysis overview

Task 4 developed transparent baseline and event-adjusted forecasts for
Ethiopia's financial-inclusion indicators through 2030.

The forecasting process used **10 comparable historical
records** selected from **20 available observations**.
The final modelling dataset covers **4 indicators** between
**2014 and 2025**.

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
   The modelling series cover 2014–2025, but observations
   are irregular and some indicators contain only a small number of measured
   years. Forecasts should therefore be treated as directional estimates.

2. **The strongest historical growth occurs in
   `Fayda Digital ID Enrollment`.**  
   Its fitted historical trend is approximately
   **5500000.00 units per year**.

3. **The weakest fitted trend occurs in
   `Account Ownership Gender Gap`.**  
   Its estimated trend is approximately
   **-0.67 units per year**.
   A negative or weak slope should be interpreted cautiously because sparse
   observations may not represent a stable long-term pattern.

4. **Event-adjusted scenarios show the potential importance of reforms.**  
   The optimistic scenario applies documented event relationships gradually
   after the latest observation. The conservative scenario assumes weaker
   progress and does not treat policy announcements as guaranteed outcomes.

5. **Uncertainty increases with the forecast horizon.**  
   `Fayda Digital ID Enrollment` has the widest modelled
   2030 interval, reflecting historical variability and limited observations.
   The width of its 95% interval is approximately
   **10562199.77 units**.

6. **Targets remain separate from forecasts.**  
   The target comparison contains **3 matched records**: **0** are met or exceeded by the baseline, **0** may be reached under the optimistic scenario, and **3** remain above the modelled scenarios.

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
