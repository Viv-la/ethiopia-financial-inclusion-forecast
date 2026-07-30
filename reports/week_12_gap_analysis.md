# Week 12 Portfolio Upgrade: Gap Analysis

## Selected project

**Forecasting Financial Inclusion in Ethiopia** was selected because it combines
a finance-sector problem, policy relevance, reproducible analysis, transparent
forecasting, and an interactive decision-support dashboard. It is therefore the
strongest project in the portfolio for demonstrating both analytical and
engineering capability.

## Baseline assessment

| Area | Week 11 baseline | Week 12 improvement |
|---|---|---|
| Code structure | Four analysis scripts and one large dashboard file | Reusable dashboard logic moved into a typed utility module |
| Testing | No automated tests | Seven focused unit tests |
| Continuous integration | No workflow | GitHub Actions lint-and-test workflow |
| Dependencies | Runtime list omitted NumPy and Streamlit | Complete runtime and development requirements |
| Documentation | Strong analytical README but no portfolio-upgrade evidence | CI badge, verification commands, engineering section, and this gap analysis |
| Reproducibility | Manual setup and execution | Explicit environment, test, lint, analysis, and dashboard commands |
| Responsible use | Limitations already documented | Retained as a core feature because forecasts are planning scenarios, not guarantees |

## Prioritised improvements

1. **Reliability:** add automated tests for data-cleaning and display helpers.
2. **Maintainability:** separate pure transformation logic from the Streamlit UI.
3. **Quality control:** run linting and tests automatically on every push and pull request.
4. **Reproducibility:** distinguish runtime from development dependencies.
5. **Professional communication:** document the upgrade, validation process, business value, and limitations.

## Acceptance criteria

- At least five meaningful tests pass locally and in GitHub Actions.
- The dashboard imports reusable helpers from `src/dashboard_utils.py`.
- A clean environment can be installed from `requirements.txt` or
  `requirements-dev.txt`.
- The README explains how to reproduce, test, and run the project.
- The original processed datasets, figures, analytical scripts, and dashboard
  behaviour remain available.
