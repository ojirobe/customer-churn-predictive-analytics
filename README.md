# Customer Churn & Predictive Analytics

**Live dashboard:** [customer-churn-predictive-analytics-ojirobe.streamlit.app](https://customer-churn-predictive-analytics-ojirobe.streamlit.app/)

Project 4 of a five-project analytics portfolio. This repository implements a reproducible customer-retention workflow with SQL segmentation, statistical testing, leakage-controlled predictive modelling, validation-based campaign economics, explainability, an interactive dashboard, and automated tests.

> IBM's sample is synthetic and cross-sectional. Results are educational and must not be treated as evidence about real customers or used for automated pricing, eligibility, service denial, or other consequential decisions.

## Business questions

1. Which customer segments have the highest observed churn?
2. How do contract, tenure, service, and payment characteristics relate to churn?
3. Are the observed differences statistically detectable, and how large are their effects?
4. Can a calibrated model prioritize likely churners on held-out data?
5. How does a retention campaign threshold change when outreach costs and expected saved margin are considered?

## Data

The project downloads IBM's synthetic Telco Customer Churn sample directly from its public repository and records a SHA-256 checksum. It contains 7,043 fictional customer records and 21 source fields covering contracts, services, tenure, billing, payment, and churn.

Validation confirmed:

- 7,043 unique synthetic customer IDs and no duplicates.
- 1,869 churned customers; overall churn rate 26.54%.
- 11 blank `TotalCharges` values, handled inside the model pipeline.
- `customerID` is excluded from all predictive features.

See [data/SOURCES.md](data/SOURCES.md) for attribution and limitations.

## Executed findings

| Segment | Customers | Churn rate |
|---|---:|---:|
| Month-to-month contract | 3,875 | 42.71% |
| One-year contract | 1,473 | 11.27% |
| Two-year contract | 1,695 | 2.83% |
| Tenure 0–6 months | 1,481 | 52.94% |
| Tenure 49+ months | 2,239 | 9.51% |

Customers with fiber internet, no tech support, and no online security show 55.01% churn in this synthetic sample. Electronic-check customers with paperless billing show 49.77% churn. These are descriptive associations, not intervention effects.

## Statistical evidence

- Contract and churn: chi-square `p < 0.001`, Cramér's V `0.410`.
- Tenure, churned versus retained: medians `10` versus `38` months; rank-biserial `-0.480`.
- Monthly charges, churned versus retained: medians `$79.65` versus `$64.43`; rank-biserial `0.242`.

Large sample sizes can make small differences significant, so the dashboard emphasizes effect sizes and limitations rather than p-values alone.

## Predictive model

The model uses median imputation, standardization, one-hot encoding, class-weighted logistic regression, and probability calibration inside a single pipeline. The split is 60% train, 20% validation, and 20% final test, with stratification and a fixed seed.

| Held-out test metric | Result |
|---|---:|
| ROC AUC | 0.843 |
| Average precision / PR AUC | 0.634 |
| Brier score | 0.138 |
| Prevalence baseline Brier | 0.195 |
| Recall at selected threshold | 97.1% |
| Precision at selected threshold | 37.7% |
| Balanced accuracy | 69.6% |

The decision threshold `0.07` was chosen only on validation data under illustrative assumptions: `$10` outreach cost, `30%` save rate, six retained months, and `60%` gross margin. Its held-out illustrative net value is `$18,855.77`. The low threshold intentionally accepts many false positives to capture more potential churners.

Because the source contains no observation date, the split cannot estimate future-period drift. It estimates generalization to similar records only.

## Project structure

```text
app.py                      Streamlit Cloud entry point
dashboards/app.py           Interactive retention dashboard
data/SOURCES.md             Attribution and limitations
src/acquire_data.py         Reproducible IBM download and checksum
src/prepare_data.py         Validation, cleaning, and SQLite creation
src/first_analysis.py       Initial executed retention analysis
src/run_sql_analysis.py     SQL reporting pipeline
src/statistical_analysis.py Effect-size and hypothesis tests
src/train_model.py          Calibrated model, economics, explainability
sql/churn_analysis.sql      Reusable business queries
tests/                      Data, pipeline, economics, and app tests
reports/                    Executed aggregate evidence
```

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python src/acquire_data.py
python src/prepare_data.py
python src/first_analysis.py
python src/run_sql_analysis.py
python src/statistical_analysis.py
python src/train_model.py

pytest -q
streamlit run app.py
```

## Responsible interpretation

- Predictions are probabilities, not certainties.
- Permutation importance measures predictive reliance, not causal influence.
- Campaign economics are scenario assumptions, not audited financial forecasts.
- Subgroup patterns require monitoring and human review before any real use.
- Real deployment would require time-based validation, drift monitoring, consent and privacy review, fairness assessment, and an appealable human decision process.

