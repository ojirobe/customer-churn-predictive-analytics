from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]


def wide_plotly(figure) -> None:
    if Version(st.__version__) >= Version("1.51.0"):
        st.plotly_chart(figure, width="stretch")
    else:
        st.plotly_chart(figure, use_container_width=True)


def wide_dataframe(frame: pd.DataFrame) -> None:
    if Version(st.__version__) >= Version("1.51.0"):
        st.dataframe(frame, hide_index=True, width="stretch")
    else:
        st.dataframe(frame, hide_index=True, use_container_width=True)


def ensure_outputs() -> None:
    required = [
        ROOT / "data" / "processed" / "customers.csv",
        ROOT / "reports" / "statistical_tests.json",
        ROOT / "reports" / "model_metrics.json",
        ROOT / "reports" / "feature_effects.csv",
    ]
    if all(path.exists() for path in required):
        return
    for script in ["acquire_data.py", "prepare_data.py", "first_analysis.py", "run_sql_analysis.py", "statistical_analysis.py", "train_model.py"]:
        subprocess.run([sys.executable, str(ROOT / "src" / script)], check=True, cwd=ROOT)


@st.cache_data
def load_outputs() -> dict:
    ensure_outputs()
    return {
        "customers": pd.read_csv(ROOT / "data" / "processed" / "customers.csv"),
        "contracts": pd.read_csv(ROOT / "reports" / "sql_results" / "contract_segments.csv"),
        "tenure": pd.read_csv(ROOT / "reports" / "sql_results" / "tenure_cohorts.csv"),
        "services": pd.read_csv(ROOT / "reports" / "sql_results" / "service_combinations.csv"),
        "payments": pd.read_csv(ROOT / "reports" / "sql_results" / "payment_segments.csv"),
        "statistics": json.loads((ROOT / "reports" / "statistical_tests.json").read_text()),
        "metrics": json.loads((ROOT / "reports" / "model_metrics.json").read_text()),
        "effects": pd.read_csv(ROOT / "reports" / "feature_effects.csv"),
        "predictions": pd.read_csv(ROOT / "reports" / "test_predictions.csv"),
    }


st.set_page_config(page_title="Customer Churn Analytics", page_icon="📉", layout="wide")
data = load_outputs()
customers = data["customers"]

st.sidebar.header("Analysis controls")
contract_options = sorted(customers["Contract"].unique())
selected_contracts = st.sidebar.multiselect("Contract type", contract_options, default=contract_options)
internet_options = sorted(customers["InternetService"].unique())
selected_internet = st.sidebar.multiselect("Internet service", internet_options, default=internet_options)
filtered = customers[customers["Contract"].isin(selected_contracts) & customers["InternetService"].isin(selected_internet)]
st.sidebar.caption("IBM synthetic sample. Filters describe the selected portfolio; they do not change model evaluation results.")

st.title("Customer Churn & Predictive Analytics")
st.caption("Retention analysis using IBM synthetic telecom data. Educational decision support—not automated customer treatment.")

if filtered.empty:
    st.warning("Choose at least one contract and internet-service group.")
    st.stop()

cols = st.columns(5)
cols[0].metric("Customers", f"{len(filtered):,}")
cols[1].metric("Churn rate", f"{filtered['churn_flag'].mean():.1%}")
cols[2].metric("Churned", f"{filtered['churn_flag'].sum():,}")
cols[3].metric("Monthly charges at risk", f"${filtered.loc[filtered['churn_flag'] == 1, 'MonthlyCharges'].sum():,.0f}")
cols[4].metric("Median tenure", f"{filtered['tenure'].median():.0f} months")

overview, segments, evidence, model_tab, methodology = st.tabs(
    ["Portfolio overview", "Retention segments", "Statistical evidence", "Model & economics", "Data & methodology"]
)

with overview:
    contract = filtered.groupby("Contract", as_index=False).agg(customers=("customerID", "size"), churn_rate=("churn_flag", "mean"))
    contract["churn_rate_pct"] = 100 * contract["churn_rate"]
    wide_plotly(px.bar(contract, x="Contract", y="churn_rate_pct", text_auto=".1f", title="Churn rate by contract", labels={"churn_rate_pct": "Churn rate (%)"}))
    tenure_plot = filtered.assign(tenure_band=pd.cut(filtered["tenure"], [-1, 6, 12, 24, 48, 100], labels=["00–06", "07–12", "13–24", "25–48", "49+"]))
    tenure_summary = tenure_plot.groupby("tenure_band", observed=False, as_index=False)["churn_flag"].mean()
    tenure_summary["churn_rate_pct"] = 100 * tenure_summary["churn_flag"]
    wide_plotly(px.line(tenure_summary, x="tenure_band", y="churn_rate_pct", markers=True, title="Churn declines with tenure", labels={"tenure_band": "Tenure (months)", "churn_rate_pct": "Churn rate (%)"}))

with segments:
    st.subheader("Highest-risk service combinations")
    wide_dataframe(data["services"])
    wide_plotly(px.bar(data["payments"], x="payment_method", y="churn_rate_pct", color="paperless_billing", barmode="group", title="Payment and paperless-billing segments"))
    st.info("Segment differences help generate retention hypotheses; they do not prove that changing a service or payment method will prevent churn.")

with evidence:
    stats = data["statistics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Contract Cramér’s V", f"{stats['contract_vs_churn_chi_square']['cramers_v']:.3f}")
    c2.metric("Tenure rank-biserial", f"{stats['tenure_churned_vs_retained_mann_whitney']['rank_biserial']:.3f}")
    c3.metric("Charges rank-biserial", f"{stats['monthly_charges_churned_vs_retained_mann_whitney']['rank_biserial']:.3f}")
    st.write("All three tests have very small p-values, but effect sizes and business context matter more than significance alone. The synthetic sample cannot support causal claims.")

with model_tab:
    metrics = data["metrics"]
    cards = st.columns(5)
    cards[0].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    cards[1].metric("PR AUC", f"{metrics['pr_auc_average_precision']:.3f}")
    cards[2].metric("Recall", f"{metrics['recall']:.1%}")
    cards[3].metric("Precision", f"{metrics['precision']:.1%}")
    cards[4].metric("Illustrative value", f"${metrics['test_campaign_value_assumptions']['illustrative_net_value']:,.0f}")
    top_effects = data["effects"].head(10).sort_values("permutation_importance_mean")
    wide_plotly(px.bar(top_effects, x="permutation_importance_mean", y="feature", orientation="h", error_x="permutation_importance_std", title="Held-out permutation importance", labels={"permutation_importance_mean": "Mean decrease in ROC AUC"}))
    st.write(f"The decision threshold ({metrics['validation_selected_threshold']:.2f}) was chosen on validation data using an illustrative $10 contact cost, 30% save rate, six retained months, and 60% gross margin. It was not tuned on the test set.")
    st.warning("This model is not approved for automated pricing, eligibility, service denial, or other consequential customer decisions.")

with methodology:
    st.markdown("""
    ### Reproducibility and limitations
    - The raw file is downloaded from IBM and recorded with a SHA-256 checksum.
    - Customer IDs are used only for validation and counts, never as model features.
    - Numeric missing values are imputed and categorical values encoded inside the model pipeline.
    - The 60/20/20 train/validation/test split is stratified and fixed. The source has no timestamp, so performance does not estimate future-period drift.
    - The threshold is selected on validation economics; the final test set is reserved for reporting.
    - Permutation importance describes predictive reliance, not causal influence.
    - The dataset is synthetic and results should be treated as a portfolio demonstration.
    """)
    wide_dataframe(pd.DataFrame([json.loads((ROOT / "data" / "processed" / "data_quality.json").read_text())]))

