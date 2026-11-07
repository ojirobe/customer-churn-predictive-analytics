from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "customers.csv"
REPORT_PATH = ROOT / "reports" / "first_analysis.md"
SUMMARY_PATH = ROOT / "reports" / "first_analysis.json"


def churn_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame.groupby(column, dropna=False)
        .agg(customers=("customerID", "size"), churned=("churn_flag", "sum"), churn_rate=("churn_flag", "mean"))
        .assign(churn_rate_pct=lambda x: (x["churn_rate"] * 100).round(2))
        .drop(columns="churn_rate")
        .sort_values("churn_rate_pct", ascending=False)
        .reset_index()
    )


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    overall_rate = float(frame["churn_flag"].mean() * 100)
    monthly_revenue_at_risk = float(frame.loc[frame["churn_flag"] == 1, "MonthlyCharges"].sum())
    contract = churn_table(frame, "Contract")
    internet = churn_table(frame, "InternetService")
    payment = churn_table(frame, "PaymentMethod")
    summary = {
        "customers": int(len(frame)),
        "churned_customers": int(frame["churn_flag"].sum()),
        "overall_churn_rate_pct": round(overall_rate, 2),
        "monthly_charges_associated_with_churned_customers": round(monthly_revenue_at_risk, 2),
        "median_tenure_churned_months": float(frame.loc[frame["churn_flag"] == 1, "tenure"].median()),
        "median_tenure_retained_months": float(frame.loc[frame["churn_flag"] == 0, "tenure"].median()),
    }
    sections = [
        "# First executed churn analysis",
        "",
        f"- Customers: **{summary['customers']:,}**",
        f"- Churned customers: **{summary['churned_customers']:,}**",
        f"- Overall churn rate: **{summary['overall_churn_rate_pct']:.2f}%**",
        f"- Monthly charges attached to churned customers: **${summary['monthly_charges_associated_with_churned_customers']:,.2f}**",
        f"- Median tenure, churned vs retained: **{summary['median_tenure_churned_months']:.0f} vs {summary['median_tenure_retained_months']:.0f} months**",
        "",
        "## Churn by contract",
        "",
        contract.to_markdown(index=False),
        "",
        "## Churn by internet service",
        "",
        internet.to_markdown(index=False),
        "",
        "## Churn by payment method",
        "",
        payment.to_markdown(index=False),
        "",
        "> Descriptive associations only. The synthetic cross-sectional data cannot establish causal drivers.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(sections) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nContract churn rates:\n", contract.to_string(index=False))


if __name__ == "__main__":
    main()

