from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "customers.csv"
OUTPUT_PATH = ROOT / "reports" / "statistical_tests.json"


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = chi2_contingency(table, correction=False).statistic
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * min(table.shape[0] - 1, table.shape[1] - 1))))


def rank_biserial(u_statistic: float, n1: int, n2: int) -> float:
    return float(2 * u_statistic / (n1 * n2) - 1)


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    contract_table = pd.crosstab(frame["Contract"], frame["Churn"])
    contract_test = chi2_contingency(contract_table)
    churned = frame.loc[frame["churn_flag"] == 1]
    retained = frame.loc[frame["churn_flag"] == 0]
    tenure_test = mannwhitneyu(churned["tenure"], retained["tenure"], alternative="two-sided")
    charge_test = mannwhitneyu(churned["MonthlyCharges"], retained["MonthlyCharges"], alternative="two-sided")
    results = {
        "contract_vs_churn_chi_square": {
            "statistic": float(contract_test.statistic),
            "p_value": float(contract_test.pvalue),
            "degrees_of_freedom": int(contract_test.dof),
            "cramers_v": cramers_v(contract_table),
        },
        "tenure_churned_vs_retained_mann_whitney": {
            "statistic": float(tenure_test.statistic),
            "p_value": float(tenure_test.pvalue),
            "rank_biserial": rank_biserial(tenure_test.statistic, len(churned), len(retained)),
            "churned_median": float(churned["tenure"].median()),
            "retained_median": float(retained["tenure"].median()),
        },
        "monthly_charges_churned_vs_retained_mann_whitney": {
            "statistic": float(charge_test.statistic),
            "p_value": float(charge_test.pvalue),
            "rank_biserial": rank_biserial(charge_test.statistic, len(churned), len(retained)),
            "churned_median": float(churned["MonthlyCharges"].median()),
            "retained_median": float(retained["MonthlyCharges"].median()),
        },
        "caution": "Associations in synthetic cross-sectional data are not causal effects.",
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

