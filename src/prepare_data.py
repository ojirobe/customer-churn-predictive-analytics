from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "customers.csv"
QUALITY_PATH = PROCESSED_DIR / "data_quality.json"

EXPECTED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
}


def load_and_validate(path: Path = RAW_PATH) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, dtype={"customerID": "string"})
    missing_columns = sorted(EXPECTED_COLUMNS - set(frame.columns))
    unexpected_columns = sorted(set(frame.columns) - EXPECTED_COLUMNS)
    if missing_columns or unexpected_columns:
        raise ValueError(f"Schema mismatch: missing={missing_columns}, unexpected={unexpected_columns}")
    if frame["customerID"].duplicated().any():
        raise ValueError("customerID must be unique")
    if not set(frame["Churn"].dropna().unique()) <= {"Yes", "No"}:
        raise ValueError("Churn must contain only Yes/No values")
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame["churn_flag"] = frame["Churn"].map({"No": 0, "Yes": 1}).astype("int8")
    blank_total_charges = int(frame["TotalCharges"].isna().sum())
    if (frame["tenure"] < 0).any() or (frame["MonthlyCharges"] < 0).any():
        raise ValueError("Tenure and charges must be non-negative")
    quality = {
        "rows": int(len(frame)),
        "columns_raw": len(EXPECTED_COLUMNS),
        "unique_customer_ids": int(frame["customerID"].nunique()),
        "duplicate_customer_ids": int(frame["customerID"].duplicated().sum()),
        "blank_total_charges": blank_total_charges,
        "churned_customers": int(frame["churn_flag"].sum()),
        "retained_customers": int((frame["churn_flag"] == 0).sum()),
        "churn_rate_pct": round(float(frame["churn_flag"].mean() * 100), 2),
    }
    return frame, quality


def main() -> None:
    frame, quality = load_and_validate()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED_PATH, index=False)
    QUALITY_PATH.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()

