from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "processed" / "churn.db"
QUERY_PATH = ROOT / "sql" / "churn_analysis.sql"
OUTPUT_DIR = ROOT / "reports" / "sql_results"
OUTPUT_NAMES = ["portfolio_summary", "contract_segments", "tenure_cohorts", "service_combinations", "payment_segments"]


def main() -> None:
    statements = [statement.strip() for statement in QUERY_PATH.read_text().split(";") if statement.strip()]
    if len(statements) != len(OUTPUT_NAMES):
        raise ValueError("SQL statement count does not match configured output names")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        for name, statement in zip(OUTPUT_NAMES, statements):
            result = pd.read_sql_query(statement, connection)
            result.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
            print(f"\n{name}\n{result.to_string(index=False)}")


if __name__ == "__main__":
    main()

