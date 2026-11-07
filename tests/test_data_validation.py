from pathlib import Path

from src.prepare_data import load_and_validate


def test_ibm_sample_schema_and_counts() -> None:
    frame, quality = load_and_validate(Path("data/raw/Telco-Customer-Churn.csv"))
    assert len(frame) == 7043
    assert quality["duplicate_customer_ids"] == 0
    assert quality["blank_total_charges"] == 11
    assert set(frame["churn_flag"].unique()) == {0, 1}

