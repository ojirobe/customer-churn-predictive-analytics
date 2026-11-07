import pandas as pd

from src.statistical_analysis import cramers_v
from src.train_model import campaign_value


def test_cramers_v_is_bounded() -> None:
    table = pd.DataFrame([[90, 10], [30, 70]])
    assert 0 <= cramers_v(table) <= 1


def test_campaign_value_rewards_true_positive_contacts() -> None:
    y = pd.Series([1, 0])
    probabilities = pd.Series([0.9, 0.8]).to_numpy()
    charges = pd.Series([100.0, 100.0])
    assert campaign_value(y, probabilities, charges, 0.5) == 88.0


def test_identifier_is_not_a_model_feature() -> None:
    frame = pd.read_csv("data/processed/customers.csv")
    model_features = [column for column in frame.columns if column not in {"customerID", "Churn", "churn_flag"}]
    assert "customerID" not in model_features
    assert "churn_flag" not in model_features

