from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "customers.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"
FEATURES_PATH = ROOT / "reports" / "feature_effects.csv"

CONTACT_COST = 10.0
SAVE_RATE = 0.30
RETAINED_MONTHS = 6
GROSS_MARGIN = 0.60
RANDOM_STATE = 42


def build_model(numeric_columns: list[str], categorical_columns: list[str]) -> CalibratedClassifierCV:
    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_columns),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns),
        ]
    )
    estimator = Pipeline(
        [("preprocess", preprocessing), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]
    )
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=5)


def campaign_value(y_true: pd.Series, probabilities: np.ndarray, monthly_charges: pd.Series, threshold: float) -> float:
    contacted = probabilities >= threshold
    preserved_margin = monthly_charges.to_numpy() * RETAINED_MONTHS * GROSS_MARGIN * SAVE_RATE
    return float(np.where(contacted, y_true.to_numpy() * preserved_margin - CONTACT_COST, 0).sum())


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    feature_columns = [column for column in frame.columns if column not in {"customerID", "Churn", "churn_flag"}]
    X = frame[feature_columns]
    y = frame["churn_flag"]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=RANDOM_STATE
    )
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    model = build_model(numeric_columns, categorical_columns)
    model.fit(X_train, y_train)
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    thresholds = np.linspace(0.05, 0.90, 86)
    values = [campaign_value(y_validation, validation_probabilities, X_validation["MonthlyCharges"], threshold) for threshold in thresholds]
    decision_threshold = float(thresholds[int(np.argmax(values))])
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= decision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()
    metrics = {
        "split_design": "60% train, 20% validation, 20% final test; all stratified with fixed seed",
        "temporal_limitation": "The source has no observation date, so this does not estimate future-period drift.",
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "test_churn_prevalence": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y_test, test_probabilities)),
        "brier_score": float(brier_score_loss(y_test, test_probabilities)),
        "baseline_brier": float(brier_score_loss(y_test, np.repeat(y_train.mean(), len(y_test)))),
        "validation_selected_threshold": decision_threshold,
        "precision": float(precision_score(y_test, test_predictions)),
        "recall": float(recall_score(y_test, test_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_predictions)),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
        "test_campaign_value_assumptions": {
            "contact_cost": CONTACT_COST,
            "save_rate": SAVE_RATE,
            "retained_months": RETAINED_MONTHS,
            "gross_margin": GROSS_MARGIN,
            "illustrative_net_value": campaign_value(y_test, test_probabilities, X_test["MonthlyCharges"], decision_threshold),
        },
        "not_for_automated_consequential_decisions": True,
    }
    predictions = pd.DataFrame({
        "actual_churn": y_test.to_numpy(),
        "predicted_probability": test_probabilities,
        "predicted_class": test_predictions,
        "monthly_charges": X_test["MonthlyCharges"].to_numpy(),
    })
    importance = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=8, random_state=RANDOM_STATE, n_jobs=1
    )
    feature_effects = pd.DataFrame({
        "feature": feature_columns,
        "permutation_importance_mean": importance.importances_mean,
        "permutation_importance_std": importance.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    feature_effects.to_csv(FEATURES_PATH, index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

