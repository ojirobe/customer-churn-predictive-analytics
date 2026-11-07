# Case study: prioritizing retention outreach responsibly

## Situation

A telecom retention team wants to focus limited outreach on customers with elevated churn risk while avoiding the claim that a model can determine why a person will leave.

## Analytical approach

The workflow validates IBM's synthetic sample, uses SQL for portfolio segmentation, reports statistical effect sizes, and trains a calibrated logistic model. Customer identifiers never enter the model. Training, threshold selection, and final evaluation use separate partitions.

## Evidence

Early-tenure and month-to-month segments show the highest observed churn. Contract has a moderate-to-large association with churn (Cramér's V 0.410), while churned customers have substantially shorter tenure. The held-out model reaches ROC AUC 0.843 and average precision 0.634.

## Decision design

Rather than defaulting to a 0.50 probability cutoff, the project selects a threshold on validation data using explicit retention economics. The selected 0.07 threshold captures 97.1% of churners in the test partition but has 37.7% precision, making the false-positive workload visible.

## Recommendation

Use the output as analyst-reviewed outreach prioritization, test interventions through controlled experiments, and monitor workload, incremental retention, customer complaints, calibration, drift, and subgroup performance. Do not use the score for automated pricing, eligibility, or service restrictions.

## Limitations

The source is synthetic, cross-sectional, and lacks timestamps. Results do not establish causality, real-world profitability, future stability, or fairness. The financial scenario excludes operational capacity, offer cost variation, and customer lifetime uncertainty.

