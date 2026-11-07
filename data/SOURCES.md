# Data sources and attribution

## IBM Telco Customer Churn

- Repository: https://github.com/IBM/telco-customer-churn-on-icp4d
- Data file: `data/Telco-Customer-Churn.csv`
- Raw URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Publisher: IBM
- Nature: synthetic/sample telecom customer data
- Retrieved reproducibly by: `python src/acquire_data.py`

The dataset has no real customer names or direct personal contact information. `customerID` is a synthetic row identifier and is excluded from predictive features.

## Important limitations

- This is synthetic demonstration data, so observed relationships do not establish real-world causal effects.
- The table is a cross-sectional snapshot and does not contain a true observation timestamp. Random stratified splits will therefore estimate performance on similar records, not future-period drift.
- Churn is imbalanced, so accuracy alone is not an acceptable evaluation metric.
- `TotalCharges` contains blank values that correspond to very new customers and must be handled explicitly.
- Predictive outputs are intended for analyst review and educational prioritization, not automated denial of service, pricing, or other consequential decisions.

