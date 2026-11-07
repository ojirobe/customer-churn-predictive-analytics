-- Portfolio-level retention metrics.
SELECT
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_charges_at_risk
FROM customers;

-- Contract segments.
SELECT
    Contract AS contract_type,
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS average_monthly_charges
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- Tenure cohorts.
SELECT
    CASE
        WHEN tenure <= 6 THEN '00-06 months'
        WHEN tenure <= 12 THEN '07-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_band,
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_band
ORDER BY MIN(tenure);

-- Service combinations with enough observations for stable reporting.
SELECT
    InternetService AS internet_service,
    TechSupport AS tech_support,
    OnlineSecurity AS online_security,
    COUNT(*) AS customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM customers
GROUP BY InternetService, TechSupport, OnlineSecurity
HAVING COUNT(*) >= 100
ORDER BY churn_rate_pct DESC;

-- Payment method and paperless-billing segments.
SELECT
    PaymentMethod AS payment_method,
    PaperlessBilling AS paperless_billing,
    COUNT(*) AS customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_charges_at_risk
FROM customers
GROUP BY PaymentMethod, PaperlessBilling
ORDER BY churn_rate_pct DESC;

