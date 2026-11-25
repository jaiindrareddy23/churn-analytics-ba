-- Revenue at risk: monthly recurring revenue currently held by customers
-- who have already churned (lost) and by still-active customers whose
-- profile segment has an above-average churn rate (at-risk), by segment.
WITH segment_rates AS (
    SELECT
        contract_type,
        internet_service,
        COUNT(*) AS segment_customers,
        SUM(churn) AS segment_churned,
        1.0 * SUM(churn) / COUNT(*) AS segment_churn_rate
    FROM customers
    GROUP BY contract_type, internet_service
),
overall AS (
    SELECT 1.0 * SUM(churn) / COUNT(*) AS overall_churn_rate FROM customers
)
SELECT
    s.contract_type,
    s.internet_service,
    s.segment_customers,
    s.segment_churned,
    ROUND(100.0 * s.segment_churn_rate, 1) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN c.churn = 1 THEN c.monthly_charges ELSE 0 END), 2) AS lost_mrr,
    ROUND(
        SUM(CASE WHEN c.churn = 0 THEN c.monthly_charges ELSE 0 END)
        * s.segment_churn_rate,
    2) AS at_risk_mrr_active,
    ROUND(
        SUM(CASE WHEN c.churn = 1 THEN c.monthly_charges ELSE 0 END)
        + SUM(CASE WHEN c.churn = 0 THEN c.monthly_charges ELSE 0 END) * s.segment_churn_rate,
    2) AS total_revenue_at_risk
FROM customers c
JOIN segment_rates s
    ON s.contract_type = c.contract_type AND s.internet_service = c.internet_service
GROUP BY s.contract_type, s.internet_service
ORDER BY total_revenue_at_risk DESC;
