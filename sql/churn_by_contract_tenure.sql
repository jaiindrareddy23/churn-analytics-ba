-- Churn rate by contract type and tenure bucket.
SELECT
    contract_type,
    CASE
        WHEN tenure_months < 6 THEN '0-5 mo'
        WHEN tenure_months < 12 THEN '6-11 mo'
        WHEN tenure_months < 24 THEN '12-23 mo'
        ELSE '24+ mo'
    END AS tenure_bucket,
    COUNT(*) AS num_customers,
    SUM(churn) AS num_churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY contract_type, tenure_bucket
ORDER BY contract_type,
    CASE tenure_bucket
        WHEN '0-5 mo' THEN 0
        WHEN '6-11 mo' THEN 1
        WHEN '12-23 mo' THEN 2
        ELSE 3
    END;
