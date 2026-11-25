-- Cohort retention by signup month: for each signup cohort, what share of
-- customers are still active N months later (month 0 = signup month).
WITH cohorts AS (
    SELECT
        customer_id,
        strftime('%Y-%m', signup_date) AS cohort_month,
        tenure_months,
        churn
    FROM customers
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM cohorts
    GROUP BY cohort_month
),
retention AS (
    SELECT
        c.cohort_month,
        m.months_since_signup,
        COUNT(DISTINCT m.customer_id) AS active_customers
    FROM monthly_activity m
    JOIN cohorts c ON c.customer_id = m.customer_id
    WHERE m.is_churn_month = 0
    GROUP BY c.cohort_month, m.months_since_signup
)
SELECT
    r.cohort_month,
    r.months_since_signup,
    cs.cohort_size,
    r.active_customers,
    ROUND(100.0 * r.active_customers / cs.cohort_size, 1) AS retention_pct
FROM retention r
JOIN cohort_sizes cs ON cs.cohort_month = r.cohort_month
WHERE r.months_since_signup IN (0, 1, 3, 6, 12)
ORDER BY r.cohort_month, r.months_since_signup;
