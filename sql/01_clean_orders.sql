-- View to clean orders table and standardize timestamps for SQLite
DROP VIEW IF EXISTS clean_orders;

CREATE VIEW clean_orders AS
SELECT
    order_id,
    customer_id,
    order_status,
    datetime(order_purchase_timestamp) AS purchase_ts,
    datetime(order_delivered_customer_date) AS delivered_ts,
    datetime(order_estimated_delivery_date) AS estimated_ts
FROM orders
WHERE order_purchase_timestamp IS NOT NULL;
