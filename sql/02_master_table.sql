-- View to aggregate payments by selecting the primary payment method per order
DROP VIEW IF EXISTS agg_payments;

CREATE VIEW agg_payments AS
SELECT 
    order_id,
    payment_type,
    payment_installments,
    payment_value
FROM (
    SELECT 
        order_id,
        payment_type,
        payment_installments,
        payment_value,
        ROW_NUMBER() OVER(PARTITION BY order_id ORDER BY payment_value DESC) as rn
    FROM order_payments
)
WHERE rn = 1;

-- View to aggregate geolocation coordinates by averaging lat/lng per zip prefix
DROP VIEW IF EXISTS avg_geolocation;

CREATE VIEW avg_geolocation AS
SELECT
    geolocation_zip_code_prefix AS zip_prefix,
    AVG(geolocation_lat) AS lat,
    AVG(geolocation_lng) AS lng
FROM geolocation
GROUP BY geolocation_zip_code_prefix;

-- Master table view joining all entities
DROP VIEW IF EXISTS master_table;

CREATE VIEW master_table AS
SELECT
    oi.order_id,
    oi.order_item_id,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value,
    co.customer_id,
    co.order_status,
    co.purchase_ts,
    co.delivered_ts,
    co.estimated_ts,
    p.product_category_name,
    c.customer_zip_code_prefix,
    c.customer_city,
    c.customer_state,
    s.seller_zip_code_prefix,
    s.seller_city,
    s.seller_state,
    ap.payment_type,
    ap.payment_installments,
    cg.lat AS customer_lat,
    cg.lng AS customer_lng,
    sg.lat AS seller_lat,
    sg.lng AS seller_lng
FROM order_items oi
INNER JOIN clean_orders co ON oi.order_id = co.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
LEFT JOIN customers c ON co.customer_id = c.customer_id
LEFT JOIN sellers s ON oi.seller_id = s.seller_id
LEFT JOIN agg_payments ap ON oi.order_id = ap.order_id
LEFT JOIN avg_geolocation cg ON c.customer_zip_code_prefix = cg.zip_prefix
LEFT JOIN avg_geolocation sg ON s.seller_zip_code_prefix = sg.zip_prefix;
