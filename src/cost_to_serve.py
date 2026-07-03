import os
import sqlite3
import pandas as pd
from geopy.distance import geodesic
import numpy as np

def calculate_geodesic_distances(df):
    print("Calculating geodesic distances (seller to customer) with cache optimization...")
    
    # Create a unique list of coordinate pairs to speed up calculation
    unique_coords = df[['seller_lat', 'seller_lng', 'customer_lat', 'customer_lng']].drop_duplicates().dropna()
    print(f"Total rows: {len(df)}, Unique coordinate pairs to calculate: {len(unique_coords)}")
    
    cache = {}
    total = len(unique_coords)
    count = 0
    
    for idx, row in unique_coords.iterrows():
        s_lat, s_lng = row['seller_lat'], row['seller_lng']
        c_lat, c_lng = row['customer_lat'], row['customer_lng']
        
        key = (s_lat, s_lng, c_lat, c_lng)
        try:
            dist = geodesic((s_lat, s_lng), (c_lat, c_lng)).km
            cache[key] = dist
        except Exception:
            cache[key] = np.nan
            
        count += 1
        if count % 5000 == 0:
            print(f"Calculated {count}/{total} coordinate pairs...")
            
    # Map back to main dataframe
    def get_dist(row):
        key = (row['seller_lat'], row['seller_lng'], row['customer_lat'], row['customer_lng'])
        return cache.get(key, np.nan)
        
    df['distance_km'] = df.apply(get_dist, axis=1)
    print("Distance calculation completed!")
    return df

def main():
    db_path = os.path.join("data", "processed", "olist.db")
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load master table from SQLite
    conn = sqlite3.connect(db_path)
    print("Reading master_table view from database...")
    df = pd.read_sql("SELECT * FROM master_table", conn)
    
    # Convert timestamps to datetime objects
    df['purchase_ts'] = pd.to_datetime(df['purchase_ts'])
    df['delivered_ts'] = pd.to_datetime(df['delivered_ts'])
    df['estimated_ts'] = pd.to_datetime(df['estimated_ts'])
    
    # Calculate distance
    df = calculate_geodesic_distances(df)
    
    # Drop rows without distance/coordinates for accuracy in subsequent modeling
    initial_len = len(df)
    df = df.dropna(subset=['distance_km'])
    print(f"Dropped {initial_len - len(df)} rows with missing geolocations. Remaining rows: {len(df)}")
    
    # Calculate delays and cancellations
    print("Flagging delivery delays and cancellations...")
    df['delivery_delay_days'] = (df['delivered_ts'] - df['estimated_ts']).dt.days
    
    # Delayed if delivery took longer than estimated (or if it hasn't delivered yet but is past estimate)
    # If delivered_ts is null, we can check if estimated_ts is past the max dataset date (simulated current time)
    # Let's keep it simple: if delivered_ts is present, is_delayed is based on delivery_delay_days > 0.
    df['is_delayed'] = (df['delivery_delay_days'] > 0).astype(int)
    df['is_cancelled'] = (df['order_status'] == 'canceled').astype(int)
    
    # Problem order is either delayed or cancelled
    df['is_problem_order'] = ((df['is_delayed'] == 1) | (df['is_cancelled'] == 1)).astype(int)
    
    # Cost-to-serve formula
    print("Calculating cost-to-serve metrics...")
    df['reverse_logistics_cost'] = df['is_cancelled'] * (df['freight_value'] * 1.5)
    df['cost_to_serve'] = df['freight_value'] + df['reverse_logistics_cost']
    df['cost_pct_of_order'] = df['cost_to_serve'] / df['price']
    
    # Save the processed data back to SQLite for quick retrieval
    print("Saving processed data back to database table 'processed_orders'...")
    df.to_sql("processed_orders", conn, if_exists="replace", index=False)
    conn.close()
    
    # Aggregate to find loss-making corridors
    print("Aggregating loss-making corridors...")
    corridor_summary = df.groupby(['product_category_name', 'customer_state']).agg(
        total_orders=('order_id', 'count'),
        avg_cost_to_serve=('cost_to_serve', 'mean'),
        avg_cost_pct=('cost_pct_of_order', 'mean'),
        problem_rate=('is_problem_order', 'mean')
    ).reset_index()
    
    # Filter for corridors with a minimum number of orders to avoid noise
    corridor_summary = corridor_summary[corridor_summary['total_orders'] >= 5]
    corridor_summary = corridor_summary.sort_values('avg_cost_pct', ascending=False)
    
    # Export corridor summary
    summary_path = os.path.join(outputs_dir, "corridor_summary.csv")
    corridor_summary.to_csv(summary_path, index=False)
    print(f"Saved corridor summary to {summary_path}")
    
    # Display top 10 worst performing corridors
    print("\nTop 10 worst-performing corridors by average cost % of order value:")
    print(corridor_summary.head(10))

if __name__ == "__main__":
    main()
