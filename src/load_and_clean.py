import os
import sqlite3
import pandas as pd

def load_csv_to_db(db_path, raw_dir):
    conn = sqlite3.connect(db_path)
    
    tables = [
        "orders", "order_items", "products", "customers", 
        "sellers", "geolocation", "order_payments", "order_reviews"
    ]
    
    for t in tables:
        csv_name = f"olist_{t}_dataset.csv"
        csv_path = os.path.join(raw_dir, csv_name)
        
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} does not exist!")
            continue
            
        print(f"Loading {csv_name} into SQLite table '{t}'...")
        
        # Read in chunks to prevent memory issues for large tables (like geolocation)
        chunksize = 50000
        first_chunk = True
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            if first_chunk:
                chunk.to_sql(t, conn, if_exists="replace", index=False)
                first_chunk = False
            else:
                chunk.to_sql(t, conn, if_exists="append", index=False)
                
        print(f"Successfully loaded table '{t}'")
        
    conn.close()

def run_sql_script(db_path, script_path):
    print(f"Running SQL script {script_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open(script_path, "r", encoding="utf-8") as f:
        sql_content = f.read()
        
    # Split by semicolon to run individual statements (standard SQLite practice)
    statements = sql_content.split(";")
    for stmt in statements:
        stmt = stmt.strip()
        if stmt:
            cursor.execute(stmt)
            
    conn.commit()
    conn.close()
    print(f"Finished executing {script_path}")

def verify_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # List all tables and views
    cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')")
    objects = cursor.fetchall()
    print("\nDatabase Catalog:")
    for obj_name, obj_type in objects:
        cursor.execute(f"SELECT COUNT(*) FROM {obj_name}")
        cnt = cursor.fetchone()[0]
        print(f"- {obj_type.upper()}: {obj_name} ({cnt} rows)")
        
    conn.close()

def main():
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    db_path = os.path.join(processed_dir, "olist.db")
    raw_dir = os.path.join("data", "raw")
    
    # Step 1: Load raw CSVs to DB
    load_csv_to_db(db_path, raw_dir)
    
    # Step 2: Create Views
    run_sql_script(db_path, os.path.join("sql", "01_clean_orders.sql"))
    run_sql_script(db_path, os.path.join("sql", "02_master_table.sql"))
    
    # Step 3: Verify the tables and views are correctly configured
    verify_db(db_path)

if __name__ == "__main__":
    main()
