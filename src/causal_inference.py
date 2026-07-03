import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def main():
    db_path = os.path.join("data", "processed", "olist.db")
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load processed data
    conn = sqlite3.connect(db_path)
    print("Reading processed_orders table from database...")
    df = pd.read_sql("SELECT * FROM processed_orders", conn)
    conn.close()
    
    # Convert purchase_ts to datetime
    df['purchase_ts'] = pd.to_datetime(df['purchase_ts'])
    
    # 1. Define treatment and control groups
    # Treatment: orders where freight was subsidized (freight_value == 0)
    # Control: orders where customer paid some freight
    df['treated'] = (df['freight_value'] == 0).astype(int)
    
    # 2. Define policy change date
    # Midpoint of the dataset range (2018-01-01)
    policy_change_date = pd.to_datetime('2018-01-01')
    
    # 3. Create pre/post indicators
    df['post'] = (df['purchase_ts'] >= policy_change_date).astype(int)
    df['treated_post'] = df['treated'] * df['post']
    
    print(f"Treatment group size: {df['treated'].sum()} orders")
    print(f"Control group size: {len(df) - df['treated'].sum()} orders")
    
    # 4. Run the DiD regression
    print("Running Difference-in-Differences regression model...")
    # Dependent variable: price (order value)
    model = smf.ols('price ~ treated + post + treated_post', data=df).fit()
    
    # Save regression summary to outputs
    summary_path = os.path.join(outputs_dir, "did_regression_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
    print(f"Saved regression summary to {summary_path}")
    print(model.summary())
    
    # Extract coefficient and p-value for reporting
    coef = model.params['treated_post']
    p_val = model.pvalues['treated_post']
    print(f"\nDiD Estimator (Impact of Free Shipping on Price): {coef:.2f} (p-value: {p_val:.4f})")
    
    # 5. Parallel trends check plot
    print("Generating parallel trends check plot...")
    df['purchase_month'] = df['purchase_ts'].dt.to_period('M')
    
    # Monthly average price for treated and control
    monthly_trends = df.groupby(['purchase_month', 'treated'])['price'].mean().unstack()
    
    plt.figure(figsize=(12, 6))
    
    # Plot trends
    plt.plot(monthly_trends.index.astype(str), monthly_trends[0], marker='o', color='blue', label='Paid Shipping (Control)')
    plt.plot(monthly_trends.index.astype(str), monthly_trends[1], marker='s', color='orange', label='Free Shipping (Treated)')
    
    # Add vertical line for policy change
    plt.axvline(x='2018-01', color='red', linestyle='--', linewidth=2, label='Policy Change (2018-01-01)')
    
    plt.title('Difference-in-Differences Parallel Trends Check (Monthly average price)', fontsize=14)
    plt.xlabel('Purchase Month', fontsize=12)
    plt.ylabel('Average Item Price (BRL)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = os.path.join(outputs_dir, "parallel_trends.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Parallel trends chart saved to {plot_path}")

if __name__ == "__main__":
    main()
