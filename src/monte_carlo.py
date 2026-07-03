import os
import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    db_path = os.path.join("data", "processed", "olist.db")
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load processed data
    conn = sqlite3.connect(db_path)
    print("Reading processed_orders table from database...")
    df = pd.read_sql("SELECT * FROM processed_orders", conn)
    conn.close()
    
    # Load corridor summary from Phase 3
    corridor_path = os.path.join(outputs_dir, "corridor_summary.csv")
    if not os.path.exists(corridor_path):
        print(f"Error: {corridor_path} does not exist. Run Phase 3 first.")
        return
        
    corridor_df = pd.read_csv(corridor_path)
    
    # Identify worst corridors: top 10% by avg_cost_pct
    cutoff = corridor_df['avg_cost_pct'].quantile(0.90)
    worst_corridors = corridor_df[corridor_df['avg_cost_pct'] >= cutoff]
    print(f"Worst corridors cutoff (90th percentile): {cutoff:.2%} cost of order value")
    print(f"Number of corridors in bottom 10%: {len(worst_corridors)}")
    
    # Get the transactions corresponding to these worst corridors
    worst_keys = worst_corridors[['product_category_name', 'customer_state']]
    
    # Perform semi-join in pandas to filter original df
    df_worst = df.merge(worst_keys, on=['product_category_name', 'customer_state'], how='inner')
    
    # Calculate baseline parameters for simulation
    total_orders = len(df_worst)
    
    # Time span in months
    df['purchase_ts'] = pd.to_datetime(df['purchase_ts'])
    days_span = (df['purchase_ts'].max() - df['purchase_ts'].min()).days
    months_span = max(1, days_span / 30.4)
    
    avg_monthly_orders = total_orders / months_span
    current_problem_rate = df_worst['is_problem_order'].mean()
    avg_freight = df_worst['freight_value'].mean()
    avg_reverse_cost = avg_freight * 1.5
    avg_order_value = df_worst['price'].mean()
    
    print("\nBaseline Parameters for Simulation:")
    print(f"- Span of dataset: {months_span:.1f} months")
    print(f"- Average monthly orders in worst corridors: {avg_monthly_orders:.2f}")
    print(f"- Baseline problem rate: {current_problem_rate:.2%}")
    print(f"- Average reverse logistics cost: BRL {avg_reverse_cost:.2f}")
    print(f"- Average order value (price): BRL {avg_order_value:.2f}")
    
    # Run Monte Carlo Simulation
    n_sims = 10000
    results = []
    
    np.random.seed(42)
    for _ in range(n_sims):
        # 1. Randomly sample variation in monthly order volume (Poisson)
        simulated_orders = np.random.poisson(lam=avg_monthly_orders)
        
        # 2. Randomly sample variation in problem rate (Normal, standard deviation of 3%)
        simulated_problem_rate = np.random.normal(loc=current_problem_rate, scale=0.03)
        simulated_problem_rate = np.clip(simulated_problem_rate, 0, 1)
        
        # Policy: Restrict free shipping in these corridors
        # Assumptions:
        # - We save all reverse logistics costs for the problem orders we prevent
        # - Let's assume the policy change cuts problem rate by 50% (better packaging / tracking / delivery validation)
        #   or if we eliminate free shipping, we might drop orders by 5% but reduce problem rates
        #   Let's simulate the direct guide logic:
        #   We save reverse shipping costs of the problem orders, but lose 5% of order value (margins)
        #   Wait, average profit margin on e-commerce is around 10%, but let's assume we lose 5% of revenue:
        #   revenue_lost = simulated_orders * 0.05 * avg_order_value
        #   cost_saved = simulated_orders * simulated_problem_rate * avg_reverse_cost
        
        cost_saved = simulated_orders * simulated_problem_rate * avg_reverse_cost
        revenue_lost = simulated_orders * 0.05 * avg_order_value
        
        net_impact = cost_saved - revenue_lost
        results.append(net_impact)
        
    results = np.array(results)
    
    median_val = np.median(results)
    p5 = np.percentile(results, 5)
    p95 = np.percentile(results, 95)
    
    print("\nSimulation Results:")
    print(f"Median impact: BRL {median_val:,.2f}")
    print(f"5th percentile: BRL {p5:,.2f}")
    print(f"95th percentile: BRL {p95:,.2f}")
    
    # Save results to JSON
    summary = {
        "dataset_months": months_span,
        "avg_monthly_orders": avg_monthly_orders,
        "current_problem_rate": current_problem_rate,
        "avg_reverse_cost": avg_reverse_cost,
        "avg_order_value": avg_order_value,
        "median_impact": median_val,
        "percentile_5": p5,
        "percentile_95": p95,
        "n_sims": n_sims
    }
    
    summary_path = os.path.join(outputs_dir, "monte_carlo_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved simulation summary stats to {summary_path}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Median: BRL {median_val:,.0f}')
    plt.axvline(p5, color='orange', linestyle=':', linewidth=2, label=f'5th Percentile: BRL {p5:,.0f}')
    plt.axvline(p95, color='green', linestyle=':', linewidth=2, label=f'95th Percentile: BRL {p95:,.0f}')
    
    plt.title('Simulated Monthly Margin Impact of Policy Change (Worst Corridors)', fontsize=14)
    plt.xlabel('Net Monthly Margin Impact (BRL)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = os.path.join(outputs_dir, "monte_carlo_results.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved simulation distribution histogram to {plot_path}")

if __name__ == "__main__":
    main()
