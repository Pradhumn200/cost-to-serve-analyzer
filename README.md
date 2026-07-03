# Cost-to-Serve & RTO Risk Analyzer

A production-grade decision-support system to identify unprofitable shipping lanes, predict order cancellation and delivery delay risks (RTO), and simulate pricing policy impacts.

![Updated Dashboard Interface](file:///C:/Users/pradh/.gemini/antigravity/brain/b3623fec-57e9-457e-9a60-63d15e3b7441/dashboard_mockup_v2_1783062334236.png)

## Core Features & Pipeline

1. **Automated Data Pipeline (`src/download_data.py`, `src/load_and_clean.py`)**
   - Retrieves all 9 CSV files of the Olist Brazilian E-commerce dataset.
   - Restructures and cleans the data inside a local **SQLite** database (`data/processed/olist.db`).
   - Builds SQL views: `clean_orders`, `agg_payments` (resolving payment card duplication by selecting primary payment), and `master_table` (joining geolocation coordinates averaged by zip prefix to resolve multiple coordinates).

2. **Cost-to-Serve Modeling (`src/cost_to_serve.py`)**
   - Computes geodesic seller-customer distance (km) using `geopy` with caching to optimize runtime.
   - Calculates **Cost-to-Serve**: `Freight Value + Cancellation Indicator * (Freight Value * 1.5)`.
   - Identifies the worst-performing product-state corridors (e.g., *eletronicos* to *MA* having a logistics overhead ratio representing 185% of order value).

3. **Predictive Risk Classifier (`src/train_model.py`)**
   - Implements an **XGBoost Classifier** to predict order delay/cancellation risks based only on indicators known at order placement.
   - Restores class balance using **SMOTE** oversampling.
   - Packages the trained model and categorical mappings (`outputs/risk_model_package.pkl`).
   - Saves feature importances to `outputs/feature_importance.png`.

4. **Causal Inference via Difference-in-Differences (`src/causal_inference.py`)**
   - Estimates the causal effect of free shipping promotions.
   - Treatment group: Orders where `freight_value == 0`.
   - Runs a DiD OLS regression: `price ~ treated + post + treated_post`.
   - Generates a parallel trends validation chart (`outputs/parallel_trends.png`).

5. **Monte Carlo Simulation (`src/monte_carlo.py`)**
   - Performs a 10,000-run simulation of the policy change impact (removing shipping subsidies in the worst corridors).
   - Simulates volume using a Poisson distribution and problem rates using a Normal distribution.
   - Saves results and histograms (`outputs/monte_carlo_results.png`, `outputs/monte_carlo_summary.json`).

6. **Interactive Streamlit Dashboard (`dashboard/app.py`)**
   - **Cost Heatmap:** Dynamic mapping of Brazil's states showing average cost-to-serve percentage.
   - **Risk Predictor:** Lets logistics managers test risk profiles *before* ship-out with local perturbation-based factor explanations.
   - **Policy Simulator:** Live slider to evaluate minimum order value thresholds for free shipping subsidies, utilizing dynamically computed freight savings.
   - **Causal Impact (DiD):** Displays Parallel Trends Validation plot and statsmodels regression summary directly.

---

## Key Analytical Insights

### 1. Loss-Making Corridors (Phase 3)
* The worst lanes are dominated by shipping electronics (`eletronicos`) and telephony (`telefonia`) to northern/northeastern states like Maranhão (`MA`), Rondônia (`RO`), and Alagoas (`AL`).
* For electronics orders to Maranhão (`MA`), Olist loses money on delivery, as logistics costs represent **185.76%** of the product price, paired with a high **25.00%** delay/cancellation rate.

### 2. Risk Predictive Model (Phase 4)
* **Classifier Performance:** The XGBoost model achieved a **ROC-AUC of 68.24%** on out-of-sample data.
* **Top Features:** Feature importance analysis (Gain) shows that **Distance (km)**, **Price**, and **Freight Value** are the strongest predictors of order failure risk.

### 3. Causal Impact of Free Shipping (Phase 5)
* **DiD Estimator:** The Difference-in-Differences regression shows that the introduction of free shipping (subsidized freight) led to a **decrease** of **BRL 83.49** in average item price, but this was not statistically significant at the 5% level (p-value: 0.138).
* **Business Takeaway:** Subsidizing freight did not drive customers to purchase higher-valued products, and instead correlated with cheaper item purchases (customers capitalizing on free shipping for lower-value goods).

### 4. Monte Carlo Simulator Results (Phase 6)
* **Policy Margin Impact:** Eliminating free shipping entirely in the worst 10% corridors yields a **median monthly margin impact of BRL -11.48** (break-even).
* **Percentile Range:** 5th to 95th percentile ranges from **BRL -229.42** (loss) to **BRL +209.95** (gain).
* **Conclusion:** Restricting free shipping is highly sensitive to customer drop-offs. If a 5% drop-off in conversion occurs, it wipes out any logistics savings. Operational optimization (regional warehouses/fulfillment centers) is preferred over a blanket removal of subsidies.

---

## Running the Project

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download & Process Data:**
   ```bash
   python src/download_data.py
   python src/load_and_clean.py
   python src/cost_to_serve.py
   ```
3. **Run Models & Simulations:**
   ```bash
   python src/train_model.py
   python src/causal_inference.py
   python src/monte_carlo.py
   ```
4. **Launch Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```
