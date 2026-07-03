import os
import pickle
import json
import sqlite3
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config for wide layout and theme
st.set_page_config(
    page_title="Cost-to-Serve & RTO Risk Analyzer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich, modern aesthetics (Glassmorphism & Sleek Accent colors)
st.markdown("""
<style>
    /* Main body background and font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Title Styling */
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B 0%, #852323 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    /* Card design */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA LOADING FUNCTIONS -----------------

@st.cache_data
def load_corridor_data():
    path = os.path.join("outputs", "corridor_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_model_package():
    path = os.path.join("outputs", "risk_model_package.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_monte_carlo_summary():
    path = os.path.join("outputs", "monte_carlo_summary.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

@st.cache_data
def fetch_brazil_geojson():
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# Load all outputs
corridor_df = load_corridor_data()
model_pkg = load_model_package()
mc_summary = load_monte_carlo_summary()

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://img.icons8.com/isometric/100/delivery.png", width=80)
st.sidebar.markdown("## Navigation & Config")
st.sidebar.markdown("This dashboard analyzes delivery cancellations (Return-To-Origin or RTO) and shipping cost efficiency.")

# Check database configuration
db_configured = False
db_path = os.path.join("data", "processed", "olist.db")
if os.path.exists(db_path):
    db_configured = True
    st.sidebar.success("Database Connected!")
else:
    st.sidebar.warning("Database file not found. Ensure load_and_clean.py was run.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Quick Stats:**")
if corridor_df is not None:
    st.sidebar.metric("Total Corridors Analyzed", f"{len(corridor_df)}")
if model_pkg is not None:
    st.sidebar.metric("Classifier ROC-AUC", f"{model_pkg['auc']:.2%}")

# ----------------- APP HEADER -----------------
st.markdown("<h1 class='main-title'>Cost-to-Serve & RTO Risk Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A decision-support system identifying unprofitable corridors, predicting delivery risks, and simulating policy impacts.</p>", unsafe_allow_html=True)

# ----------------- MAIN TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs(["🌎 Cost-to-Serve Corridors", "🔍 Order Risk Predictor", "📊 Policy Margin Simulator", "⚖️ Causal Impact (DiD)"])

# ================= TAB 1: COST HEATMAP =================
with tab1:
    st.header("Loss-Making Corridors & Cost-to-Serve Heatmap")
    st.markdown("""
    This tab highlights geographic regions and SKU category intersections that incur disproportionate logistics overhead. 
    **Cost-to-Serve** is computed as: `Freight Value + Cancelled Orders * (Freight Value * 1.5)`.
    """)
    
    if corridor_df is not None:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Cost-to-Serve Ratio by Customer State")
            
            # Aggregate to state level for the map
            state_summary = corridor_df.groupby('customer_state').agg(
                avg_cost_pct=('avg_cost_pct', 'mean'),
                total_orders=('total_orders', 'sum')
            ).reset_index()
            
            geojson = fetch_brazil_geojson()
            if geojson is not None:
                # Choropleth Map
                fig = px.choropleth(
                    state_summary,
                    geojson=geojson,
                    locations='customer_state',
                    featureidkey="properties.sigla",
                    color='avg_cost_pct',
                    color_continuous_scale="Reds",
                    labels={'avg_cost_pct': 'Cost % of Price'},
                    title="Average Cost-to-Serve as % of Order Price by State"
                )
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to bar chart if geojson download fails
                st.info("GeoJSON map source unavailable. Displaying bar chart fallback.")
                fig = px.bar(
                    state_summary.sort_values('avg_cost_pct', ascending=False),
                    x='customer_state',
                    y='avg_cost_pct',
                    color='avg_cost_pct',
                    color_continuous_scale="Reds",
                    labels={'avg_cost_pct': 'Cost % of Price', 'customer_state': 'State'},
                    title="Average Cost-to-Serve as % of Order Price by State"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        with col2:
            st.subheader("Top 10 Worst Category-State Corridors")
            display_df = corridor_df.copy()
            # Beautify column names for user display
            display_df.rename(columns={
                'product_category_name': 'Category',
                'customer_state': 'Customer State',
                'total_orders': 'Orders',
                'avg_cost_to_serve': 'Avg Cost (BRL)',
                'avg_cost_pct': 'Cost % of Price',
                'problem_rate': 'Problem Rate'
            }, inplace=True)
            
            # Format percentage columns
            display_df['Cost % of Price'] = display_df['Cost % of Price'].map(lambda x: f"{x:.2%}")
            display_df['Problem Rate'] = display_df['Problem Rate'].map(lambda x: f"{x:.2%}")
            display_df['Avg Cost (BRL)'] = display_df['Avg Cost (BRL)'].map(lambda x: f"BRL {x:.2f}")
            
            st.dataframe(
                display_df.head(10)[['Category', 'Customer State', 'Orders', 'Avg Cost (BRL)', 'Cost % of Price', 'Problem Rate']],
                hide_index=True,
                use_container_width=True
            )
            
            if corridor_df is not None and len(corridor_df) > 0:
                top_cor = display_df.iloc[0]
                st.markdown(f"""
                > **Key Finding:** The worst performing corridor is **{top_cor['Category']} in {top_cor['Customer State']}**, with an average cost-to-serve ratio representing **{top_cor['Cost % of Price']}** of the product price, and an order problem rate of **{top_cor['Problem Rate']}**.
                """)
    else:
        st.warning("Corridor summary data is missing. Run Phase 3 first.")

# ================= TAB 2: RISK PREDICTOR =================
with tab2:
    st.header("Order Delivery & Cancellation Risk Predictor")
    st.markdown("Use this interface to query the trained XGBoost model for risk estimations *before* an order is shipped.")
    
    if model_pkg is not None:
        # Load encoders/schemas
        categories = model_pkg['categories']
        feature_names = model_pkg['feature_names']
        model = model_pkg['model']
        
        # User input form
        st.subheader("Enter Order Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            price = st.number_input("Product Price (BRL)", min_value=1.0, max_value=10000.0, value=120.0, step=10.0)
            freight_value = st.number_input("Freight Charge (BRL)", min_value=0.0, max_value=1000.0, value=25.0, step=5.0)
            distance_km = st.number_input("Distance to Customer (km)", min_value=1.0, max_value=5000.0, value=450.0, step=50.0)
            
        with col2:
            payment_type = st.selectbox("Payment Method", categories['payment_type'])
            payment_installments = st.slider("Payment Installments", min_value=1, max_value=24, value=2)
            order_month = st.slider("Purchase Month (Seasonality)", min_value=1, max_value=12, value=7)
            
        with col3:
            product_category = st.selectbox("Product Category", categories['product_category_name'])
            customer_state = st.selectbox("Customer State", categories['customer_state'], index=categories['customer_state'].index('SP') if 'SP' in categories['customer_state'] else 0)
            seller_state = st.selectbox("Seller State", categories['seller_state'], index=categories['seller_state'].index('SP') if 'SP' in categories['seller_state'] else 0)
            
        if st.button("Predict Order Risk", type="primary"):
            # Construct a row corresponding to the original features
            # Categorical columns need one-hot encoding matching the training schema
            input_dict = {
                'price': price,
                'freight_value': freight_value,
                'distance_km': distance_km,
                'payment_installments': payment_installments,
                'order_month': order_month
            }
            
            # Set dummy categories
            # categorical features: ['payment_type', 'product_category_name', 'customer_state', 'seller_state']
            input_cat_vals = {
                'payment_type': payment_type,
                'product_category_name': product_category,
                'customer_state': customer_state,
                'seller_state': seller_state
            }
            
            # Map input categories to dummy names
            # format: f"{col}_{val}"
            for col, val in input_cat_vals.items():
                for v in categories[col]:
                    # skip base category if drop_first was used in training, but we can map everything
                    dummy_name = f"{col}_{v}"
                    if dummy_name in feature_names:
                        input_dict[dummy_name] = 1.0 if v == val else 0.0
                        
            # Fill missing feature columns from training (to handle drop_first or missing categories)
            for col in feature_names:
                if col not in input_dict:
                    input_dict[col] = 0.0
                    
            # Reorder columns to match the model training schema
            input_df = pd.DataFrame([input_dict])[feature_names]
            
            # Make prediction
            prob = model.predict_proba(input_df)[0, 1]
            
            # Display risk score
            st.markdown("---")
            st.subheader("Risk Score Analysis")
            
            r_col1, r_col2 = st.columns([1, 2])
            
            with r_col1:
                # Radial/gauge progress plot
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Problem Probability (%)", 'font': {'size': 16}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "red" if prob > 0.15 else "orange" if prob > 0.07 else "green"},
                        'steps': [
                            {'range': [0, 7], 'color': 'rgba(0, 128, 0, 0.1)'},
                            {'range': [7, 15], 'color': 'rgba(255, 165, 0, 0.1)'},
                            {'range': [15, 100], 'color': 'rgba(255, 0, 0, 0.1)'}
                        ]
                    }
                ))
                fig.update_layout(height=250, margin={"t":0, "b":0})
                st.plotly_chart(fig, use_container_width=True)
                
            with r_col2:
                # Interpret risk warning
                if prob > 0.15:
                    st.error("⚠️ HIGH RISK ORDER")
                    st.markdown(f"This order has a **{prob:.2%}** probability of experiencing delivery failure (delay or cancellation). Consider additional shipping verification, calling the customer, or using a premium carrier.")
                elif prob > 0.07:
                    st.warning("⚠️ MEDIUM RISK ORDER")
                    st.markdown(f"This order has a **{prob:.2%}** probability of experiencing delivery failure. Monitor transit closely.")
                else:
                    st.success("✅ LOW RISK ORDER")
                    st.markdown(f"This order has a **{prob:.2%}** probability of experiencing delivery failure. Standard delivery procedures apply.")
                    
                # Compute local perturbation explanation
                st.markdown("**Key Factors Influencing Risk (Local Attribution):**")
                
                # Baselines for comparison
                baselines = {
                    'distance_km': 150.0,
                    'price': 80.0,
                    'freight_value': 15.0,
                    'payment_installments': 1.0
                }
                
                contributions = []
                for feat, baseline_val in baselines.items():
                    if feat in feature_names:
                        perturbed_df = input_df.copy()
                        perturbed_df[feat] = baseline_val
                        prob_perturbed = model.predict_proba(perturbed_df)[0, 1]
                        change = prob - prob_perturbed
                        contributions.append({'Feature': feat, 'Change': change, 'UserValue': input_df[feat].iloc[0], 'Baseline': baseline_val})
                        
                contrib_df = pd.DataFrame(contributions)
                name_map = {
                    'distance_km': 'Distance (km)',
                    'price': 'Product Price (BRL)',
                    'freight_value': 'Freight Value (BRL)',
                    'payment_installments': 'Payment Installments'
                }
                contrib_df['Feature Name'] = contrib_df['Feature'].map(name_map)
                contrib_df['AbsChange'] = contrib_df['Change'].abs()
                contrib_df = contrib_df.sort_values('AbsChange', ascending=False)
                
                # Plot contributions
                fig_contrib = px.bar(
                    contrib_df,
                    x='Change',
                    y='Feature Name',
                    orientation='h',
                    color='Change',
                    color_continuous_scale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
                    color_continuous_midpoint=0,
                    labels={'Change': 'Risk Contribution (% points)', 'Feature Name': 'Feature'},
                    title="Risk Contribution of Factors vs. Reference Baseline"
                )
                fig_contrib.update_layout(
                    showlegend=False,
                    coloraxis_showscale=False,
                    height=220,
                    margin={"t":30, "b":10, "l":10, "r":10}
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                for _, row in contrib_df.head(3).iterrows():
                    direction = "increased" if row['Change'] > 0 else "decreased"
                    icon = "🔺" if row['Change'] > 0 else "🟢"
                    st.markdown(f"{icon} **{row['Feature Name']}** ({row['UserValue']}) {direction} prediction risk by **{abs(row['Change']):.2%}** compared to standard reference ({row['Baseline']}).")
    else:
        st.warning("Model package is missing. Run Phase 4 first.")

# ================= TAB 3: POLICY SIMULATOR =================
with tab3:
    st.header("Free Shipping Policy Margin Simulator")
    st.markdown("""
    Test the financial impact of changing the free-shipping threshold. 
    Currently, shipping subsidies cost Olist profit margins. If we restrict free shipping, we save subsidies but might lose order conversions.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulator Settings")
        threshold = st.slider(
            "Minimum Order Value for Free Shipping (BRL)", 
            min_value=0, 
            max_value=300, 
            value=80, 
            step=10,
            help="Orders below this price threshold will not receive free shipping."
        )
        
        drop_rate = st.slider(
            "Expected Order Drop Rate (%)", 
            min_value=0.0, 
            max_value=20.0, 
            value=5.0, 
            step=0.5,
            help="Percentage of orders below the threshold that are lost due to shipping charges."
        )
        
        profit_margin = st.slider(
            "Average Product Profit Margin (%)", 
            min_value=1.0, 
            max_value=30.0, 
            value=10.0, 
            step=1.0,
            help="Average profit margin of products before shipping costs."
        )
        
        st.markdown("""
        **Simulation Logic:**
        - Identifies orders that currently receive free shipping (`freight_value == 0`) but fall below the threshold.
        - Assumes we lose `Drop Rate %` of these orders (representing lost conversion).
        - For the remaining orders, the customer pays the freight cost, saving Olist the subsidy.
        - Calculates the net margin impact (Freight Saved + Cancelled Problem Costs Saved - Lost Profit Margin).
        
        **Data-Driven Anchor:**
        - Our causal **DiD Analysis** (see Tab 4) indicates that shipping subsidies did not statistically increase average order values. Removing subsidies will save the freight costs on remaining orders, but conversion drop-offs must be evaluated. We anchor our default expected drop rate to 5% based on typical e-commerce elasticity.
        """)
        
    with col2:
        st.subheader("Financial Projection Analysis")
        
        # Load database dynamically to run simulation on fresh data
        if db_configured:
            try:
                conn = sqlite3.connect(db_path)
                df_sim = pd.read_sql("SELECT price, freight_value, is_problem_order, purchase_ts FROM processed_orders", conn)
                conn.close()
                
                # Get dataset span in months
                df_sim['purchase_ts'] = pd.to_datetime(df_sim['purchase_ts'])
                days_span = (df_sim['purchase_ts'].max() - df_sim['purchase_ts'].min()).days
                months_span = max(1, days_span / 30.4)
                
                # Filter for subsidized orders below threshold
                subsidized_orders = df_sim[(df_sim['freight_value'] == 0) & (df_sim['price'] < threshold)]
                n_affected = len(subsidized_orders)
                
                if n_affected > 0:
                    # Compute mean freight dynamically from database (non-subsidized shipping average)
                    non_subsidized = df_sim[df_sim['freight_value'] > 0]
                    assumed_freight_saved = non_subsidized['freight_value'].mean() if len(non_subsidized) > 0 else 20.0
                    
                    avg_price = subsidized_orders['price'].mean()
                    prob_rate = subsidized_orders['is_problem_order'].mean()
                    
                    # 1. Revenue/Margin lost on lost conversions
                    lost_orders = n_affected * (drop_rate / 100.0)
                    margin_lost = lost_orders * avg_price * (profit_margin / 100.0)
                    
                    # 2. Freight subsidy saved on remaining orders
                    saved_freight_orders = n_affected * (1 - drop_rate / 100.0)
                    freight_subsidy_saved = saved_freight_orders * assumed_freight_saved
                    
                    # 3. Reverse logistics costs saved on lost problem orders
                    reverse_saved = lost_orders * prob_rate * (assumed_freight_saved * 1.5)
                    
                    net_impact_total = freight_subsidy_saved + reverse_saved - margin_lost
                    net_impact_monthly = net_impact_total / months_span
                    
                    # Metrics block
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Affected Orders (Total)", f"{n_affected:,}", help="Subsidized orders under threshold")
                    with m2:
                        st.metric("Estimated Monthly Freight Saved", f"BRL {freight_subsidy_saved/months_span:,.2f}")
                    with m3:
                        st.metric("Estimated Monthly Margin Lost", f"BRL {margin_lost/months_span:,.2f}")
                        
                    st.markdown("---")
                    
                    # Visual representation
                    fig_gau = go.Figure(go.Indicator(
                        mode = "number+delta",
                        value = net_impact_monthly,
                        number = {'prefix': "BRL ", 'valueformat': ",.2f"},
                        delta = {'position': "top", 'reference': 0},
                        title = {'text': "Net Monthly Margin Impact", 'font': {'size': 20}},
                        domain = {'x': [0, 1], 'y': [0, 1]}
                    ))
                    fig_gau.update_layout(height=180, margin={"t":30, "b":0})
                    st.plotly_chart(fig_gau, use_container_width=True)
                    
                    if net_impact_monthly > 0:
                        st.success(f"📈 **Positive Margin Impact:** Implementing a BRL {threshold} threshold is projected to improve monthly operating margins by **BRL {net_impact_monthly:,.2f}**.")
                    else:
                        st.error(f"📉 **Negative Margin Impact:** Implementing a BRL {threshold} threshold is projected to decrease monthly operating margins by **BRL {abs(net_impact_monthly):,.2f}** due to lost conversions.")
                        
                else:
                    st.info("No orders in the dataset would be affected by this threshold (all subsidized orders are priced above it, or freight subsidies are rare).")
            except Exception as e:
                st.error(f"Error running simulator: {e}")
        else:
            st.warning("Database not connected. Unable to pull processed order tables for simulation.")
            
    # Include Monte Carlo static results for Worst Corridors
    if mc_summary is not None:
        st.markdown("---")
        st.subheader("Worst Lanes Monte Carlo Simulation Baseline (Static Result)")
        
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Median Monthly Net Impact", f"BRL {mc_summary['median_impact']:,.2f}")
        with mc2:
            st.metric("5th Percentile (Risk Floor)", f"BRL {mc_summary['percentile_5']:,.2f}")
        with mc3:
            st.metric("95th Percentile (Return Ceiling)", f"BRL {mc_summary['percentile_95']:,.2f}")
            
        # Count the number of worst corridors dynamically
        if corridor_df is not None:
            cutoff = corridor_df['avg_cost_pct'].quantile(0.90)
            n_worst_lanes = len(corridor_df[corridor_df['avg_cost_pct'] >= cutoff])
        else:
            n_worst_lanes = 87
            
        st.info(f"This Monte Carlo simulation reflects the impact of eliminating free shipping entirely for the worst 10% corridors (representing {n_worst_lanes} lane-category combinations), showing that the policy change is highly sensitive to conversion drop rates and RTO costs.")
        
# ================= TAB 4: CAUSAL INFERENCE =================
with tab4:
    st.header("Difference-in-Differences (DiD) Causal Analysis")
    st.markdown("""
    Does free shipping actually drive higher order values (prices)? 
    To answer this causally, we compare subsidized shipping orders (**Treatment**) against paid shipping orders (**Control**) before and after a simulated policy change date of **2018-01-01** (**Pre vs. Post**).
    """)
    
    c_col1, c_col2 = st.columns([1, 1])
    
    with c_col1:
        st.subheader("Parallel Trends Validation")
        trends_img_path = os.path.join("outputs", "parallel_trends.png")
        if os.path.exists(trends_img_path):
            st.image(trends_img_path, caption="Monthly trends for Treated (Free Shipping) vs Control (Paid Shipping)", use_container_width=True)
        else:
            st.info("Parallel trends plot image is missing. Run Phase 5 script.")
            
    with c_col2:
        st.subheader("OLS Regression Results")
        summary_path = os.path.join("outputs", "did_regression_summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_text = f.read()
            st.text_area("DiD Regression Summary (statsmodels)", summary_text, height=380, disabled=True)
        else:
            st.warning("Regression summary output is missing. Run Phase 5 script.")
            
    st.markdown("---")
    st.subheader("Methodology, Assumptions & Takeaways")
    
    with st.expander("Show Methodology, Assumptions & Key Insights"):
        st.markdown(r"""
        ### Methodology
        We implement a Difference-in-Differences (DiD) framework to estimate the treatment effect:
        $$Price_{it} = \beta_0 + \beta_1 Treated_i + \beta_2 Post_t + \beta_3 (Treated_i \times Post_t) + \epsilon_{it}$$
        
        - **Treated ($Treated_i = 1$):** Orders with fully subsidized shipping (freight value = 0).
        - **Post ($Post_t = 1$):** Orders placed on or after the policy date (2018-01-01).
        - **Interaction Term ($Treated_i \times Post_t$):** The coefficient $\beta_3$ represents the true causal effect of the shipping subsidy on purchase value.
        
        ### Key Assumptions
        1. **Parallel Trends:** Prior to the intervention, the average prices of the treatment and control groups moved in parallel (validated in the monthly trends chart).
        2. **Exogeneity:** The simulated policy date (2018-01-01) represents a clean pre/post split without other major simultaneous confounding policy shifts.
        
        ### Business Takeaways
        * **DiD Estimator:** The DiD estimator is **-BRL 83.49** with a p-value of **0.138**.
        * **Selection Effects:** Subsidizing shipping did not causally drive higher-value item orders. Instead, it correlated with cheaper item purchases, indicating customers capitalized on the subsidy to purchase low-value goods.
        * **Lack of Statistical Significance:** Since $p > 0.05$, the price difference is not statistically significant. This suggests that shipping subsidies do not causally increase order value, supporting our recommendation to optimize logistics margins (regional warehouses/fulfillment centers) rather than offering broad customer shipping subsidies.
        """)
