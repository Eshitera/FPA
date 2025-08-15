# app.py
import streamlit as st
import pandas as pd
import pickle
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import numpy as np

# ==============================================================================
# 1. STREAMLIT CONFIGURATION
#    This must be the first Streamlit command in the script.
# ==============================================================================
st.set_page_config(layout="wide")

# ==============================================================================
# 2. DATA AND MODEL LOADING
#    This section loads all the files you created in the previous phase.
# ==============================================================================

try:
    @st.cache_data
    def load_data():
        master_df = pd.read_csv('ecomm_fpna_master_data.csv')
        master_df['month'] = pd.to_datetime(master_df['month'])
        
        customer_segments_df = pd.read_csv('customer_segments.csv')
        
        # Load Prophet forecast data
        prophet_forecast_df = pd.read_csv('prophet_forecast.csv')
        prophet_forecast_df['ds'] = pd.to_datetime(prophet_forecast_df['ds'])
        
        return master_df, customer_segments_df, prophet_forecast_df

    master_df, customer_segments_df, prophet_forecast_df = load_data()
    
    # Load the churn model
    with open('churn_model.pkl', 'rb') as f:
        churn_model = pickle.load(f)
        
    # Load the Prophet model for scenario analysis
    with open('prophet_model.pkl', 'rb') as f:
        prophet_model = pickle.load(f)

except FileNotFoundError as e:
    st.error(f"Error: {e}. Please ensure all necessary CSV and model files are in the same directory as this app.py file.")
    st.stop()


# ==============================================================================
# 3. HELPER FUNCTIONS FOR SCENARIO MODELING
# ==============================================================================

def run_scenario(df, marketing_spend_change, churn_rate_change):
    scenario_df = df.copy()

    # Calculate the total marketing spend from individual channels
    scenario_df['total_marketing_spend'] = scenario_df['social_media_spend'] + scenario_df['search_engine_spend'] + scenario_df['affiliate_spend']
    
    # Apply changes to key drivers
    scenario_df['total_marketing_spend'] = scenario_df['total_marketing_spend'] * (1 + marketing_spend_change / 100)
    
    # A simplified way to model churn's impact on revenue.
    # We assume revenue is impacted by churn.
    scenario_df['monthly_revenue'] = scenario_df['monthly_revenue'] * (1 - churn_rate_change / 100)

    # Re-calculate profitability metrics based on new revenue and spend
    scenario_df['monthly_cogs'] = scenario_df['monthly_revenue'] * (scenario_df['monthly_cogs'] / scenario_df['monthly_revenue'].shift(1)).fillna(0.3)
    scenario_df['gross_profit'] = scenario_df['monthly_revenue'] - scenario_df['monthly_cogs']
    scenario_df['total_operating_expenses'] = scenario_df['total_operating_expenses'].replace(np.nan, 0)
    scenario_df['net_income'] = (scenario_df['gross_profit'] - scenario_df['total_operating_expenses']) * 0.75

    return scenario_df

# ==============================================================================
# 4. STREAMLIT APP LAYOUT AND CONTENT
# ==============================================================================

st.title("📊 Nova Essentials: Strategic FP&A Dashboard")

# Sidebar for Scenario Comparison Tool
st.sidebar.header("Scenario Comparison Tool")
st.sidebar.write("Adjust the sliders to model different business strategies.")
marketing_spend_change = st.sidebar.slider("Change in Marketing Spend (%)", -20, 20, 0, 5)
churn_rate_change = st.sidebar.slider("Change in Churn Rate (%)", -5, 5, 0, 1)

# --- Main Dashboard ---
st.header("1. Core Financial Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Revenue (3Y)", f"${master_df['monthly_revenue'].sum():,.0f}")
with col2:
    st.metric("Avg Net Profit Margin", f"{master_df['net_profit_margin'].mean():.2f}%")
with col3:
    st.metric("Avg Monthly New Customers", f"{master_df['new_customers_acquired'].mean():.0f}")

st.subheader("Historical Revenue & Profitability")
st.line_chart(master_df.set_index('month')[['monthly_revenue', 'net_income']])

# --- Customer Segmentation Analysis ---
st.header("2. Customer Segmentation Analysis")
st.write("Customers are segmented based on their purchasing behavior (Recency, Frequency, Monetary) and estimated Lifetime Value (CLV).")

segment_summary = customer_segments_df.groupby('cluster').agg(
    count=('customer_id', 'count'),
    avg_clv=('clv', 'mean'),
    avg_monetary=('monetary', 'mean'),
    avg_recency=('recency', 'mean')
).reset_index()

segment_summary['cluster'] = segment_summary['cluster'].map({
    0: 'Champions (High-Value)',
    1: 'Loyalists (Steady)',
    2: 'At-Risk (High Recency)',
    3: 'Newbies (Low Frequency)'
})

st.dataframe(segment_summary, use_container_width=True)

# --- Revenue Forecast & Scenario Modeling ---
st.header("3. Revenue Forecast & Scenario Modeling")
st.write("View the 12-month revenue forecast and compare a baseline to your customized scenario.")

col_forecast1, col_forecast2 = st.columns(2)

with col_forecast1:
    st.subheader("Baseline Forecast")
    fig1 = plot_plotly(prophet_model, prophet_forecast_df)
    st.plotly_chart(fig1)

with col_forecast2:
    st.subheader("Scenario Comparison")
    
    # Run scenarios
    baseline_scenario = run_scenario(master_df, 0, 0)
    new_scenario = run_scenario(master_df, marketing_spend_change, churn_rate_change)
    
    # Create comparison chart
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=baseline_scenario['month'], y=baseline_scenario['monthly_revenue'], mode='lines', name='Baseline'))
    fig_comp.add_trace(go.Scatter(x=new_scenario['month'], y=new_scenario['monthly_revenue'], mode='lines', name='Scenario'))
    fig_comp.update_layout(title='Scenario Impact on Monthly Revenue', height=450, width=650)
    st.plotly_chart(fig_comp)


# --- Final Business Recommendations ---
st.header("4. Key Business Recommendations")
st.markdown("""
Based on the analysis, here are key recommendations to drive growth and profitability:

- **Focus on High-Value Segments**: Direct marketing spend towards the 'Champions' customer segment, as they have the highest CLV and monetary value.
- **Implement Retention Campaigns**: Use the churn model to target 'At-Risk' customers with personalized offers to improve retention and CLV.
- **Optimize Marketing Channels**: Analyze the ROI of different marketing channels to re-allocate budget for maximum efficiency.
- **Monitor Profit Margins**: Continuously track profit margins to ensure operational costs are not eroding the business's bottom line.
""")