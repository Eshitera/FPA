
import os
import subprocess
import pip
# app.py
import streamlit as st
import pandas as pd
import pickle
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import numpy as np
from packaging import version


# ==============================================================================
# 1. STREAMLIT CONFIGURATION
#    This must be the first Streamlit command in the script.
# ==============================================================================
st.set_page_config(layout="wide", page_title="Advanced FP&A Dashboard")

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
# 3. ADVANCED HELPER FUNCTIONS FOR SCENARIO MODELING
#    This function has been upgraded for more realistic modeling.
# ==============================================================================

def run_scenario(df, marketing_spend_change, churn_rate_change):
    scenario_df = df.copy()

    # Calculate historical ratios for a more robust model
    avg_cogs_ratio = (scenario_df['monthly_cogs'] / scenario_df['monthly_revenue']).mean()
    avg_op_ex_ratio = (scenario_df['total_operating_expenses'] / scenario_df['monthly_revenue']).mean()
    
    # Apply changes to key drivers
    scenario_df['monthly_revenue_scenario'] = scenario_df['monthly_revenue'] * (1 - churn_rate_change / 100)
    
    # Adjust marketing spend and re-calculate total marketing spend
    scenario_df['total_marketing_spend_scenario'] = (scenario_df['social_media_spend'] + scenario_df['search_engine_spend'] + scenario_df['affiliate_spend']) * (1 + marketing_spend_change / 100)
    
    # Calculate profitability metrics based on new revenue and spend
    scenario_df['monthly_cogs_scenario'] = scenario_df['monthly_revenue_scenario'] * avg_cogs_ratio
    scenario_df['gross_profit_scenario'] = scenario_df['monthly_revenue_scenario'] - scenario_df['monthly_cogs_scenario']
    
    # Calculate new total operating expenses, assuming marketing spend is part of it
    scenario_df['total_operating_expenses_scenario'] = scenario_df['monthly_revenue_scenario'] * avg_op_ex_ratio + (scenario_df['total_marketing_spend_scenario'] - (scenario_df['social_media_spend'] + scenario_df['search_engine_spend'] + scenario_df['affiliate_spend']))
    
    scenario_df['net_income_scenario'] = scenario_df['gross_profit_scenario'] - scenario_df['total_operating_expenses_scenario']
    scenario_df['net_profit_margin_scenario'] = (scenario_df['net_income_scenario'] / scenario_df['monthly_revenue_scenario']) * 100
    
    return scenario_df

# ==============================================================================
# 4. STREAMLIT APP LAYOUT AND CONTENT
# ==============================================================================

st.title("📊 Nova Essentials: Advanced FP&A Dashboard")

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

#st.dataframe(segment_summary, use_container_width=True)



def df_args():
    return {"width": "stretch"} if version.parse(st.__version__) >= version.parse("1.37.0") else {"use_container_width": True}

st.dataframe(segment_summary, **df_args())

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
    fig_comp.add_trace(go.Scatter(x=baseline_scenario['month'], y=baseline_scenario['net_income'], mode='lines', name='Baseline Net Income'))
    fig_comp.add_trace(go.Scatter(x=new_scenario['month'], y=new_scenario['net_income_scenario'], mode='lines', name='Scenario Net Income'))
    fig_comp.update_layout(title='Scenario Impact on Net Income', height=450, width=650, legend=dict(x=0, y=1.2, orientation='h'))
    st.plotly_chart(fig_comp)
    
    # Add a summary table to see the final numbers
    st.subheader("Scenario Financial Summary")
    final_baseline = baseline_scenario.iloc[-1]
    final_scenario = new_scenario.iloc[-1]
    
    summary_data = {
        "Metric": ["Monthly Revenue", "Gross Profit", "Net Income", "Net Profit Margin (%)"],
        "Baseline": [
            f"${final_baseline['monthly_revenue']:,.0f}",
            f"${final_baseline['gross_profit']:,.0f}",
            f"${final_baseline['net_income']:,.0f}",
            f"{final_baseline['net_profit_margin']:.2f}%"
        ],
        "Scenario": [
            f"${final_scenario['monthly_revenue_scenario']:,.0f}",
            f"${final_scenario['gross_profit_scenario']:,.0f}",
            f"${final_scenario['net_income_scenario']:,.0f}",
            f"{final_scenario['net_profit_margin_scenario']:.2f}%"
        ]
    }
    st.table(pd.DataFrame(summary_data))


# --- Final Business Recommendations ---
st.header("4. Key Business Recommendations")
st.markdown("""
Based on the analysis, here are key recommendations to drive growth and profitability:

- **Focus on High-Value Segments**: Direct marketing spend towards the 'Champions' customer segment, as they have the highest CLV and monetary value.
- **Implement Retention Campaigns**: Use the churn model to target 'At-Risk' customers with personalized offers to improve retention and CLV.
- **Optimize Marketing Channels**: Analyze the ROI of different marketing channels to re-allocate budget for maximum efficiency.
- **Monitor Profit Margins**: Continuously track profit margins to ensure operational costs are not eroding the business's bottom line.
""")