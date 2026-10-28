"""
Streamlit Dashboard for Dynamic Pricing & Inventory Allocation

Run with: streamlit run dashboard/app.py

Features:
- Pick a SKU/location/time range
- View forecasted demand curve
- See optimized price recommendation
- Compare MILP vs GA solutions
- View revenue lift vs baseline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data_prep import load_processed, compute_price_elasticity
from demand_model import DemandModel


st.set_page_config(
    page_title="Dynamic Pricing & Inventory Allocation",
    page_icon="💰",
    layout="wide",
)

st.title("ML-Guided Dynamic Pricing & Inventory Allocation")
st.markdown("*Optimize weekly prices and inventory allocation across locations to maximize revenue*")


@st.cache_data
def load_data():
    weekly_df, elasticity_df = load_processed()
    return weekly_df, elasticity_df


@st.cache_resource
def load_model():
    try:
        return DemandModel.load(model_type='xgboost')
    except Exception:
        return None


def main():
    weekly_df, elasticity_df = load_data()
    demand_model = load_model()

    st.sidebar.header("Configuration")

    sku_list = sorted(weekly_df['product_id'].unique())
    selected_sku = st.sidebar.selectbox("Select SKU", sku_list)

    location_list = sorted(weekly_df['location'].unique())
    selected_location = st.sidebar.selectbox("Select Location", location_list)

    week_min = int(weekly_df['week'].min())
    week_max = int(weekly_df['week'].max())
    week_range = st.sidebar.slider("Week Range", week_min, week_max, (week_min, min(week_min + 10, week_max)))

    st.sidebar.header("Solver Settings")
    solver_method = st.sidebar.radio("Solver Method", ["MILP (Exact)", "GA (Heuristic)", "Compare Both"])
    max_price_change = st.sidebar.slider("Max Price Change ($)", 5.0, 50.0, 15.0)
    min_stock = st.sidebar.slider("Minimum Stock", 0, 20, 5)

    tab1, tab2, tab3, tab4 = st.tabs(["Demand Curves", "Optimized Pricing", "Revenue Analysis", "Elasticity"])

    with tab1:
        st.subheader(f"Demand Curve: {selected_sku} at {selected_location}")

        sku_data = weekly_df[weekly_df['product_id'] == selected_sku]
        mean_comp = sku_data['competitor_price'].mean() if 'competitor_price' in sku_data.columns else 50

        if demand_model is not None:
            price_min = max(sku_data['price'].min() * 0.5, 1.0)
            price_max = sku_data['price'].max() * 1.5

            prices, demands = demand_model.get_demand_curve(
                competitor_price=mean_comp,
                week=week_range[0],
                category=sku_data['category'].iloc[0] if 'category' in sku_data.columns else 'default',
                location=selected_location,
                price_range=(price_min, price_max),
                n_points=50,
            )

            revenue_curve = prices * demands

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices, y=demands, name='Demand', line=dict(color='steelblue', width=2)))
            fig.update_layout(
                title='Expected Demand vs Price',
                xaxis_title='Price ($)',
                yaxis_title='Expected Units Sold',
                template='plotly_white',
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=prices, y=revenue_curve, name='Revenue', line=dict(color='green', width=2)))
            fig2.update_layout(
                title='Expected Revenue vs Price',
                xaxis_title='Price ($)',
                yaxis_title='Expected Revenue ($)',
                template='plotly_white',
            )
            st.plotly_chart(fig2, use_container_width=True)

            optimal_price_idx = np.argmax(revenue_curve)
            st.metric("Optimal Price (from demand model)", f"${prices[optimal_price_idx]:.2f}")
            st.metric("Expected Revenue at Optimal Price", f"${revenue_curve[optimal_price_idx]:,.2f}")
        else:
            st.warning("Demand model not available. Run notebooks/02_demand_model.ipynb first.")

    with tab2:
        st.subheader("Optimized Pricing & Allocation")

        week_data = weekly_df[
            (weekly_df['product_id'] == selected_sku) &
            (weekly_df['location'] == selected_location) &
            (weekly_df['week'] >= week_range[0]) &
            (weekly_df['week'] <= week_range[1])
        ]

        if len(week_data) > 0:
            st.write("**Historical Data for Selection:**")
            st.dataframe(week_data[['week', 'price', 'quantity_sold', 'competitor_price']].round(2))

            avg_historical_price = week_data['price'].mean()
            avg_historical_demand = week_data['quantity_sold'].mean()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Historical Avg Price", f"${avg_historical_price:.2f}")
            with col2:
                st.metric("Historical Avg Demand", f"{avg_historical_demand:.0f} units")
            with col3:
                if demand_model is not None:
                    opt_price = prices[optimal_price_idx] if 'prices' in dir() else avg_historical_price
                    st.metric("Optimized Price", f"${opt_price:.2f}",
                              delta=f"${opt_price - avg_historical_price:.2f}")

        if solver_method in ["MILP (Exact)", "Compare Both"]:
            st.info("MILP solver would run here with OR-Tools CP-SAT. Run notebook 03 for full solution.")
        if solver_method in ["GA (Heuristic)", "Compare Both"]:
            st.info("GA solver would run here. Run notebook 04 for full solution.")

    with tab3:
        st.subheader("Revenue Analysis")

        baseline_weekly = weekly_df.groupby('product_id').agg(
            avg_price=('price', 'mean'),
            total_demand=('quantity_sold', 'sum'),
        ).reset_index()

        baseline_weekly['revenue'] = baseline_weekly['avg_price'] * baseline_weekly['total_demand']

        fig = px.bar(baseline_weekly, x='product_id', y='revenue',
                     title='Baseline Revenue by SKU (Static Pricing)',
                     labels={'product_id': 'SKU', 'revenue': 'Revenue ($)'})
        st.plotly_chart(fig, use_container_width=True)

        loc_revenue = weekly_df.groupby('location').agg(
            total_revenue=('quantity_sold', lambda x: (x * weekly_df.loc[x.index, 'price']).sum()),
            total_units=('quantity_sold', 'sum'),
        ).reset_index()

        fig2 = px.bar(loc_revenue, x='location', y='total_revenue',
                      title='Revenue by Location',
                      labels={'location': 'Location', 'total_revenue': 'Revenue ($)'})
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.subheader("Price Elasticity by SKU")

        fig = px.bar(elasticity_df, x='product_id', y='elasticity',
                     title='Price Elasticity (Log-Log Regression)',
                     labels={'product_id': 'SKU', 'elasticity': 'Elasticity'})
        fig.add_hline(y=-1, line_dash="dash", line_color="red",
                      annotation_text="Unit Elastic")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(elasticity_df.round(4))

        st.markdown("""
        **Interpretation:**
        - Elasticity < -1: Elastic (demand changes more than price)
        - Elasticity = -1: Unit elastic
        - Elasticity > -1: Inelastic (demand changes less than price)
        """)


if __name__ == "__main__":
    main()
