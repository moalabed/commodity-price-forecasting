import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import sqlite3
import os
import sys
from datetime import datetime, timedelta
from prophet import Prophet
import yfinance as yf

# Add the src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import local modules
try:
    from commodity_price_data import COMMODITIES, create_database, fetch_commodity_data
    from forecast_commodity_prices import (
        prepare_data_for_prophet, train_prophet_model, create_prophet_components_plot,
        train_arima_model, train_lstm_model, VIBRANT_COLORS
    )
except ImportError:
    # If that fails, try with the full path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.commodity_price_data import COMMODITIES, create_database, fetch_commodity_data
    from src.forecast_commodity_prices import (
        prepare_data_for_prophet, train_prophet_model, create_prophet_components_plot, 
        train_arima_model, train_lstm_model, VIBRANT_COLORS
    )

# Set page config
st.set_page_config(
    page_title="Commodity Price Forecasting",
    page_icon="📈",
    layout="wide"
)

# Database path
DB_PATH = 'commodities.db'

# Define forecasting models
FORECASTING_MODELS = {
    "Prophet": "Facebook Prophet - handles seasonality well, good for long-term forecasts",
    "ARIMA": "Autoregressive Integrated Moving Average - classical statistical approach",
    "LSTM": "Long Short-Term Memory Neural Network - captures complex patterns but requires more data"
}

def load_commodity_data(commodity_name=None, start_date=None):
    """Load commodity data from SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    
    query = "SELECT * FROM commodities"
    conditions = []
    
    if commodity_name and commodity_name != "All":
        conditions.append(f"name = '{commodity_name}'")
    
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY date"
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def create_correlation_heatmap(df, commodities, years=None):
    """Create an interactive correlation heatmap between commodities"""
    if df.empty:
        return None
    
    # Filter by years if specified
    if years:
        end_date = df['date'].max()
        start_date = end_date - timedelta(days=365 * years)
        df = df[df['date'] >= start_date]
    
    # Create pivot table with dates as index and commodities as columns
    pivot_df = pd.pivot_table(df, values='close', index='date', columns='name')
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    
    # Create heatmap with Plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Plasma',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation'),
        text=np.round(corr_matrix.values, 2),
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='Commodity Price Correlation Matrix',
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_historical_prices(df, title, years=None):
    """Plot historical prices for the selected commodity using Plotly in Yahoo Finance style"""
    if df.empty:
        st.warning("No data available for the selected time period.")
        return
    
    # Filter by years if specified
    if years:
        end_date = df['date'].max()
        start_date = end_date - timedelta(days=365 * years)
        df = df[df['date'] >= start_date]
    
    # Create interactive Plotly figure
    fig = go.Figure()
    
    # Add price line using the green color as in Yahoo Finance
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='#00873c', width=1.5)  # Yahoo Finance green
        )
    )
    
    # Fill area under the line with light green
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(0, 135, 60, 0.1)',  # Light green fill
            showlegend=False
        )
    )
    
    # Calculate and mark the current price with a dot
    latest_date = df['date'].max()
    latest_price = df.loc[df['date'] == latest_date, 'close'].values[0]
    
    fig.add_trace(
        go.Scatter(
            x=[latest_date],
            y=[latest_price],
            mode='markers',
            marker=dict(color='#00873c', size=8),
            showlegend=False
        )
    )
    
    # Get price change info (for 1D/5D/1M/YTD, etc. that we'll add as buttons)
    current_price = df.iloc[-1]['close']
    
    # Update layout to match Yahoo Finance style
    fig.update_layout(
        title=None,  # Yahoo Finance doesn't show title on the chart itself
        xaxis=dict(
            title=None,  # No x-axis title
            rangeslider=dict(visible=True, thickness=0.05),  # Thinner rangeslider
            type='date',
            gridcolor='rgba(230, 230, 230, 0.3)',
            linecolor='rgb(234, 234, 234)',
            showgrid=False,
        ),
        yaxis=dict(
            title=None,  # No y-axis title
            gridcolor='rgba(230, 230, 230, 0.3)',
            linecolor='rgb(234, 234, 234)',
            showgrid=True,
            zeroline=False,
            tickformat='$.2f',  # Format with dollar sign
            side='right',  # Yahoo Finance shows y-axis on the right
        ),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500,
        margin=dict(l=0, r=40, t=10, b=20),  # Tighter margins like Yahoo Finance
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    # Add buttons for time range selection as in Yahoo Finance
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.1,
                buttons=list([
                    dict(
                        args=[{"xaxis.range": [(df['date'].max() - timedelta(days=1)), df['date'].max()]}],
                        label="1D",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [(df['date'].max() - timedelta(days=5)), df['date'].max()]}],
                        label="5D",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [(df['date'].max() - timedelta(days=30)), df['date'].max()]}],
                        label="1M",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [(df['date'].max() - timedelta(days=90)), df['date'].max()]}],
                        label="3M",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [(df['date'].max() - timedelta(days=180)), df['date'].max()]}],
                        label="6M",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [(df['date'].max() - timedelta(days=365)), df['date'].max()]}],
                        label="1Y",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [(df['date'].max() - timedelta(days=365*2)), df['date'].max()]}],
                        label="2Y",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [(df['date'].max() - timedelta(days=365*5)), df['date'].max()]}],
                        label="5Y",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [df['date'].min(), df['date'].max()]}],
                        label="MAX",
                        method="relayout"
                    ),
                ]),
                font=dict(size=12),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.1)',
                borderwidth=1,
                pad=dict(t=10, r=10, b=10, l=10),
                active=7 if years and years == 5 else 8,  # Highlight the active button
            )
        ]
    )
    
    return fig

def generate_forecast(df, commodity_name, forecast_days=365, model_name="Prophet"):
    """Generate and visualize forecast for the selected commodity using Plotly with the selected model"""
    
    if model_name == "Prophet":
        # Prepare data for Prophet
        df_prophet = prepare_data_for_prophet(df)
        
        # Train model and generate forecast
        with st.spinner(f"Training Prophet forecasting model for {commodity_name}..."):
            model, forecast = train_prophet_model(df_prophet, forecast_days)
            
            # Get forecast components for Prophet only
            fig_components = create_prophet_components_plot(model, forecast)
    
    elif model_name == "ARIMA":
        # Train ARIMA model and generate forecast
        with st.spinner(f"Training ARIMA forecasting model for {commodity_name}..."):
            model, forecast = train_arima_model(df, forecast_days)
            
            # ARIMA doesn't have components like Prophet
            fig_components = None
    
    elif model_name == "LSTM":
        # Train LSTM model and generate forecast
        with st.spinner(f"Training LSTM neural network for {commodity_name}... This may take a while."):
            model, forecast = train_lstm_model(df, forecast_days)
            
            # LSTM doesn't have components like Prophet
            fig_components = None
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get the current price and forecasted price
    last_date = df['date'].max()
    last_price = df.loc[df['date'] == last_date, 'close'].values[0]
    
    forecast_end_date = last_date + timedelta(days=forecast_days)
    forecast_price = forecast.loc[forecast['ds'] == pd.Timestamp(forecast_end_date), 'yhat'].values[0]
    
    forecast_lower = forecast.loc[forecast['ds'] == pd.Timestamp(forecast_end_date), 'yhat_lower'].values[0]
    forecast_upper = forecast.loc[forecast['ds'] == pd.Timestamp(forecast_end_date), 'yhat_upper'].values[0]
    
    percent_change = ((forecast_price / last_price) - 1) * 100
    
    # Create interactive forecast plot with Plotly
    fig_forecast = go.Figure()
    
    # Add confidence intervals
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line=dict(color='rgba(0, 0, 0, 0)'),
            showlegend=False,
            name='Upper Bound'
        )
    )
    
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(0, 0, 0, 0)'),
            fillcolor='rgba(0, 135, 60, 0.2)',  # Light green matching Yahoo Finance
            name='95% Confidence Interval'
        )
    )
    
    # Mark the dividing line between historical and forecast
    fig_forecast.add_shape(
        type="line",
        x0=last_date,
        y0=0,
        x1=last_date,
        y1=1,
        yref="paper",
        line=dict(
            color='rgba(0, 0, 0, 0.3)',
            width=1.5,
            dash="dash",
        )
    )
    
    # Add historical data as a solid line
    historical_mask = forecast['ds'] <= last_date
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast.loc[historical_mask, 'ds'],
            y=forecast.loc[historical_mask, 'y'] if 'y' in forecast.columns else forecast.loc[historical_mask, 'yhat'],
            mode='lines',
            line=dict(color='#00873c', width=1.5),
            name='Historical'
        )
    )
    
    # Add forecast line
    forecast_mask = forecast['ds'] > last_date
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast.loc[forecast_mask, 'ds'],
            y=forecast.loc[forecast_mask, 'yhat'],
            mode='lines',
            line=dict(color='#ff6a00', width=1.5),  # Orange for forecast line
            name='Forecast'
        )
    )
    
    # Update layout to match Yahoo Finance style
    fig_forecast.update_layout(
        title=None,
        xaxis=dict(
            title=None,
            rangeslider=dict(visible=True, thickness=0.05),
            type='date',
            gridcolor='rgba(230, 230, 230, 0.3)',
            linecolor='rgb(234, 234, 234)',
            showgrid=False,
        ),
        yaxis=dict(
            title=None,
            gridcolor='rgba(230, 230, 230, 0.3)',
            linecolor='rgb(234, 234, 234)',
            showgrid=True,
            zeroline=False,
            tickformat='$.2f',
            side='right',
        ),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500,
        margin=dict(l=0, r=40, t=10, b=20),
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    # Add time range selection buttons
    fig_forecast.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.1,
                buttons=list([
                    dict(
                        args=[{"xaxis.range": [(last_date - timedelta(days=30)), (last_date + timedelta(days=30))]}],
                        label="2M",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [(last_date - timedelta(days=30)), (last_date + timedelta(days=90))]}],
                        label="4M",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [(last_date - timedelta(days=90)), (last_date + timedelta(days=180))]}],
                        label="9M",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [(last_date - timedelta(days=180)), forecast_end_date]}],
                        label="1Y+",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [forecast['ds'].min(), forecast['ds'].max()]}],
                        label="MAX",
                        method="relayout"
                    ),
                ]),
                font=dict(size=12),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.1)',
                borderwidth=1,
                pad=dict(t=10, r=10, b=10, l=10),
                active=3 if forecast_days >= 365 else 2,
            )
        ]
    )
    
    return fig_forecast, fig_components, {
        "last_date": last_date,
        "last_price": last_price,
        "forecast_end_date": forecast_end_date,
        "forecast_price": forecast_price,
        "forecast_lower": forecast_lower,
        "forecast_upper": forecast_upper,
        "percent_change": percent_change,
        "model_name": model_name
    }

def main():
    # Initialize session states
    if 'updating' not in st.session_state:
        st.session_state.updating = False
    
    # Header
    st.title("Commodity Price Forecasting")
    st.markdown("""
    This app allows you to visualize historical commodity prices and generate forecasts using machine learning.
    Select a commodity from the dropdown and customize your view with the options below.
    """)
    
    # Database check and setup
    if not os.path.exists(DB_PATH):
        st.warning("Database not found. Please fetch the initial data.")
        if st.button("Fetch Initial Data"):
            with st.spinner("Fetching commodity data (this may take several minutes)..."):
                fetch_commodity_data('2000-01-01', DB_PATH)
            st.success("Initial data fetched successfully!")
            st.experimental_rerun()
        return
    
    # Sidebar for controls
    st.sidebar.title("Controls")
    
    # Add automatic update notification instead of refresh button
    with st.sidebar:
        st.info("📅 Data is automatically updated daily at midnight")
        
        # Show last update time
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(date) FROM commodities")
            last_date = cursor.fetchone()[0]
            conn.close()
            
            if last_date:
                st.caption(f"Last updated: {last_date}")
        except:
            st.caption("Database status: Unknown")
    
    # Commodity selection
    commodities_list = ["All"] + list(COMMODITIES.keys())
    selected_commodity = st.sidebar.selectbox(
        "Select Commodity",
        commodities_list,
        index=0
    )
    
    # Time range for historical data
    time_options = {
        "All Time": None,
        "Last 10 Years": 10,
        "Last 5 Years": 5,
        "Last 2 Years": 2,
        "Last Year": 1
    }
    selected_time = st.sidebar.selectbox(
        "Time Range",
        list(time_options.keys()),
        index=2
    )
    years_to_show = time_options[selected_time]
    
    # Theme selection for plots
    available_themes = ["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn"]
    selected_theme = st.sidebar.selectbox(
        "Visualization Theme",
        available_themes,
        index=0
    )
    
    # Load data
    if selected_commodity == "All":
        df = load_commodity_data(start_date="2000-01-01")
        
        # Add correlation analysis tab
        st.subheader("Commodity Correlation Analysis")
        st.markdown("This heatmap shows how commodity prices move in relation to each other.")
        
        corr_fig = create_correlation_heatmap(df, COMMODITIES.keys(), years_to_show)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
        
        # Group data for visualization
        grouped_commodities = {
            "Energy": ["OIL", "NATURAL_GAS"],
            "Metals": ["GOLD", "SILVER", "COPPER"],
            "Agriculture": ["CORN", "WHEAT", "SOYBEANS", "COFFEE", "SUGAR"]
        }
        
        st.subheader("Commodity Price Trends")
        
        # Display each group in tabs
        tabs = st.tabs(list(grouped_commodities.keys()) + ["All Commodities"])
        
        # For each group, create a normalized plot
        for i, (group_name, commodities) in enumerate(grouped_commodities.items()):
            with tabs[i]:
                group_df = df[df['name'].isin(commodities)]
                
                if not group_df.empty:
                    # Create interactive Plotly chart for normalized comparison
                    fig = go.Figure()
                    
                    for i, commodity in enumerate(commodities):
                        commodity_df = group_df[group_df['name'] == commodity].sort_values('date')
                        if not commodity_df.empty and years_to_show:
                            end_date = commodity_df['date'].max()
                            start_date = end_date - timedelta(days=365 * years_to_show)
                            commodity_df = commodity_df[commodity_df['date'] >= start_date]
                        
                        if not commodity_df.empty:
                            # Normalize prices
                            prices = commodity_df['close'].values
                            min_price = np.min(prices)
                            max_price = np.max(prices)
                            if max_price > min_price:  # Avoid division by zero
                                norm_prices = (prices - min_price) / (max_price - min_price)
                                fig.add_trace(
                                    go.Scatter(
                                        x=commodity_df['date'],
                                        y=norm_prices,
                                        mode='lines',
                                        name=commodity,
                                        line=dict(color=VIBRANT_COLORS[i % len(VIBRANT_COLORS)], width=2)
                                    )
                                )
                    
                    # Update layout with interactive features
                    fig.update_layout(
                        title=f"{group_name} Commodity Prices (Normalized)",
                        xaxis=dict(
                            title='Date',
                            rangeslider=dict(visible=True),
                            type='date'
                        ),
                        yaxis=dict(
                            title='Normalized Price',
                            gridcolor='rgba(230, 230, 230, 0.3)'
                        ),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        height=500,
                        template=selected_theme,
                        hovermode='x unified',
                        plot_bgcolor='rgba(255, 255, 255, 0.9)'
                    )
                    
                    # Add buttons for time range selection
                    if not group_df.empty:
                        min_date = group_df['date'].min()
                        max_date = group_df['date'].max()
                        fig.update_layout(
                            updatemenus=[
                                dict(
                                    type="buttons",
                                    direction="right",
                                    x=0.1,
                                    y=1.1,
                                    buttons=list([
                                        dict(
                                            args=[{"xaxis.range": [min_date, max_date]}],
                                            label="All Time",
                                            method="relayout"
                                        ),
                                        dict(
                                            args=[{"xaxis.range": [(max_date - timedelta(days=365*5)), max_date]}],
                                            label="5 Years",
                                            method="relayout"
                                        ),
                                        dict(
                                            args=[{"xaxis.range": [(max_date - timedelta(days=365)), max_date]}],
                                            label="1 Year",
                                            method="relayout"
                                        ),
                                        dict(
                                            args=[{"xaxis.range": [(max_date - timedelta(days=90)), max_date]}],
                                            label="3 Months",
                                            method="relayout"
                                        ),
                                    ]),
                                )
                            ]
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add raw price comparison toggle
                    if st.checkbox(f"Show {group_name} Raw Prices (Non-Normalized)", key=f"raw_{group_name}"):
                        raw_fig = go.Figure()
                        
                        for i, commodity in enumerate(commodities):
                            commodity_df = group_df[group_df['name'] == commodity].sort_values('date')
                            if not commodity_df.empty and years_to_show:
                                end_date = commodity_df['date'].max()
                                start_date = end_date - timedelta(days=365 * years_to_show)
                                commodity_df = commodity_df[commodity_df['date'] >= start_date]
                            
                            if not commodity_df.empty:
                                raw_fig.add_trace(
                                    go.Scatter(
                                        x=commodity_df['date'],
                                        y=commodity_df['close'],
                                        mode='lines',
                                        name=commodity,
                                        line=dict(color=VIBRANT_COLORS[i % len(VIBRANT_COLORS)], width=2)
                                    )
                                )
                        
                        raw_fig.update_layout(
                            title=f"{group_name} Commodity Prices (Raw USD)",
                            xaxis=dict(
                                title='Date',
                                rangeslider=dict(visible=True),
                                type='date'
                            ),
                            yaxis=dict(
                                title='Price (USD)',
                                gridcolor='rgba(230, 230, 230, 0.3)'
                            ),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                            height=500,
                            template=selected_theme,
                            hovermode='x unified',
                            plot_bgcolor='rgba(255, 255, 255, 0.9)'
                        )
                        
                        st.plotly_chart(raw_fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {group_name} commodities.")
        
        # All commodities tab
        with tabs[-1]:
            # Show all commodities in a normalized plot with Plotly
            fig = go.Figure()
            
            for i, commodity in enumerate(COMMODITIES):
                commodity_df = df[df['name'] == commodity].sort_values('date')
                if not commodity_df.empty and years_to_show:
                    end_date = commodity_df['date'].max()
                    start_date = end_date - timedelta(days=365 * years_to_show)
                    commodity_df = commodity_df[commodity_df['date'] >= start_date]
                
                if not commodity_df.empty:
                    # Normalize prices
                    prices = commodity_df['close'].values
                    min_price = np.min(prices)
                    max_price = np.max(prices)
                    if max_price > min_price:  # Avoid division by zero
                        norm_prices = (prices - min_price) / (max_price - min_price)
                        fig.add_trace(
                            go.Scatter(
                                x=commodity_df['date'],
                                y=norm_prices,
                                mode='lines',
                                name=commodity,
                                line=dict(color=VIBRANT_COLORS[i % len(VIBRANT_COLORS)], width=2)
                            )
                        )
            
            # Update layout with interactive features
            fig.update_layout(
                title="All Commodity Prices (Normalized)",
                xaxis=dict(
                    title='Date',
                    rangeslider=dict(visible=True),
                    type='date'
                ),
                yaxis=dict(
                    title='Normalized Price',
                    gridcolor='rgba(230, 230, 230, 0.3)'
                ),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=500,
                template=selected_theme,
                hovermode='x unified',
                plot_bgcolor='rgba(255, 255, 255, 0.9)'
            )
            
            # Add time range selector buttons
            if not df.empty:
                min_date = df['date'].min()
                max_date = df['date'].max()
                fig.update_layout(
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="right",
                            x=0.1,
                            y=1.1,
                            buttons=list([
                                dict(
                                    args=[{"xaxis.range": [min_date, max_date]}],
                                    label="All Time",
                                    method="relayout"
                                ),
                                dict(
                                    args=[{"xaxis.range": [(max_date - timedelta(days=365*5)), max_date]}],
                                    label="5 Years",
                                    method="relayout"
                                ),
                                dict(
                                    args=[{"xaxis.range": [(max_date - timedelta(days=365)), max_date]}],
                                    label="1 Year",
                                    method="relayout"
                                ),
                                dict(
                                    args=[{"xaxis.range": [(max_date - timedelta(days=90)), max_date]}],
                                    label="3 Months",
                                    method="relayout"
                                ),
                            ]),
                        )
                    ]
                )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Single commodity view
        df = load_commodity_data(selected_commodity)
        
        if df.empty:
            st.error(f"No data found for {selected_commodity}. Please try refreshing the data.")
            return
        
        # Display historical prices
        st.subheader(f"{selected_commodity} Historical Prices")
        
        # Plot historical price chart
        fig = plot_historical_prices(
            df, 
            f"{selected_commodity} ({COMMODITIES[selected_commodity]}) Price History",
            years_to_show
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display commodity stats
        latest_price = df.loc[df['date'] == df['date'].max(), 'close'].values[0]
        earliest_date = df['date'].min()
        latest_date = df['date'].max()
        min_price = df['close'].min()
        max_price = df['close'].max()
        
        # Calculate price change for different time periods
        # 1-day change
        if len(df) >= 2:
            prev_day_price = df.iloc[-2]['close'] if len(df) > 1 else df.iloc[0]['close']
            price_change_1d = latest_price - prev_day_price
            pct_change_1d = (price_change_1d / prev_day_price) * 100
            
            # YTD change
            year_start = pd.Timestamp(latest_date.year, 1, 1)
            ytd_price = df[df['date'] >= year_start].iloc[0]['close'] if not df[df['date'] >= year_start].empty else df.iloc[0]['close']
            price_change_ytd = latest_price - ytd_price
            pct_change_ytd = (price_change_ytd / ytd_price) * 100
        else:
            price_change_1d, pct_change_1d = 0, 0
            price_change_ytd, pct_change_ytd = 0, 0
        
        # Create main header with current price and change
        st.markdown(f"""
        <div style="display: flex; align-items: baseline; margin-bottom: 10px;">
            <h2 style="margin: 0; padding: 0; margin-right: 15px;">${latest_price:.2f}</h2>
            <span style="font-size: 1.2rem; color: {'green' if price_change_1d >= 0 else 'red'};">
                {price_change_1d:+.2f} ({pct_change_1d:+.2f}%)
            </span>
        </div>
        <p style="color: gray; margin-top: 0;">As of {latest_date.strftime('%b %d, %Y')}</p>
        """, unsafe_allow_html=True)
        
        # Create 3 columns for additional stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="border: 1px solid #eee; padding: 10px; border-radius: 5px;">
                <p style="margin: 0; color: gray; font-size: 0.9rem;">YTD Change</p>
                <p style="margin: 0; font-size: 1.1rem; color: {'green' if pct_change_ytd >= 0 else 'red'};">
                    {pct_change_ytd:+.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="border: 1px solid #eee; padding: 10px; border-radius: 5px;">
                <p style="margin: 0; color: gray; font-size: 0.9rem;">52-Week Range</p>
                <p style="margin: 0; font-size: 1.1rem;">
                    ${min_price:.2f} - ${max_price:.2f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            # Get data date range
            data_days = (latest_date - earliest_date).days
            year_str = f"{data_days//365} years" if data_days >= 365 else f"{data_days} days"
            st.markdown(f"""
            <div style="border: 1px solid #eee; padding: 10px; border-radius: 5px;">
                <p style="margin: 0; color: gray; font-size: 0.9rem;">Data Period</p>
                <p style="margin: 0; font-size: 1.1rem;">
                    {year_str}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Forecast section
        st.subheader("Price Forecasting")
        
        # Forecast model selection
        forecast_model = st.selectbox(
            "Select Forecasting Model",
            list(FORECASTING_MODELS.keys()),
            index=0,
            help="Choose the algorithm to use for price forecasting"
        )
        
        # Show model description
        st.info(FORECASTING_MODELS[forecast_model])
        
        # Forecast period selection
        forecast_days = st.slider(
            "Forecast Period (days)",
            min_value=30,
            max_value=365,
            value=180,
            step=30
        )
        
        if st.button("Generate Forecast"):
            # Generate and display forecast
            with st.spinner("Generating forecast..."):
                fig_forecast, fig_components, forecast_stats = generate_forecast(
                    df, selected_commodity, forecast_days, forecast_model
                )
            
            # Display forecast summary
            st.subheader("Forecast Summary")
            
            # Create 3 columns for forecast stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"Current Price ({forecast_stats['last_date'].strftime('%Y-%m-%d')})",
                    f"${forecast_stats['last_price']:.2f}"
                )
            with col2:
                st.metric(
                    f"Forecasted Price ({forecast_stats['forecast_end_date'].strftime('%Y-%m-%d')})", 
                    f"${forecast_stats['forecast_price']:.2f}",
                    f"{forecast_stats['percent_change']:.2f}%"
                )
            with col3:
                st.metric(
                    "Prediction Interval", 
                    f"${forecast_stats['forecast_lower']:.2f} to ${forecast_stats['forecast_upper']:.2f}"
                )
            
            # Display forecast and components plots
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Only show components plot for Prophet model
            if fig_components and forecast_model == "Prophet":
                with st.expander("Show Forecast Components"):
                    st.plotly_chart(fig_components, use_container_width=True)
            
            # Display model-specific notes
            if forecast_model == "Prophet":
                st.caption("Note: Facebook Prophet analyzes historical patterns including seasonality to predict future values.")
            elif forecast_model == "ARIMA":
                st.caption("Note: ARIMA (Autoregressive Integrated Moving Average) is a classical statistical forecasting method that focuses on temporal dependencies.")
            elif forecast_model == "LSTM":
                st.caption("Note: LSTM (Long Short-Term Memory) is a neural network architecture designed to model sequential data and capture long-term dependencies.")

if __name__ == "__main__":
    main() 