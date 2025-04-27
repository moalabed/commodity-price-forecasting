import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, timedelta
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# Define a vibrant color palette
VIBRANT_COLORS = [
    '#FF5A5F',  # Coral Red
    '#087E8B',  # Teal
    '#FF9A00',  # Orange
    '#44BBA4',  # Seafoam
    '#3A1772',  # Purple
    '#E63946',  # Bright Red
    '#56C596',  # Mint
    '#6A0572',  # Magenta
    '#3C91E6',  # Bright Blue
    '#F0C808',  # Yellow
]

def load_commodity_data(db_path, commodity_name):
    """Load the historical data for a specific commodity from SQLite database"""
    conn = sqlite3.connect(db_path)
    query = f"SELECT date, close FROM commodities WHERE name = '{commodity_name}' ORDER BY date"
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Convert date to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def prepare_data_for_prophet(df):
    """Prepare the data for Prophet forecasting model with log transform to prevent negative forecasts"""
    # Prophet requires columns named 'ds' for date and 'y' for the value
    df_prophet = df.rename(columns={'date': 'ds', 'close': 'y'})
    
    # Apply log transform to prevent negative price predictions
    df_prophet['y'] = np.log(df_prophet['y'])
    
    return df_prophet

def train_prophet_model(df, forecast_days=180, seasonality_mode='additive'):
    """Train Prophet model and generate forecasts"""
    # Initialize the model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=0.05,
        changepoint_range=0.9
    )
    
    # Fit the model
    model.fit(df)
    
    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=forecast_days)
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Reverse the log transform
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
    
    return model, forecast

def create_prophet_components_plot(model, forecast):
    """Create interactive Plotly components plot for Prophet forecast"""
    # Get components
    components = ['trend', 'yearly', 'weekly']
    valid_components = [c for c in components if c in forecast.columns]
    
    # Create subplots
    fig = make_subplots(
        rows=len(valid_components),
        cols=1,
        subplot_titles=[c.capitalize() for c in valid_components],
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Add component traces
    for i, component in enumerate(valid_components):
        row = i + 1
        color = VIBRANT_COLORS[i % len(VIBRANT_COLORS)]
        
        if component == 'trend':
            # Trend is a line
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=np.exp(forecast[component]),
                    mode='lines',
                    name=component.capitalize(),
                    line=dict(color=color, width=2.5)
                ),
                row=row, col=1
            )
        elif component in ['yearly', 'weekly']:
            # Seasonal components
            # For yearly, we convert to day-of-year
            if component == 'yearly':
                # Group by day of year for yearly seasonality
                grouped = forecast.copy()
                grouped['day_of_year'] = grouped['ds'].dt.strftime('%m-%d')
                avg_by_day = grouped.groupby('day_of_year')[component].mean().reset_index()
                
                # Sort by month and day
                avg_by_day['date'] = pd.to_datetime('2000-' + avg_by_day['day_of_year'])
                avg_by_day = avg_by_day.sort_values('date')
                
                # To create a continuous line, we need to wrap around
                x_values = avg_by_day['date'].tolist()
                y_values = avg_by_day[component].tolist()
                
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode='lines',
                        name='Yearly Seasonality',
                        line=dict(color=color, width=2.5)
                    ),
                    row=row, col=1
                )
                
                # Format x-axis to show month names
                fig.update_xaxes(
                    tickformat='%b',
                    tickmode='array',
                    tickvals=pd.date_range('2000-01-01', '2000-12-31', freq='MS'),
                    row=row, col=1
                )
                
            elif component == 'weekly':
                # Group by day of week for weekly seasonality
                grouped = forecast.copy()
                grouped['day_of_week'] = grouped['ds'].dt.dayofweek
                avg_by_day = grouped.groupby('day_of_week')[component].mean().reset_index()
                avg_by_day = avg_by_day.sort_values('day_of_week')
                
                # Day names
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                fig.add_trace(
                    go.Scatter(
                        x=days,
                        y=avg_by_day[component].tolist(),
                        mode='lines+markers',
                        name='Weekly Seasonality',
                        line=dict(color=color, width=2.5),
                        marker=dict(size=8, color=color)
                    ),
                    row=row, col=1
                )
    
    # Update layout
    fig.update_layout(
        height=300 * len(valid_components),
        title='Forecast Components',
        showlegend=False,
        template='plotly_white',
        plot_bgcolor='rgba(255, 255, 255, 0.9)'
    )
    
    # Update y-axis titles
    for i, component in enumerate(valid_components):
        if component == 'trend':
            y_title = 'Trend (Price)'
        else:
            y_title = f'{component.capitalize()} Effect'
            
        fig.update_yaxes(title_text=y_title, row=i+1, col=1)
    
    return fig

# ARIMA Model Functions

def prepare_data_for_arima(df):
    """Prepare the data for ARIMA forecasting"""
    # Make a copy to avoid modifying the original
    df_arima = df.copy()
    
    # Set date as index for time series analysis
    df_arima.set_index('date', inplace=True)
    
    # Apply log transform to stabilize variance
    df_arima['log_close'] = np.log(df_arima['close'])
    
    return df_arima

def train_arima_model(df, forecast_days=180, order=(5,1,0)):
    """Train ARIMA model and generate forecasts"""
    # Prepare data
    df_arima = prepare_data_for_arima(df)
    
    # Fit ARIMA model
    model = ARIMA(df_arima['log_close'], order=order)
    fitted_model = model.fit()
    
    # Generate forecast
    forecast_steps = forecast_days
    forecast = fitted_model.forecast(steps=forecast_steps)
    
    # Create forecast dataframe with confidence intervals
    last_date = df_arima.index[-1]
    forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps)
    
    # Convert forecasts back from log scale
    forecast_values = np.exp(forecast)
    
    # Calculate confidence intervals (approximation)
    std_dev = fitted_model.params[-1]  # Use the last parameter as an approximation
    confidence_interval = 1.96 * std_dev  # 95% confidence interval
    
    forecast_df = pd.DataFrame({
        'ds': forecast_index,
        'yhat': forecast_values,
        'yhat_lower': np.exp(forecast - confidence_interval),
        'yhat_upper': np.exp(forecast + confidence_interval)
    })
    
    # Add historical data
    historical_df = df.copy()
    historical_df = historical_df.rename(columns={'date': 'ds', 'close': 'yhat'})
    historical_df['yhat_lower'] = historical_df['yhat']
    historical_df['yhat_upper'] = historical_df['yhat']
    
    # Combine historical and forecast data
    result_df = pd.concat([historical_df, forecast_df])
    
    return fitted_model, result_df

# LSTM Model Functions

def prepare_data_for_lstm(df, time_steps=60):
    """Prepare data for LSTM model"""
    # Make a copy to avoid modifying the original
    df_lstm = df.copy()
    
    # Set date as index
    df_lstm.set_index('date', inplace=True)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_lstm[['close']])
    
    # Create sequences for LSTM input
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps, 0])
        y.append(scaled_data[i + time_steps, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split data into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler, scaled_data, time_steps

def train_lstm_model(df, forecast_days=180, time_steps=60):
    """Train LSTM model and generate forecasts"""
    # Prepare data
    X_train, y_train, X_test, y_test, scaler, scaled_data, time_steps = prepare_data_for_lstm(df)
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    
    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=32, 
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)
    
    # Generate forecast
    # First, we need to create inputs for forecasting
    df_lstm = df.copy()
    df_lstm.set_index('date', inplace=True)
    
    # Use the last [time_steps] data points to predict the next point
    last_time_steps = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    
    # Store forecasts
    forecasts = []
    current_prediction = last_time_steps
    
    for _ in range(forecast_days):
        # Predict next value
        next_pred = model.predict(current_prediction)
        forecasts.append(next_pred[0, 0])
        
        # Update prediction input for next iteration
        current_prediction = np.append(current_prediction[:, 1:, :], [[next_pred[0]]], axis=1)
    
    # Inverse transform forecasts to original scale
    forecast_values = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
    
    # Create forecast dataframe
    last_date = df_lstm.index[-1]
    forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
    
    # Estimate uncertainty (simple approach)
    forecast_std = np.std(df_lstm['close']) * 0.1  # 10% of historical std as approximation
    
    forecast_df = pd.DataFrame({
        'ds': forecast_index,
        'yhat': forecast_values.flatten(),
        'yhat_lower': forecast_values.flatten() - 1.96 * forecast_std,
        'yhat_upper': forecast_values.flatten() + 1.96 * forecast_std
    })
    
    # Add historical data
    historical_df = df.copy()
    historical_df = historical_df.rename(columns={'date': 'ds', 'close': 'yhat'})
    historical_df['yhat_lower'] = historical_df['yhat']
    historical_df['yhat_upper'] = historical_df['yhat']
    
    # Combine historical and forecast data
    result_df = pd.concat([historical_df, forecast_df])
    
    # Ensure no negative values in forecast
    result_df['yhat_lower'] = result_df['yhat_lower'].clip(lower=0)
    result_df['yhat'] = result_df['yhat'].clip(lower=0)
    
    return model, result_df
