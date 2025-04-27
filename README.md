# Commodity Price Forecasting

A collection of tools for fetching, analyzing, visualizing, and forecasting commodity price data.

## Overview

This project provides Python scripts and a Streamlit web application to:

1. Fetch historical commodity price data using yfinance
2. Store this data in a SQLite database for easy access
3. Visualize price trends and relationships
4. Forecast future commodity prices using multiple models:
   - Facebook Prophet
   - ARIMA (Autoregressive Integrated Moving Average)
   - LSTM (Long Short-Term Memory) Neural Networks

## Installation

This project requires Python 3.7+ and uses a conda environment.

```bash
# Create and activate conda environment
conda create -n commodity-price-forecasting python=3.11
conda activate commodity-price-forecasting

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Streamlit Web Application (Recommended)

The easiest way to use this tool is through the Streamlit web application:

```bash
streamlit run src/streamlit_app.py
```

This will open a browser window with the interactive application where you can:
- Select a commodity from the dropdown menu
- View historical price data with customizable time ranges
- Refresh the database with the latest price data
- Choose between three different forecasting models:
  - **Prophet**: Facebook's time series model - good for seasonal data
  - **ARIMA**: Traditional statistical approach for time series
  - **LSTM**: Deep learning approach capable of capturing complex patterns
- Generate price forecasts for different time periods
- Explore forecast components and statistical analysis

### Command Line Scripts

Alternatively, you can use the command-line scripts:

#### 1. Fetch Commodity Data

```bash
python src/commodity_price_data.py
```

This script:
- Fetches historical data for major commodities since 2000 using ETFs/ETNs as proxies
- Stores data in a SQLite database (commodities.db)
- Provides summary statistics for each commodity

#### 2. Visualize Price Trends

```bash
python src/visualize_commodity_prices.py
```

This script:
- Generates multiple plots showing commodity price trends
- Creates visualizations for different commodity groups (energy, metals, agriculture)
- Saves plots in the "plots" directory

#### 3. Forecast Commodity Prices

```bash
python src/forecast_commodity_prices.py
```

This script:
- Uses multiple forecasting methods:
  - Facebook Prophet for time series forecasting with seasonality
  - ARIMA for statistical time series analysis
  - LSTM neural networks for complex pattern recognition
- Generates one-year price forecasts for key commodities
- Creates visualization of forecasts with confidence intervals
- Shows trend and seasonality components
- Provides summary statistics with expected price changes

## Available Commodities

- **OIL**: United States Oil Fund (USO)
- **GOLD**: SPDR Gold Shares (GLD)
- **SILVER**: iShares Silver Trust (SLV)
- **NATURAL_GAS**: United States Natural Gas Fund (UNG)
- **COPPER**: United States Copper Index Fund (CPER)
- **CORN**: Teucrium Corn Fund (CORN)
- **WHEAT**: Teucrium Wheat Fund (WEAT)
- **SOYBEANS**: Teucrium Soybean Fund (SOYB)
- **COFFEE**: iPath Series B Bloomberg Coffee Subindex Total Return ETN (JO)
- **SUGAR**: Teucrium Sugar Fund (CANE)

## Forecasting Models Comparison

The application offers three different forecasting models, each with its own strengths:

1. **Prophet**: Developed by Facebook, this model excels at handling time series with strong seasonal patterns and holiday effects. It's robust to missing data and trend changes, making it suitable for long-term forecasts.

2. **ARIMA**: A classical statistical method that models temporal dependencies. ARIMA works well for stationary time series and can capture autoregressive patterns and moving averages in the data.

3. **LSTM**: A deep learning approach using recurrent neural networks. LSTM can capture complex non-linear relationships and long-term dependencies in the data, making it powerful for datasets with intricate patterns. However, it typically requires more data to perform well.

## Project Structure

```
.
├── src/
│   ├── commodity_price_data.py      # Fetch and store commodity data
│   ├── visualize_commodity_prices.py # Visualize price trends
│   ├── forecast_commodity_prices.py  # Forecast future prices
│   └── streamlit_app.py             # Interactive web application
├── plots/                          # Generated visualizations
├── forecast_plots/                 # Forecast visualizations
├── commodities.db                  # SQLite database
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## Example Output

After running the forecasting script, you'll see output like:

```
GOLD Forecast Summary:
Current price (2025-04-04): $279.72
Forecasted price (2026-04-04): $340.64
Prediction interval: $279.24 to $404.32
Expected change: 21.78%
```

## Limitations

- ETFs and ETNs are used as proxies for commodity prices
- These may not perfectly track the spot price due to various factors
- The forecast models are for educational purposes and should not be the sole basis for investment decisions
- Different models may produce different forecasts for the same data
- LSTM models typically require substantial amounts of data for accurate predictions

## License

MIT 