# Commodity Price Forecasting

A sophisticated web application for analyzing, visualizing, and forecasting commodity price data using advanced machine learning models.

## 🌐 Live Application

**Access the deployed application:** https://commodity-price-forecasting-bdc3uifcwmjsmijfhy3mh5.streamlit.app/

## Overview

This project provides a comprehensive Streamlit web application that enables users to:

1. **Interactive Data Visualization**: View historical commodity price data with customizable time ranges and Yahoo Finance-style charts
2. **Correlation Analysis**: Explore relationships between different commodities through interactive heatmaps
3. **Multi-Model Forecasting**: Generate price forecasts using three different approaches:
   - **Facebook Prophet**: Handles seasonality and trends effectively
   - **ARIMA**: Classical statistical time series analysis
   - **LSTM**: Deep learning for complex pattern recognition
4. **Automated Data Management**: Daily automatic updates ensure fresh data without manual intervention
5. **Real-time Analytics**: Current price tracking with percentage changes and 52-week ranges

## Features

### 📈 Advanced Visualization
- **Yahoo Finance-style Charts**: Interactive price charts with range selectors and zoom functionality
- **Correlation Heatmaps**: Understand how commodities move relative to each other
- **Multi-commodity Views**: Compare normalized and raw prices across commodity groups
- **Customizable Themes**: Multiple visualization themes for different preferences

### 🤖 Machine Learning Forecasting
- **Prophet Model**: Facebook's robust time series forecasting with seasonality detection
- **ARIMA Model**: Statistical approach for autoregressive patterns
- **LSTM Neural Networks**: Deep learning for capturing complex non-linear relationships
- **Confidence Intervals**: Statistical ranges for forecast reliability
- **Component Analysis**: Breakdown of trends, seasonality, and residuals (Prophet)

### 🔄 Automated Data Pipeline
- **Daily Refresh**: GitHub Actions automatically update data at midnight UTC
- **Data Integrity Checks**: Automated verification and backup systems
- **Historical Coverage**: Complete price history from 2000 to present
- **Real-time Updates**: Latest market data integrated daily

## Installation & Local Development

### Prerequisites
- Python 3.11+
- Conda environment manager

### Setup

```bash
# Clone the repository
git clone <repository-url> #use https or ssh
cd commodity-price-forecasting

# Create and activate conda environment
conda create -n commodity-price-forecasting python=3.11
conda activate commodity-price-forecasting

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Activate the conda environment
conda activate commodity-price-forecasting

# Run the Streamlit application
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Initial Data Setup

If running locally for the first time, the app will prompt you to fetch initial data:

```bash
# Or manually fetch data using the core module
python -c "from src import fetch_commodity_data; fetch_commodity_data('2000-01-01', 'commodities.db')"
```

## Available Commodities

The application tracks 10 major commodities across three categories:

### Energy
- **OIL**: United States Oil Fund (USO)
- **NATURAL_GAS**: United States Natural Gas Fund (UNG)

### Metals
- **GOLD**: SPDR Gold Shares (GLD)
- **SILVER**: iShares Silver Trust (SLV)
- **COPPER**: United States Copper Index Fund (CPER)

### Agriculture
- **CORN**: Teucrium Corn Fund (CORN)
- **WHEAT**: Teucrium Wheat Fund (WEAT)
- **SOYBEANS**: Teucrium Soybean Fund (SOYB)
- **COFFEE**: iPath Series B Bloomberg Coffee Subindex Total Return ETN (JO)
- **SUGAR**: Teucrium Sugar Fund (CANE)

*Note: ETFs and ETNs are used as proxies for commodity spot prices*

## Architecture & Project Structure

```
.
├── streamlit_app.py              # Main Streamlit web application
├── src/                          # Core functionality modules
│   ├── __init__.py              # Module exports
│   ├── commodity_price_data.py  # Data fetching and storage
│   └── forecast_commodity_prices.py # ML forecasting models
├── tasks/                        # Automation scripts
│   ├── daily_data_refresh.py    # Daily data update script
│   └── __init__.py
├── .github/workflows/           # GitHub Actions automation
│   └── daily-data-refresh.yml  # Daily data refresh workflow
├── commodities.db              # SQLite database (auto-updated)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Forecasting Models Deep Dive

### 1. Facebook Prophet
- **Best for**: Long-term forecasts with seasonal patterns
- **Strengths**: Handles missing data, holidays, and trend changes
- **Use cases**: Annual forecasting, seasonal commodity analysis
- **Components**: Trend, seasonality, and holiday effects visualization

### 2. ARIMA (AutoRegressive Integrated Moving Average)
- **Best for**: Short to medium-term forecasts with clear temporal patterns
- **Strengths**: Statistical rigor, interpretable parameters
- **Use cases**: Technical analysis, stationary time series
- **Components**: Autoregressive, differencing, and moving average terms

### 3. LSTM (Long Short-Term Memory)
- **Best for**: Complex pattern recognition in large datasets
- **Strengths**: Captures non-linear relationships and long-term dependencies
- **Use cases**: High-frequency data, complex market dynamics
- **Requirements**: Substantial historical data for optimal performance

## Automated Data Management

### Daily Refresh Process
1. **Scheduled Execution**: GitHub Actions runs daily at midnight UTC
2. **Data Validation**: Integrity checks ensure data quality
3. **Backup Management**: Automatic backups with 7-day retention
4. **Error Recovery**: Automatic restoration from backups if issues occur
5. **Deployment Update**: Streamlit Cloud automatically uses refreshed data

### Manual Data Refresh
For immediate updates or troubleshooting:

```bash
# Run the refresh script manually
python tasks/daily_data_refresh.py
```

## Usage Guide

### Web Application Interface

1. **Commodity Selection**: Choose individual commodities or view all
2. **Time Range Controls**: Filter data by various time periods
3. **Visualization Options**: Switch between normalized and raw price views
4. **Correlation Analysis**: Explore relationships between commodities
5. **Forecasting**: Select models and forecast periods for predictions

### Key Metrics Displayed

- **Current Price**: Latest market value with daily change
- **YTD Performance**: Year-to-date percentage change
- **52-Week Range**: Annual high and low prices
- **Forecast Intervals**: Prediction ranges with confidence levels
- **Historical Performance**: Long-term price trends and patterns

## Performance & Limitations

### Performance Characteristics
- **Data Retrieval**: < 1 second for most queries
- **Forecast Generation**: 5-30 seconds depending on model and period
- **Auto-updates**: Daily refresh maintains current data
- **Scalability**: Designed for 100+ concurrent users

### Known Limitations
- **ETF Tracking**: Proxy instruments may not perfectly match spot prices
- **Educational Purpose**: Forecasts are for analysis, not investment advice
- **Data Dependencies**: Relies on Yahoo Finance API availability
- **Model Accuracy**: Different models may produce varying forecasts
- **LSTM Requirements**: Neural networks need substantial historical data



## License

MIT License - see LICENSE file for details

## Support & Contact

For issues, feature requests, or contributions, please use the project's issue tracker or submit pull requests through the repository.

---

*Last updated: 2025 - Reflecting major architectural changes including Streamlit deployment, automated data pipeline, and enhanced forecasting capabilities.* 