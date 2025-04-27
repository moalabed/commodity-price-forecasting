import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime
import os

# Define commodity ticker symbols
# These are commodity ETFs/ETNs that track major commodities
COMMODITIES = {
    'OIL': 'USO',     # United States Oil Fund
    'GOLD': 'GLD',    # SPDR Gold Shares
    'SILVER': 'SLV',  # iShares Silver Trust
    'NATURAL_GAS': 'UNG',  # United States Natural Gas Fund
    'COPPER': 'CPER', # United States Copper Index Fund
    'CORN': 'CORN',   # Teucrium Corn Fund
    'WHEAT': 'WEAT',  # Teucrium Wheat Fund
    'SOYBEANS': 'SOYB',  # Teucrium Soybean Fund
    'COFFEE': 'JO',   # iPath Series B Bloomberg Coffee Subindex Total Return ETN
    'SUGAR': 'CANE',  # Teucrium Sugar Fund
}

def create_database(db_path):
    """Create SQLite database and commodities table if it doesn't exist"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create commodities table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS commodities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adj_close REAL,
        volume INTEGER,
        UNIQUE(symbol, date)
    )
    ''')
    
    conn.commit()
    return conn

def fetch_commodity_data(start_date, db_path):
    """Fetch commodity price data and store in SQLite database"""
    conn = create_database(db_path)
    cursor = conn.cursor()
    
    print(f"Fetching commodity data since {start_date}...")
    
    for commodity_name, symbol in COMMODITIES.items():
        try:
            print(f"Processing {commodity_name} ({symbol})...")
            
            # Use Ticker object to get historical data
            commodity = yf.Ticker(symbol)
            data = commodity.history(start=start_date)
            
            if data.empty:
                print(f"No data found for {commodity_name} ({symbol})")
                continue
            
            print(f"Downloaded {len(data)} rows")
            
            # Prepare data for insertion
            records = []
            for date, row in data.iterrows():
                records.append((
                    commodity_name,
                    symbol,
                    date.strftime('%Y-%m-%d'),
                    float(row['Open']) if 'Open' in row and not pd.isna(row['Open']) else None,
                    float(row['High']) if 'High' in row and not pd.isna(row['High']) else None,
                    float(row['Low']) if 'Low' in row and not pd.isna(row['Low']) else None,
                    float(row['Close']) if 'Close' in row and not pd.isna(row['Close']) else None,
                    float(row['Close']) if 'Close' in row and not pd.isna(row['Close']) else None,  # Using Close as Adj Close
                    int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0
                ))
            
            # Insert data into database
            cursor.executemany(
                '''
                INSERT OR REPLACE INTO commodities (name, symbol, date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                records
            )
            conn.commit()
            print(f"Successfully stored {len(records)} records for {commodity_name}")
            
        except Exception as e:
            print(f"Error processing {commodity_name} ({symbol}): {str(e)}")
    
    conn.close()

def get_commodity_stats(db_path):
    """Print statistics about the commodity data in the database"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    # Get count of records for each commodity
    cursor.execute('''
    SELECT name, symbol, COUNT(*) as count, 
           MIN(date) as first_date, MAX(date) as last_date,
           MIN(close) as min_price, MAX(close) as max_price,
           AVG(close) as avg_price
    FROM commodities
    GROUP BY name, symbol
    ORDER BY name
    ''')
    
    stats = cursor.fetchall()
    
    print("\nCommodity Data Statistics:")
    print("-" * 80)
    print(f"{'Commodity':<15} {'Symbol':<8} {'Records':<10} {'Date Range':<25} {'Price Range':<20} {'Avg Price':<10}")
    print("-" * 80)
    
    for row in stats:
        print(f"{row['name']:<15} {row['symbol']:<8} {row['count']:<10} {row['first_date']} - {row['last_date']} " +
              f"${row['min_price']:.2f} - ${row['max_price']:.2f} ${row['avg_price']:.2f}")
    
    conn.close()

def main():
    # Configuration
    start_date = '2000-01-01'
    db_path = 'commodities.db'
    
    # Check if database exists
    db_exists = os.path.exists(db_path)
    
    if db_exists:
        user_input = input(f"Database {db_path} already exists. Overwrite? (y/n): ")
        if user_input.lower() == 'y':
            os.remove(db_path)
            print(f"Removed existing database file: {db_path}")
        else:
            print("Using existing database.")
    
    # Fetch data if database doesn't exist or user chose to overwrite
    if not db_exists or user_input.lower() == 'y':
        fetch_commodity_data(start_date, db_path)
    
    # Show statistics
    get_commodity_stats(db_path)
    
    print("\nData can be accessed using SQLite commands or directly with pandas.")
    print("Example:")
    print("import pandas as pd")
    print("import sqlite3")
    print("conn = sqlite3.connect('commodities.db')")
    print("df = pd.read_sql('SELECT * FROM commodities WHERE name=\"OIL\"', conn)")

if __name__ == "__main__":
    main() 