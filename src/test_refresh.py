#!/usr/bin/env python3
"""
Test script to manually run the daily refresh process
Use this to test the refresh functionality before setting up the cron job
"""

import sys
from pathlib import Path

# Add the src directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from daily_data_refresh import refresh_commodity_data

if __name__ == "__main__":
    print("Testing daily commodity data refresh...")
    success = refresh_commodity_data()
    
    if success:
        print("✅ Test refresh completed successfully!")
    else:
        print("❌ Test refresh failed!")
    
    sys.exit(0 if success else 1) 