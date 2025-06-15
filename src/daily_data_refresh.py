#!/usr/bin/env python3
"""
Daily Commodity Data Refresh Script
Designed to run as a cron job at midnight to refresh all commodity data from scratch.
This accounts for any corrections or adjustments made by the Yahoo Finance API.
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from commodity_price_data import COMMODITIES, fetch_commodity_data

# Configuration - Use root directory for database and backups to match Streamlit app
project_root = current_dir.parent  # Go up one level from src/ to project root
DB_PATH = project_root / 'commodities.db'
LOG_PATH = project_root / 'data_refresh.log'  # Move log to root as well
START_DATE = '2000-01-01'  # Full historical data refresh

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def backup_database():
    """Create a backup of the current database before refresh"""
    if DB_PATH.exists():
        backup_path = DB_PATH.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
        try:
            import shutil
            shutil.copy2(DB_PATH, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return None
    return None

def cleanup_old_backups(max_backups=7):
    """Keep only the last N backups to save disk space"""
    try:
        # Look for backup files in the project root directory now
        backup_files = sorted(project_root.glob('commodities.backup_*.db'))
        if len(backup_files) > max_backups:
            for old_backup in backup_files[:-max_backups]:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup}")
    except Exception as e:
        logger.error(f"Failed to cleanup old backups: {e}")

def verify_data_integrity():
    """Verify that the refreshed data looks reasonable"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if we have data for all commodities
        cursor.execute("SELECT name, COUNT(*) as count FROM commodities GROUP BY name")
        results = cursor.fetchall()
        
        if len(results) != len(COMMODITIES):
            logger.warning(f"Expected {len(COMMODITIES)} commodities, found {len(results)}")
        
        # Check if we have recent data (within last 7 days)
        cursor.execute("SELECT MAX(date) as latest_date FROM commodities")
        latest_date = cursor.fetchone()[0]
        
        if latest_date:
            latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
            days_old = (datetime.now() - latest_dt).days
            
            if days_old > 7:
                logger.warning(f"Latest data is {days_old} days old")
            else:
                logger.info(f"Data is current (latest: {latest_date})")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Data integrity check failed: {e}")
        return False

def refresh_commodity_data():
    """Main function to refresh commodity data from scratch"""
    logger.info("=== Starting Daily Commodity Data Refresh ===")
    
    try:
        # Create backup before refresh
        backup_path = backup_database()
        
        # Remove existing database for fresh start
        if DB_PATH.exists():
            DB_PATH.unlink()
            logger.info("Removed existing database for fresh refresh")
        
        # Fetch all data from scratch
        logger.info(f"Fetching commodity data from {START_DATE} to present...")
        fetch_commodity_data(START_DATE, str(DB_PATH))
        
        # Verify data integrity
        if verify_data_integrity():
            logger.info("Data refresh completed successfully")
            
            # Cleanup old backups
            cleanup_old_backups()
            
            # Remove today's backup since refresh was successful
            if backup_path and backup_path.exists():
                backup_path.unlink()
                logger.info("Removed backup (refresh successful)")
                
        else:
            logger.error("Data integrity check failed after refresh")
            
            # Restore from backup if available
            if backup_path and backup_path.exists():
                import shutil
                shutil.copy2(backup_path, DB_PATH)
                logger.info("Restored database from backup due to integrity issues")
            
            return False
            
    except Exception as e:
        logger.error(f"Failed to refresh commodity data: {e}")
        
        # Restore from backup if available
        if backup_path and backup_path.exists():
            import shutil
            shutil.copy2(backup_path, DB_PATH)
            logger.info("Restored database from backup due to error")
        
        return False
    
    logger.info("=== Daily Commodity Data Refresh Complete ===")
    return True

if __name__ == "__main__":
    success = refresh_commodity_data()
    sys.exit(0 if success else 1) 