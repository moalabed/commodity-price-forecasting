#!/bin/bash

# Script to setup daily commodity data refresh cron job
# Run this script once to configure the automated daily updates

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$PROJECT_DIR/src/daily_data_refresh.py"
CONDA_ENV="commodity-price-forecasting"

# Make the Python script executable
chmod +x "$PYTHON_SCRIPT"

# Create the cron command
# This will run at midnight (00:00) every day
# Updated to use root directory for cron logs to match the database and log location changes
CRON_COMMAND="0 0 * * * /home/$(whoami)/miniconda3/envs/$CONDA_ENV/bin/python $PYTHON_SCRIPT >> $PROJECT_DIR/cron.log 2>&1"

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_COMMAND") | crontab -

echo "Cron job setup complete!"
echo "Daily commodity data refresh will run at midnight."
echo "Command added: $CRON_COMMAND"
echo ""
echo "To verify the cron job was added, run: crontab -l"
echo "To remove the cron job later, run: crontab -e"
echo ""
echo "Logs will be written to:"
echo "- $PROJECT_DIR/data_refresh.log (detailed logs)"
echo "- $PROJECT_DIR/cron.log (cron execution logs)" 