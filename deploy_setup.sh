#!/bin/bash
# Deployment setup script for trading bot
# Run this on your server after SSH connection

echo "🚀 Setting up trading bot on server..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.12
sudo apt-get install -y python3.12 python3.12-venv python3-pip

# Install system dependencies
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev

# Create project directory
mkdir -p ~/trading-bot
cd ~/trading-bot

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install numpy pandas stable-baselines3 metaapi-cloud-sdk scikit-learn ta-lib

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your code: scp -r drl-trading/ user@server:~/trading-bot/"
echo "2. Create a systemd service to run the bot automatically"
