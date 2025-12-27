#!/bin/bash
# Quick deploy script - Run this on your Mac to upload code to Google Cloud

echo "📦 Creating deployment package..."
cd ~/Desktop/trading/drl-trading

# Create tarball with all necessary files
tar -czf trading-bot.tar.gz \
  live_trade_metaapi.py \
  features/ \
  train/ppo_xauusd_latest.zip

echo "✅ Package created!"
echo ""
echo "📤 Uploading to Google Cloud..."

# Upload to server
gcloud compute scp trading-bot.tar.gz trading-bot:~/trading-bot/ --zone=us-central1-a

if [ $? -eq 0 ]; then
  echo "✅ Upload successful!"
  echo ""
  echo "To deploy on server, run:"
  echo "  gcloud compute ssh trading-bot --zone=us-central1-a"
  echo ""
  echo "Then on the server:"
  echo "  cd ~/trading-bot"
  echo "  tar -xzf trading-bot.tar.gz"
  echo "  sudo systemctl restart trading-bot"
else
  echo "❌ Upload failed. Make sure you:"
  echo "  1. Have gcloud installed: brew install --cask google-cloud-sdk"
  echo "  2. Are logged in: gcloud auth login"
  echo "  3. Have created the VM named 'trading-bot'"
fi
