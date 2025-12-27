#!/usr/bin/env python3
"""
Analyze the trained PPO model performance
"""
import numpy as np
from stable_baselines3 import PPO
import pandas as pd
from features.make_features import compute_features

def analyze_model(model_path, data_csv, window=64, cost=0.0001):
    """Analyze model trading behavior on historical data"""

    # Load model
    print(f"📊 Loading model: {model_path}")
    model = PPO.load(model_path)

    # Load and prepare data
    print(f"📈 Loading data: {data_csv}")
    from data.load_data import load_ohlc_csv
    df = load_ohlc_csv(data_csv)

    # Compute features
    _, features, returns = compute_features(df)

    print(f"✅ Data loaded: {len(df)} candles")
    print(f"   Time range: {df['time'].min()} to {df['time'].max()}")

    # Simulate trading
    print(f"\n🤖 Running bot simulation...")

    equity = 1.0
    position = 0
    trades = []
    equities = [equity]
    positions = [position]

    for t in range(window, len(features)):
        # Get observation
        obs_features = features[t-window:t].reshape(-1)
        obs = np.concatenate([obs_features, np.array([position], dtype=np.float32)])

        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        new_position = 1 if action == 1 else 0

        # Calculate PnL
        pnl = position * returns[t]

        # Trading cost
        if new_position != position:
            trade_cost = cost
            trades.append({
                'time': df['time'].iloc[t],
                'from': position,
                'to': new_position,
                'price': df['close'].iloc[t]
            })
        else:
            trade_cost = 0

        # Update equity
        equity *= (1 + pnl - trade_cost)

        # Update position
        position = new_position

        equities.append(equity)
        positions.append(position)

    # Calculate statistics
    equities = np.array(equities)
    positions = np.array(positions)

    total_return = (equities[-1] - 1) * 100
    n_trades = len(trades)
    pct_time_long = np.mean(positions == 1) * 100
    pct_time_flat = np.mean(positions == 0) * 100

    # Calculate max drawdown
    peak = np.maximum.accumulate(equities)
    drawdown = (equities - peak) / peak
    max_drawdown = np.min(drawdown) * 100

    # Sharpe ratio (annualized, assuming 1h candles)
    returns_equity = np.diff(np.log(equities))
    sharpe = np.mean(returns_equity) / (np.std(returns_equity) + 1e-8) * np.sqrt(24 * 365)

    # Print results
    print("\n" + "="*60)
    print("📊 TRADING PERFORMANCE ANALYSIS")
    print("="*60)

    print(f"\n💰 Performance Metrics:")
    print(f"   Total Return:        {total_return:>10.2f}%")
    print(f"   Final Equity:        {equities[-1]:>10.4f}x")
    print(f"   Max Drawdown:        {max_drawdown:>10.2f}%")
    print(f"   Sharpe Ratio:        {sharpe:>10.2f}")

    print(f"\n📈 Trading Statistics:")
    print(f"   Total Trades:        {n_trades:>10,}")
    print(f"   Total Periods:       {len(equities):>10,}")
    print(f"   Trades per 1000:     {n_trades/len(equities)*1000:>10.1f}")
    print(f"   Avg Hold Time:       {len(equities)/max(n_trades, 1):>10.1f} periods")

    print(f"\n⏱️  Position Distribution:")
    print(f"   Time in LONG:        {pct_time_long:>10.1f}%")
    print(f"   Time FLAT:           {pct_time_flat:>10.1f}%")

    print(f"\n💸 Cost Analysis:")
    print(f"   Total Trade Cost:    {n_trades * cost * 100:>10.4f}%")
    print(f"   Net Return:          {total_return:>10.2f}%")

    # Recent trades
    if len(trades) > 0:
        print(f"\n🔄 Last 10 Trades:")
        for trade in trades[-10:]:
            direction = "FLAT→LONG" if trade['to'] == 1 else "LONG→FLAT"
            print(f"   {trade['time']} | {direction} @ ${trade['price']:.2f}")

    print("\n" + "="*60)

    return {
        'total_return': total_return,
        'n_trades': n_trades,
        'pct_time_long': pct_time_long,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'final_equity': equities[-1],
    }


if __name__ == "__main__":
    import sys

    model_path = "train/ppo_xauusd_latest.zip"

    # Check if data file exists
    data_files = ["data/xauusd_1h.csv", "data/XAUUSD_1h.csv"]
    data_csv = None

    for f in data_files:
        try:
            import os
            if os.path.exists(f):
                data_csv = f
                break
        except:
            pass

    if not data_csv:
        print("❌ Error: Could not find data file!")
        print("   Looking for: data/xauusd_1h.csv")
        print("\n   If you don't have the historical data file,")
        print("   I can only show you the model architecture.")
        sys.exit(1)

    analyze_model(model_path, data_csv)
