#!/usr/bin/env python3
"""
Validate if model is ready for live trading
"""
import numpy as np
from stable_baselines3 import PPO
import pandas as pd
from features.make_features import compute_features
from data.load_data import load_ohlc_csv

def validate_model(model_path, data_csv, train_end_date="2022-01-01"):
    """
    Validate model on out-of-sample data (data it never saw during training)
    """

    print("🔍 VALIDATING MODEL FOR LIVE TRADING")
    print("="*60)

    # Load model
    model = PPO.load(model_path)

    # Load data
    df = load_ohlc_csv(data_csv)
    _, features, returns = compute_features(df)

    # Split into train/test
    train_end_idx = np.searchsorted(df["time"].to_numpy(), np.datetime64(train_end_date))

    print(f"\n📊 Data Split:")
    print(f"   Training period: {df['time'].iloc[0]} to {df['time'].iloc[train_end_idx-1]}")
    print(f"   Testing period:  {df['time'].iloc[train_end_idx]} to {df['time'].iloc[-1]}")
    print(f"   Training samples: {train_end_idx}")
    print(f"   Testing samples:  {len(df) - train_end_idx}")

    # Test on OUT-OF-SAMPLE data (data the model never saw)
    test_features = features[train_end_idx:]
    test_returns = returns[train_end_idx:]
    test_df = df.iloc[train_end_idx:].reset_index(drop=True)

    print(f"\n🤖 Running validation on UNSEEN data...")

    window = 64
    equity = 1.0
    position = 0
    trades = []
    equities = [equity]
    positions = [position]
    cost = 0.0001

    for t in range(window, len(test_features)):
        obs_features = test_features[t-window:t].reshape(-1)
        obs = np.concatenate([obs_features, np.array([position], dtype=np.float32)])

        action, _ = model.predict(obs, deterministic=True)
        new_position = 1 if action == 1 else 0

        pnl = position * test_returns[t]
        trade_cost = cost if new_position != position else 0

        if new_position != position:
            trades.append({
                'time': test_df['time'].iloc[t],
                'from': position,
                'to': new_position,
                'price': test_df['close'].iloc[t],
                'equity': equity
            })

        equity *= (1 + pnl - trade_cost)
        position = new_position
        equities.append(equity)
        positions.append(position)

    # Calculate metrics
    equities = np.array(equities)
    positions = np.array(positions)

    total_return = (equities[-1] - 1) * 100
    n_trades = len(trades)

    # Max drawdown
    peak = np.maximum.accumulate(equities)
    drawdown = (equities - peak) / peak
    max_drawdown = np.min(drawdown) * 100

    # Sharpe
    returns_equity = np.diff(np.log(equities))
    sharpe = np.mean(returns_equity) / (np.std(returns_equity) + 1e-8) * np.sqrt(24 * 365)

    # Win rate
    trade_pnls = []
    for i in range(1, len(trades)):
        entry_equity = trades[i-1]['equity']
        exit_equity = trades[i]['equity']
        trade_pnl = (exit_equity - entry_equity) / entry_equity * 100
        trade_pnls.append(trade_pnl)

    if len(trade_pnls) > 0:
        win_rate = np.mean(np.array(trade_pnls) > 0) * 100
        avg_win = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0
        avg_loss = np.mean([p for p in trade_pnls if p < 0]) if any(p < 0 for p in trade_pnls) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    # Results
    print("\n" + "="*60)
    print("📈 OUT-OF-SAMPLE PERFORMANCE (Never Seen Before!)")
    print("="*60)

    print(f"\n💰 Returns:")
    print(f"   Total Return:        {total_return:>10.2f}%")
    print(f"   Final Equity:        {equities[-1]:>10.4f}x")
    print(f"   Max Drawdown:        {max_drawdown:>10.2f}%")
    print(f"   Sharpe Ratio:        {sharpe:>10.2f}")

    print(f"\n📊 Trading Quality:")
    print(f"   Win Rate:            {win_rate:>10.1f}%")
    print(f"   Avg Win:             {avg_win:>10.2f}%")
    print(f"   Avg Loss:            {avg_loss:>10.2f}%")
    print(f"   Profit Factor:       {profit_factor:>10.2f}")
    print(f"   Total Trades:        {n_trades:>10,}")

    print(f"\n⏱️  Position Info:")
    print(f"   Time in Long:        {np.mean(positions == 1) * 100:>10.1f}%")
    print(f"   Time Flat:           {np.mean(positions == 0) * 100:>10.1f}%")

    # Readiness assessment
    print("\n" + "="*60)
    print("🎯 LIVE TRADING READINESS ASSESSMENT")
    print("="*60)

    checks = []

    # Check 1: Positive returns
    if total_return > 0:
        checks.append(("✅", "Positive returns on unseen data", "PASS"))
    else:
        checks.append(("❌", "Negative returns on unseen data", "FAIL"))

    # Check 2: Reasonable Sharpe
    if sharpe > 0.5:
        checks.append(("✅", f"Sharpe ratio {sharpe:.2f} is acceptable", "PASS"))
    else:
        checks.append(("⚠️", f"Sharpe ratio {sharpe:.2f} is low", "WARNING"))

    # Check 3: Drawdown control
    if max_drawdown > -30:
        checks.append(("✅", f"Max drawdown {max_drawdown:.1f}% is controlled", "PASS"))
    else:
        checks.append(("❌", f"Max drawdown {max_drawdown:.1f}% is too high", "FAIL"))

    # Check 4: Win rate
    if win_rate > 40:
        checks.append(("✅", f"Win rate {win_rate:.1f}% is reasonable", "PASS"))
    else:
        checks.append(("⚠️", f"Win rate {win_rate:.1f}% is low", "WARNING"))

    # Check 5: Not overtrading
    trades_per_1000 = n_trades / len(equities) * 1000
    if trades_per_1000 < 200:
        checks.append(("✅", f"Trading frequency {trades_per_1000:.1f} per 1000 is reasonable", "PASS"))
    else:
        checks.append(("⚠️", f"Trading frequency {trades_per_1000:.1f} per 1000 is high", "WARNING"))

    print()
    for emoji, message, status in checks:
        print(f"{emoji} {message}")

    # Final verdict
    fails = sum(1 for _, _, s in checks if s == "FAIL")
    warnings = sum(1 for _, _, s in checks if s == "WARNING")

    print("\n" + "="*60)
    if fails == 0 and warnings <= 1:
        print("✅ VERDICT: Model is READY for live trading")
        print("   Recommendation: Start with SMALL position size (0.01 lots)")
        print("   Monitor closely for first week!")
    elif fails == 0:
        print("⚠️  VERDICT: Model is CAUTIOUSLY READY")
        print("   Recommendation: Start with VERY SMALL size (0.01 lots)")
        print("   Watch performance carefully!")
    else:
        print("❌ VERDICT: Model needs MORE TRAINING")
        print("   Recommendation: Retrain or adjust parameters before going live")

    print("="*60)

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_trades': n_trades,
    }

if __name__ == "__main__":
    validate_model("train/ppo_xauusd_latest.zip", "data/xauusd_1h.csv")
