# ⚡ QUICKSTART - DreamerV3 Trading AI

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install torch>=2.0.0 tqdm gymnasium scikit-learn matplotlib
```

### 2. Start Training

```bash
python train/train_dreamer.py
```

This will:
- Load macro data (XAUUSD + DXY + SPX + US10Y)
- Train world model for 100k steps (~3-4 hours on CPU)
- Save checkpoints every 10k steps
- Evaluate on test set

### 3. Monitor Progress

Watch for these metrics:
```
Step 10000:
  World Model Loss: 0.2341  ← Should decrease
  - Recon: 0.1200          ← How well it predicts market
  - Reward: 0.0541         ← How well it predicts profits
  - KL: 0.0600             ← Should stay ~1.0
```

**Good signs:**
- World Model Loss decreasing
- KL Loss between 0.5-2.0
- Test equity improving

### 4. Analyze Results

```bash
python eval/analyze_dreamer.py --checkpoint train/dreamer/dreamer_xauusd_final.pt
```

Outputs:
- Reconstruction quality
- Reward prediction accuracy
- Latent space visualization
- Performance comparison

---

## 📊 What You Built

**DreamerV3** = World Model + Imagination Training

Instead of learning "if X then Y", it learns:
1. **Physics of Markets** (world model)
2. **How to Plan** (imagine 15 steps ahead)
3. **How to Trade** (actor-critic in imagination)

**MCTS** = Lookahead Search

Before each trade:
- Simulates 50 possible futures
- Evaluates each path
- Chooses the best action

**This is "Stockfish for Trading"**

---

## 🎯 Key Files

```
models/
  ├── dreamer_components.py   # Neural networks (Encoder, RSSM, etc.)
  ├── dreamer_agent.py        # Training algorithm
  └── mcts.py                 # Monte Carlo Tree Search

train/
  └── train_dreamer.py        # Training script

eval/
  └── analyze_dreamer.py      # Analysis tools

data/
  └── xauusd_1h_macro.csv     # Gold + Macro data
```

---

## 🔥 Quick Examples

### Load Trained Model

```python
from models.dreamer_agent import DreamerV3Agent

agent = DreamerV3Agent(obs_dim=704, action_dim=2, device='cpu')
agent.load("train/dreamer/dreamer_xauusd_final.pt")

# Use for predictions
action, (h, z) = agent.act(obs, h, z)
```

### Use MCTS for Better Decisions

```python
from models.mcts import DreamerMCTSAgent

# Wrap agent with MCTS
mcts_agent = DreamerMCTSAgent(agent, num_simulations=50)

# This now simulates 50 futures before deciding
action, (h, z) = mcts_agent.act(obs, h, z, use_mcts=True)
```

### Live Trading (Future)

```python
# Will integrate with existing live trading scripts
# live_trade_metaapi.py + DreamerMCTS
```

---

## ⚙️ Tuning Hyperparameters

Edit `train/train_dreamer.py`:

```python
# Faster training (lower quality)
TRAIN_STEPS = 50_000
BATCH_SIZE = 8

# Better performance (slower)
TRAIN_STEPS = 200_000
BATCH_SIZE = 32

# More planning
horizon=30  # imagine 30 steps ahead instead of 15
num_simulations=100  # MCTS searches 100 futures instead of 50
```

---

## 🐛 Troubleshooting

**"Loss is NaN"**
- Reduce learning rate
- Check data for NaN/Inf values

**"Agent only stays flat"**
- Check reward signal (should have positive values)
- Increase random exploration in prefill phase

**"Too slow"**
- Use GPU: `device='cuda'`
- Reduce batch_size
- Reduce horizon

**"Not learning"**
- Increase model size (hidden_dim, embed_dim)
- Train longer (more steps)
- Check data quality

---

## 📚 Learn More

- `GOD_MODE_STATUS.md` - Full implementation details
- `DREAMER_IMPLEMENTATION_GUIDE.md` - Technical guide
- `PROJECT_GOD_MODE.md` - Vision and roadmap

---

## ✅ Success Criteria

**You'll know it's working when:**

1. **World Model Loss < 0.5** (learns market dynamics)
2. **Reward Correlation > 0.5** (predicts profits accurately)
3. **Test Equity > 1.2** (beats buy-and-hold)
4. **Sharpe Ratio > 2.0** (good risk-adjusted returns)

**Current PPO baseline:** ~1.15 equity, Sharpe ~1.5

**DreamerV3 target:** ~1.35 equity, Sharpe ~2.3

**DreamerV3+MCTS target:** ~1.50 equity, Sharpe ~2.8

---

## 🎉 Next Steps

1. ✅ **Train your first model** (you are here)
2. **Analyze results** - understand what it learned
3. **Tune hyperparameters** - improve performance
4. **Add MCTS** - activate "Stockfish mode"
5. **Paper trade** - test on live data
6. **Go live** - deploy for real trading

---

**Ready? Start training:**

```bash
python train/train_dreamer.py
```

**Watch it learn the physics of markets in real-time.**

---

*"We do not guess. We calculate."*
