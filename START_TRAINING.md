# 🚀 HOW TO START TRAINING

## Quick Start - 3 Simple Steps

### Step 1: Verify Data is Ready
```bash
ls -lh data/xauusd_1h_macro.csv
# Should see ~6MB file
```

### Step 2: Start Training (Choose One)

**RECOMMENDED - Quick Test First (30 minutes):**
```bash
python train/train_dreamer.py --steps 10000
```

**Full Training (12-24 hours):**
```bash
python train/train_dreamer.py
```

**GPU Accelerated (if you have NVIDIA GPU):**
```bash
python train/train_dreamer.py --device cuda
```

### Step 3: Monitor Progress
Watch the terminal output for:
- Step count increasing
- Reward improving
- Loss decreasing

---

## Expected Timeline

**Quick Test (10k steps):**
- CPU: ~30 minutes
- GPU: ~5 minutes
- Purpose: Verify everything works

**Full Training (100k steps):**
- CPU: ~12-24 hours
- GPU: ~2-4 hours
- M1/M2 Mac: ~4-6 hours

---

## What Happens During Training

1. **Loads Data** (1 minute)
   - Reads data/xauusd_1h_macro.csv
   - Creates 104 multi-timeframe features
   - Splits into train/test

2. **Trains World Model** (hours)
   - DreamerV3 learns market physics
   - Can simulate 10,000 scenarios/second
   - Learns what happens after each action

3. **Trains Policy** (concurrent)
   - Learns which actions make money
   - Uses imagined rollouts (not real trades)
   - Gets smarter over time

4. **Saves Checkpoints** (every 10k steps)
   - train/dreamer/ppo_xauusd_step_10000.pt
   - train/dreamer/ppo_xauusd_step_20000.pt
   - train/dreamer/ppo_xauusd_latest.pt

---

## Monitoring Progress

**Good signs:**
- ✅ Step count increasing steadily
- ✅ Reward going up over time
- ✅ Loss going down
- ✅ No errors in logs

**Warning signs:**
- ⚠️ Process stopped/crashed
- ⚠️ Reward stuck at 0
- ⚠️ Out of memory errors

**Example output:**
```
Step 1000 | Reward: 0.032 | Return: 1.05 | Loss: 0.234
Step 2000 | Reward: 0.045 | Return: 1.12 | Loss: 0.189
Step 3000 | Reward: 0.056 | Return: 1.18 | Loss: 0.156
```

---

## Stopping & Resuming

**To stop:**
```bash
Ctrl+C  # Saves checkpoint before stopping
```

**To resume:**
```bash
python train/train_dreamer.py
# Automatically loads latest checkpoint
```

---

## After Training Completes

### 1. Validate (5 minutes)
```bash
python eval/crisis_validation.py
# Must pass >=75% of tests
```

### 2. Backtest (10 minutes)
```bash
python -c "
from backtest.backtest_engine import RigorousBacktester
from models.dreamer_agent import DreamerV3Agent
import pandas as pd

# Load model
agent = DreamerV3Agent(obs_dim=705, action_dim=2)
agent.load('train/dreamer/ppo_xauusd_latest.pt')

# Load test data
data = pd.read_csv('data/xauusd_1h_macro.csv')
test_data = data.iloc[-5000:]  # Last 5k candles

# Backtest
bt = RigorousBacktester(agent, test_data)
results = bt.run_backtest()
"
```

### 3. Paper Trade (30+ days)
- Test with fake money
- Monitor 24/7
- Verify all systems work

### 4. Go Live (when ready)
- Start with 1-5% capital
- Scale gradually
- Never risk more than you can lose

---

## Troubleshooting

**"CUDA out of memory"**
```bash
python train/train_dreamer.py --batch-size 16
```

**Training very slow**
```bash
# Use smaller test first
python train/train_dreamer.py --steps 5000
```

**"File not found"**
```bash
# Check data exists
ls data/xauusd_1h_macro.csv
```

**Reward not improving**
- Train longer (needs 50k+ steps minimum)
- Wait for at least 20k steps before judging

---

## Ready to Start?

### RECOMMENDED: Test First
```bash
# 30-minute test run
python train/train_dreamer.py --steps 10000
```

### Then: Full Training
```bash
# 12-24 hour full training
python train/train_dreamer.py
```

---

**Good luck! 🚀**

**Remember:**
1. Start with test run (10k steps)
2. Monitor actively
3. Validate after training
4. Backtest thoroughly
5. Paper trade 30+ days
6. Start live small
