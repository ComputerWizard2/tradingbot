# 🎮 Command Reference - DreamerV3 Trading AI

## 📦 Installation

```bash
# Install core dependencies
pip install torch>=2.0.0 tqdm gymnasium

# Install analysis tools (optional)
pip install scikit-learn matplotlib pandas numpy
```

---

## 🚀 Training

### Basic Training

```bash
# Train DreamerV3 with default settings
python train/train_dreamer.py
```

**Output:**
- Checkpoints saved to: `train/dreamer/`
- Files: `dreamer_xauusd_10k.pt`, `dreamer_xauusd_20k.pt`, ..., `dreamer_xauusd_final.pt`

### Training with GPU

```bash
# Automatically uses GPU if available
python train/train_dreamer.py

# Force CPU
CUDA_VISIBLE_DEVICES="" python train/train_dreamer.py
```

### Monitor Training

```bash
# Watch training progress
tail -f train/dreamer/training.log

# Or just watch the terminal output
python train/train_dreamer.py | tee training_output.log
```

---

## 📊 Analysis

### Analyze Trained Model

```bash
# Analyze final checkpoint
python eval/analyze_dreamer.py --checkpoint train/dreamer/dreamer_xauusd_final.pt

# Analyze specific checkpoint
python eval/analyze_dreamer.py --checkpoint train/dreamer/dreamer_xauusd_50k.pt

# Analyze with more steps
python eval/analyze_dreamer.py --checkpoint train/dreamer/dreamer_xauusd_final.pt --num_steps 2000
```

**Outputs:**
- `eval/dreamer_reward_prediction.png` - Reward prediction quality
- `eval/dreamer_latent_space.png` - Market regime visualization
- Console statistics

### Visualize Latent Space

```bash
# Requires scikit-learn and matplotlib
pip install scikit-learn matplotlib

python eval/analyze_dreamer.py --checkpoint train/dreamer/dreamer_xauusd_final.pt
```

---

## 🧪 Testing Components

### Test DreamerV3 Components

```bash
# Test all neural networks
python models/dreamer_components.py
```

**Expected output:**
```
✅ Encoder: torch.Size([4, 704]) -> torch.Size([4, 256])
✅ RSSM: h=torch.Size([4, 512]), z=torch.Size([4, 1024])
✅ Decoder: torch.Size([4, 1536]) -> torch.Size([4, 704])
...
```

### Test DreamerV3 Agent

```bash
# Test agent and training loop
python models/dreamer_agent.py
```

### Test MCTS

```bash
# Test Monte Carlo Tree Search
python models/mcts.py
```

**Expected output:**
```
✅ MCTS Search Complete!
   Best Action: 1
   Visit Counts: {0: 23, 1: 77}
   Q-Values: {0: -0.0012, 1: 0.0034}
```

---

## 💾 Data Management

### Merge Macro Data

```bash
# If you need to regenerate macro data
python data/merge_macro.py
```

**Creates:** `data/xauusd_1h_macro.csv`

### Check Data

```bash
# View data structure
head -5 data/xauusd_1h_macro.csv

# Count rows
wc -l data/xauusd_1h_macro.csv

# Check for missing values
python -c "import pandas as pd; df = pd.read_csv('data/xauusd_1h_macro.csv'); print(df.isnull().sum())"
```

---

## 🔧 Utilities

### Create Checkpoints Directory

```bash
mkdir -p train/dreamer
mkdir -p eval
```

### Clean Old Checkpoints

```bash
# Keep only final checkpoint
find train/dreamer -name "*.pt" ! -name "dreamer_xauusd_final.pt" -delete

# Keep only latest 3 checkpoints
ls -t train/dreamer/*.pt | tail -n +4 | xargs rm
```

### Check Model Size

```bash
# Size of checkpoint
ls -lh train/dreamer/dreamer_xauusd_final.pt

# Count parameters
python -c "
import torch
ckpt = torch.load('train/dreamer/dreamer_xauusd_final.pt', map_location='cpu')
total = sum(p.numel() for p in ckpt['encoder'].values())
print(f'Encoder params: {total:,}')
"
```

---

## 🐍 Python Usage

### Load and Use Model

```python
from models.dreamer_agent import DreamerV3Agent
import numpy as np

# Load agent
agent = DreamerV3Agent(obs_dim=704, action_dim=2, device='cpu')
agent.load("train/dreamer/dreamer_xauusd_final.pt")

# Make prediction
obs = np.random.randn(704).astype(np.float32)
action, (h, z) = agent.act(obs, h=None, z=None)

print(f"Action: {np.argmax(action)}")  # 0=flat, 1=long
```

### Use MCTS for Planning

```python
from models.dreamer_agent import DreamerV3Agent
from models.mcts import DreamerMCTSAgent

# Load agent
base_agent = DreamerV3Agent(obs_dim=704, action_dim=2, device='cpu')
base_agent.load("train/dreamer/dreamer_xauusd_final.pt")

# Wrap with MCTS
mcts_agent = DreamerMCTSAgent(base_agent, num_simulations=50)

# Use for trading (500ms thinking time)
obs = get_market_observation()
action, (h, z) = mcts_agent.act(obs, h=None, z=None, use_mcts=True)
```

### Evaluate on Custom Data

```python
from features.make_features import make_features
from train.train_dreamer import TradingEnvironment
from models.dreamer_agent import DreamerV3Agent
import numpy as np

# Load your data
df, X, r = make_features("data/xauusd_1h_macro.csv", window=64)

# Create environment
env = TradingEnvironment(X, r, window=64)

# Load agent
agent = DreamerV3Agent(obs_dim=X.shape[1] * 64 + 1, action_dim=2)
agent.load("train/dreamer/dreamer_xauusd_final.pt")

# Run evaluation
obs = env.reset()
h, z = None, None
equity_curve = []

while True:
    action, (h, z) = agent.act(obs, h, z, deterministic=True)
    obs, reward, done, info = env.step(action)
    equity_curve.append(info['equity'])
    if done:
        break

print(f"Final Equity: {equity_curve[-1]:.4f}")
print(f"Return: {(equity_curve[-1] - 1) * 100:.2f}%")
```

---

## 🔍 Debugging

### Enable Verbose Logging

```python
# In train_dreamer.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Gradient Flow

```python
# After backward pass
for name, param in agent.encoder.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

### Monitor GPU Memory

```bash
# While training
watch -n 1 nvidia-smi
```

### Profile Training

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Training code here
agent.train_step(batch_size=16)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

---

## 📝 Configuration

### Quick Settings

Edit `train/train_dreamer.py`:

```python
# Fast training (for testing)
TRAIN_STEPS = 10_000
BATCH_SIZE = 4
SAVE_EVERY = 2_000

# Normal training
TRAIN_STEPS = 100_000
BATCH_SIZE = 16
SAVE_EVERY = 10_000

# High quality (slow)
TRAIN_STEPS = 500_000
BATCH_SIZE = 32
SAVE_EVERY = 25_000
```

### Model Size

```python
# Small model (fast, less capacity)
embed_dim=128
hidden_dim=256
stoch_dim=16

# Medium model (default)
embed_dim=256
hidden_dim=512
stoch_dim=32

# Large model (slow, more capacity)
embed_dim=512
hidden_dim=1024
stoch_dim=64
```

---

## 🎯 Common Workflows

### 1. Quick Test Run

```bash
# Fast training for testing
python train/train_dreamer.py  # Edit TRAIN_STEPS to 10000 first

# Check if it works
python models/dreamer_components.py
python models/mcts.py
```

### 2. Full Training

```bash
# Start training (3-4 hours)
nohup python train/train_dreamer.py > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Analyze when done
python eval/analyze_dreamer.py --checkpoint train/dreamer/dreamer_xauusd_final.pt
```

### 3. Hyperparameter Tuning

```bash
# Train with different settings
python train/train_dreamer.py  # Edit hyperparameters

# Compare checkpoints
python eval/analyze_dreamer.py --checkpoint train/dreamer/dreamer_xauusd_v1.pt
python eval/analyze_dreamer.py --checkpoint train/dreamer/dreamer_xauusd_v2.pt
```

### 4. Production Deployment

```python
# 1. Train best model
# 2. Save with specific name
agent.save("models/production_v1.pt")

# 3. Load in production
from models.mcts import DreamerMCTSAgent
from models.dreamer_agent import DreamerV3Agent

agent = DreamerV3Agent(obs_dim=704, action_dim=2)
agent.load("models/production_v1.pt")

mcts_agent = DreamerMCTSAgent(agent, num_simulations=50)

# 4. Use in live trading
# (integrate with live_trade_metaapi.py)
```

---

## 🆘 Getting Help

### Read Documentation

```bash
cat QUICKSTART.md
cat GOD_MODE_STATUS.md
cat DREAMER_IMPLEMENTATION_GUIDE.md
```

### Check Implementation

```bash
# View component implementation
less models/dreamer_components.py

# View training loop
less models/dreamer_agent.py

# View MCTS
less models/mcts.py
```

### File Locations

```
models/dreamer_components.py    # Line 41: Encoder
models/dreamer_components.py    # Line 58: RSSM
models/dreamer_components.py    # Line 178: Decoder
models/dreamer_agent.py         # Line 100: DreamerV3Agent
models/mcts.py                  # Line 100: MCTS
train/train_dreamer.py          # Line 25: main()
```

---

**Quick Reference:**

- Train: `python train/train_dreamer.py`
- Analyze: `python eval/analyze_dreamer.py --checkpoint <path>`
- Test: `python models/dreamer_components.py`
- Docs: Read `QUICKSTART.md` or `GOD_MODE_STATUS.md`
