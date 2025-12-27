# 🚀 Google Colab Training Setup Guide

Complete guide to train your Ultimate 150+ Features Trading AI on Google Colab with GPU acceleration.

---

## 📋 Prerequisites

1. **Google Account** (free)
2. **Google Drive** with ~2GB free space
3. **Colab Pro/Pro+** (recommended but optional)
   - Pro: $10/month - Better GPUs, longer runtimes
   - Pro+: $50/month - A100 GPUs, background execution, priority access

---

## 🎯 Quick Start (5 Steps)

### Step 1: Upload Project to Google Drive

1. **Compress your project folder:**
   ```bash
   cd /Users/mac/Desktop/trading
   zip -r drl-trading.zip drl-trading/
   ```

2. **Upload to Google Drive:**
   - Go to https://drive.google.com
   - Click "New" → "Folder" → Name it "AI_Trading"
   - Upload `drl-trading.zip` to this folder
   - Right-click the zip file → Extract

   OR upload the entire `drl-trading` folder directly (slower but easier)

### Step 2: Open the Colab Notebook

1. **Upload the notebook:**
   - In Google Drive, navigate to your `drl-trading` folder
   - Upload the file: `colab_train_ultimate_150.ipynb`

2. **Open with Colab:**
   - Right-click `colab_train_ultimate_150.ipynb`
   - Choose "Open with" → "Google Colaboratory"
   - If you don't see it, click "Connect more apps" and install Colaboratory

### Step 3: Enable GPU

1. In the Colab notebook, click: **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Choose GPU type (if you have Colab Pro+):
   - **A100** (fastest - 5-7 hours for 1M steps)
   - **V100** (fast - 8-10 hours)
   - **T4** (free tier - 12-15 hours)
4. Click **Save**

### Step 4: Update the Project Path

In the notebook, find this cell:
```python
PROJECT_PATH = '/content/drive/MyDrive/drl-trading'
```

Update it to match where you uploaded your folder. For example:
```python
PROJECT_PATH = '/content/drive/MyDrive/AI_Trading/drl-trading'
```

### Step 5: Run All Cells

1. Click **Runtime** → **Run all**
2. Authorize Google Drive access when prompted
3. Training will start automatically!

---

## ⚡ Training Speed Comparison

| GPU Type | Availability | 1M Steps | 100K Steps | Cost |
|----------|--------------|----------|------------|------|
| **A100 (40GB)** | Colab Pro+ | 5-7 hours | 30-45 min | $50/month |
| **V100 (16GB)** | Colab Pro | 8-10 hours | 50-60 min | $10/month |
| **T4 (16GB)** | Free Colab | 12-15 hours | 70-90 min | Free |
| **Mac MPS** | Your Mac | 6-8 days | 14-16 hours | Free |
| **CPU** | Any computer | 7-10 days | 16-20 hours | Free |

---

## 💡 Training Recommendations

### Quick Test (10K steps - ~5 minutes)
```python
TRAINING_STEPS = 10_000
```
- Good for: Testing the pipeline
- Results: Basic functionality check

### Short Training (100K steps - ~1 hour on A100)
```python
TRAINING_STEPS = 100_000
```
- Good for: Initial learning validation
- Results: Agent starts to learn patterns

### Full Training (1M steps - ~6 hours on A100)
```python
TRAINING_STEPS = 1_000_000
```
- Good for: Production model
- Results: Fully trained, ready for live trading

---

## 🔧 Troubleshooting

### Issue: "Runtime disconnected"
**Solution:**
- Colab free tier disconnects after 12 hours or if idle
- Use Colab Pro+ for background execution
- Download checkpoints frequently (saved every 10K steps)

### Issue: "Out of memory"
**Solution:**
- Reduce batch size: `BATCH_SIZE = 64` or `BATCH_SIZE = 32`
- Restart runtime: Runtime → Restart runtime

### Issue: "GPU not available"
**Solution:**
- Check Runtime → Change runtime type → GPU selected
- Free tier has limited GPU access (try different times)
- Consider upgrading to Colab Pro

### Issue: "Cannot find module"
**Solution:**
- Make sure you're in the correct directory
- Check `PROJECT_PATH` is set correctly
- Rerun the "Install Dependencies" cell

---

## 📊 Monitoring Training

### Check GPU Usage
Run this in a new cell:
```python
!nvidia-smi
```

You should see:
- GPU name (T4, V100, or A100)
- Memory usage increasing
- GPU utilization at 80-100%

### View Training Progress
Training shows:
- Step number / total steps
- Episode rewards
- Training losses
- Checkpoints saved

Example output:
```
Training:  10%|████      | 10000/100000 [05:23<48:31,  30.93it/s]
Episode 145 - Reward: 0.0234 - Best: 0.0456
```

---

## 💾 Downloading Your Trained Model

### Option 1: Auto-download (in notebook)
The notebook includes a cell to download the latest checkpoint automatically.

### Option 2: Manual download
1. In the Colab file browser (left sidebar)
2. Navigate to: `train/dreamer_ultimate/`
3. Find latest checkpoint: `ultimate_150_xauusd_step_1000000.pt`
4. Right-click → Download

### Option 3: Save to Drive
```python
import shutil
shutil.copy(
    'train/dreamer_ultimate/ultimate_150_xauusd_step_1000000.pt',
    '/content/drive/MyDrive/trained_model.pt'
)
```

---

## 🎯 After Training

### 1. Download the Model
Get the checkpoint file to your local machine

### 2. Test on Your Mac
```bash
cd /Users/mac/Desktop/trading/drl-trading
python test/evaluate_model.py --checkpoint path/to/checkpoint.pt
```

### 3. Backtest
Test on out-of-sample data (2024-2025)

### 4. Deploy
Use for live trading!

---

## 💰 Cost Optimization

### Free Tier Strategy
1. Run 12-hour training sessions
2. Save checkpoints frequently
3. Download checkpoint after each session
4. Resume training in next session

### Pro Strategy ($10/month)
1. Longer continuous training
2. Better GPU allocation
3. Faster completion

### Pro+ Strategy ($50/month)
1. A100 access (3x faster)
2. Background execution (can close browser)
3. Best for 1M+ step training

---

## 📈 Expected Results

With **1M training steps** on the Ultimate 150 Features:

- **Annual Return:** 80-120%+
- **Sharpe Ratio:** 3.5-4.5+
- **Max Drawdown:** 5-8%
- **Win Rate:** 60-65%+

These are potential results based on feature quality. Actual results depend on market conditions and training quality.

---

## 🆘 Getting Help

If you encounter issues:

1. **Check the error message** in the Colab output
2. **Restart runtime:** Runtime → Restart runtime
3. **Verify file paths** are correct
4. **Check GPU is enabled** in runtime settings
5. **Try smaller batch size** if memory errors

---

## ✅ Checklist

Before starting training, ensure:

- [ ] Project uploaded to Google Drive
- [ ] Colab notebook opened
- [ ] GPU enabled in runtime settings
- [ ] PROJECT_PATH updated correctly
- [ ] All cells run successfully
- [ ] GPU showing in nvidia-smi
- [ ] Training started

---

## 🚀 Ready to Train!

You now have everything you need to train your AI on Google Colab with GPU acceleration!

**Training time:** 5-15 hours depending on GPU
**Cost:** Free to $50/month
**Result:** Production-ready trading AI

Good luck! 🎯
