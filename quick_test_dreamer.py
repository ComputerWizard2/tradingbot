"""
Quick test of DreamerV3 training (1000 steps)
This verifies everything works before starting full training
"""

import os
import sys

# Apply torch compatibility fix
import torch
if not hasattr(torch, 'uint64'):
    torch.uint64 = torch.int64

import numpy as np

from features.make_features import make_features
from models.dreamer_agent import DreamerV3Agent


class SimpleTradingEnv:
    """Minimal trading environment for testing"""
    def __init__(self, features, returns, window=64):
        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)
        self.window = int(window)
        self.T = len(self.r)
        self.reset()

    def reset(self):
        self.t = self.window
        self.pos = 0
        self.equity = 1.0
        return self._get_obs()

    def _get_obs(self):
        w = self.X[self.t - self.window : self.t]
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])
        return obs.astype(np.float32)

    def step(self, action_onehot):
        new_pos = int(np.argmax(action_onehot))
        new_pos = max(0, min(1, new_pos))

        delta = abs(new_pos - self.pos)
        trade_cost = 0.0001 * delta
        pnl = self.pos * self.r[self.t]
        reward = pnl - trade_cost

        self.equity *= (1.0 + reward)
        self.pos = new_pos
        self.t += 1

        done = self.t >= self.T
        obs = np.zeros_like(self._get_obs()) if done else self._get_obs()

        return obs, float(reward), done, {"equity": float(self.equity), "pos": int(self.pos)}


def main():
    print("=" * 70)
    print("QUICK TEST: DreamerV3 Training (1000 steps)")
    print("=" * 70)

    device = 'cpu'
    print(f"Device: {device}\n")

    # Load small subset of data
    print("Loading data...")
    if os.path.exists("data/xauusd_1h_macro.csv"):
        df, X, r = make_features("data/xauusd_1h_macro.csv", window=64)
        print(f"✅ Using macro data: {len(X)} bars\n")
    else:
        print("❌ Macro data not found")
        return

    # Use just first 5000 bars for quick test
    X_test = X[:5000]
    r_test = r[:5000]

    # Create environment
    env = SimpleTradingEnv(X_test, r_test, window=64)
    obs_dim = env._get_obs().shape[0]
    print(f"Observation dim: {obs_dim}")

    # Create agent
    print("\nInitializing DreamerV3...")
    agent = DreamerV3Agent(
        obs_dim=obs_dim,
        action_dim=2,
        device=device,
        embed_dim=128,  # Smaller for quick test
        hidden_dim=256,
        stoch_dim=16,
        num_categories=16,
    )

    # Prefill
    print("\n" + "=" * 70)
    print("Phase 1: Prefill (500 steps)")
    print("=" * 70)

    obs = env.reset()
    for step in range(500):
        action = np.zeros(2, dtype=np.float32)
        action[np.random.randint(0, 2)] = 1.0
        next_obs, reward, done, info = env.step(action)
        agent.replay_buffer.add(obs, action, reward, done)
        obs = next_obs
        if done:
            obs = env.reset()

    print(f"✅ Buffer filled: {len(agent.replay_buffer)} transitions\n")

    # Train
    print("=" * 70)
    print("Phase 2: Training (1000 steps)")
    print("=" * 70)

    obs = env.reset()
    h, z = None, None

    for step in range(1000):
        # Act
        action, (h, z) = agent.act(obs, h, z, deterministic=False)
        next_obs, reward, done, info = env.step(action)
        agent.replay_buffer.add(obs, action, reward, done)

        obs = next_obs
        if done:
            obs = env.reset()
            h, z = None, None

        # Train every 4 steps
        if step % 4 == 0:
            losses = agent.train_step(batch_size=4)

            if losses and step % 200 == 0:
                print(f"\nStep {step}:")
                print(f"  World Model: {losses['world_model_loss']:.4f}")
                print(f"    - Recon: {losses['recon_loss']:.4f}")
                print(f"    - Reward: {losses['reward_loss']:.4f}")
                print(f"    - KL: {losses['kl_loss']:.4f}")
                print(f"  Value: {losses['value_loss']:.4f}")
                print(f"  Policy: {losses['policy_loss']:.4f}")

    print("\n" + "=" * 70)
    print("✅ QUICK TEST COMPLETE!")
    print("=" * 70)
    print("\nEverything is working! Ready for full training.")
    print("\nTo start full training:")
    print("  python train/train_dreamer.py")
    print("\nThis will train for 100k steps (~3-4 hours)")


if __name__ == "__main__":
    main()
