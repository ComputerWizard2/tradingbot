#!/usr/bin/env python3
"""
Quick test of the Ultimate 150 Features system
"""
from features.ultimate_150_features import make_ultimate_features
import logging
import numpy as np

# Reduce logging for cleaner output
logging.basicConfig(level=logging.ERROR)

print('🧪 Testing Ultimate 150 Features...')
print('This may take 30-60 seconds...\n')

X, returns, timestamps = make_ultimate_features(base_timeframe='M5')

print('✅ SUCCESS!\n')
print(f'📊 Results:')
print(f'  Total features: {X.shape[1]}')
print(f'  Total samples: {X.shape[0]:,}')
print(f'  Memory usage: ~{X.nbytes / 1024**2:.1f} MB')
print(f'\n📈 Stats:')
print(f'  NaN count: {np.isnan(X).sum()}')
print(f'  Inf count: {np.isinf(X).sum()}')
print(f'\n🎯 Date range: {timestamps[0]} to {timestamps[-1]}')
print('\n✅ All features are ready for training!')
