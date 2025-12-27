"""
Workaround for torch/transformers compatibility issue
Sets torch.uint64 attribute to prevent AttributeError
"""
import sys

# Monkey-patch torch before any other imports
import torch
if not hasattr(torch, 'uint64'):
    torch.uint64 = torch.int64  # Fallback

print("✅ Torch compatibility fix applied")
