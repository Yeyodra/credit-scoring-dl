"""
Export bobot model PyTorch ke format NumPy agar server tidak perlu PyTorch.

Model: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
Saat inferensi dropout tidak aktif, sehingga forward pass dapat dihitung
dengan operasi matriks biasa.
"""
import json
from pathlib import Path

import numpy as np
import torch

from model import CreditMLP
from train import CATEGORICAL, NUMERIC

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "artifacts"

metrics = json.load(open(ARTIFACT / "metrics.json"))
cards = [len(metrics["categories"][c]) for c in CATEGORICAL]

model = CreditMLP(len(NUMERIC), cards)
model.load_state_dict(torch.load(ARTIFACT / "model.pt", map_location="cpu"))
model.eval()

weights = {}
for name, param in model.state_dict().items():
    weights[name] = param.detach().numpy().astype(np.float32)

np.savez(ARTIFACT / "model_weights.npz", **weights)

print("Bobot tersimpan:")
for k, v in weights.items():
    print(f"  {k:24s} {v.shape}")
print(f"\nTotal: {sum(v.size for v in weights.values()):,} parameter")
