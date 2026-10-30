"""Verifikasi bahwa forward pass NumPy identik dengan PyTorch."""
import json
from pathlib import Path

import numpy as np
import torch

from model import CreditMLP
from server import CATEGORICAL, NUMERIC, predict as np_predict
from train import CATEGORICAL as TC, NUMERIC as TN

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "artifacts"

metrics = json.load(open(ARTIFACT / "metrics.json"))
cards = [len(metrics["categories"][c]) for c in TC]

model = CreditMLP(len(TN), cards)
model.load_state_dict(torch.load(ARTIFACT / "model.pt", map_location="cpu"))
model.eval()

rng = np.random.default_rng(0)
max_diff = 0.0

for _ in range(200):
    app = {
        "LIMIT_BAL": float(rng.uniform(10000, 800000)),
        "AGE": float(rng.uniform(21, 75)),
        "PAY_0": float(rng.integers(-1, 8)),
        "PAY_2": float(rng.integers(-1, 8)),
        "PAY_3": float(rng.integers(-1, 8)),
        "PAY_4": float(rng.integers(-1, 8)),
        "PAY_5": float(rng.integers(-1, 8)),
        "PAY_6": float(rng.integers(-1, 8)),
        "BILL_AMT1": float(rng.uniform(0, 500000)),
        "BILL_AMT2": float(rng.uniform(0, 500000)),
        "BILL_AMT3": float(rng.uniform(0, 500000)),
        "BILL_AMT4": float(rng.uniform(0, 500000)),
        "BILL_AMT5": float(rng.uniform(0, 500000)),
        "BILL_AMT6": float(rng.uniform(0, 500000)),
        "PAY_AMT1": float(rng.uniform(0, 200000)),
        "PAY_AMT2": float(rng.uniform(0, 200000)),
        "PAY_AMT3": float(rng.uniform(0, 200000)),
        "PAY_AMT4": float(rng.uniform(0, 200000)),
        "PAY_AMT5": float(rng.uniform(0, 200000)),
        "PAY_AMT6": float(rng.uniform(0, 200000)),
        "SEX": int(rng.integers(1, 3)),
        "EDUCATION": int(rng.integers(1, 5)),
        "MARRIAGE": int(rng.integers(1, 4)),
    }

    p_np = np_predict(app)

    mean = np.array([metrics["mean"][c] for c in TN], dtype=np.float32)
    std = np.array([metrics["std"][c] for c in TN], dtype=np.float32)
    xn = torch.from_numpy(
        (np.array([[app[c] for c in TN]], dtype=np.float32) - mean) / std
    )
    xc = torch.tensor(
        [[{v: i for i, v in enumerate(metrics["categories"][c])}.get(app[c], 0)
          for c in TC]], dtype=torch.long
    )
    with torch.no_grad():
        p_torch = float(torch.sigmoid(model(xn, xc)).item())

    max_diff = max(max_diff, abs(p_np - p_torch))

print(f"Diuji 200 input acak")
print(f"Selisih maksimum NumPy vs PyTorch: {max_diff:.10f}")
assert max_diff < 1e-5, f"Tidak identik! selisih {max_diff}"
print("IDENTIK - server NumPy setara dengan model PyTorch")
