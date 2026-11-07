"""
Melatih model prediksi gagal bayar pada dataset Taiwan Credit Card Default.

Jalankan: python train.py
Output  : artifacts/model.pt dan artifacts/metrics.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import CreditMLP

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "artifacts"
ARTIFACT.mkdir(exist_ok=True)

DATA = ROOT / "data" / "taiwan" / "default of credit card clients.xls"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUMERIC = [
    "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
TARGET = "default payment next month"


def load_data():
    df = pd.read_excel(DATA, header=1)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def main():
    df = load_data()
    print(f"Data: {len(df):,} nasabah | tingkat gagal bayar: {df[TARGET].mean():.2%}")

    y = df[TARGET].values.astype(np.float32)

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y
    )

    train, test = df.iloc[idx_train], df.iloc[idx_test]

    mean = train[NUMERIC].mean()
    std = train[NUMERIC].std().replace(0, 1)

    x_num_train = ((train[NUMERIC] - mean) / std).values.astype(np.float32)
    x_num_test = ((test[NUMERIC] - mean) / std).values.astype(np.float32)

    categories = {c: sorted(train[c].unique()) for c in CATEGORICAL}
    x_cat_train = np.stack(
        [train[c].map({v: i for i, v in enumerate(categories[c])}).values
         for c in CATEGORICAL], axis=1
    ).astype(np.int64)
    x_cat_test = np.stack(
        [test[c].map({v: i for i, v in enumerate(categories[c])}).fillna(0).values
         for c in CATEGORICAL], axis=1
    ).astype(np.int64)

    y_train = y[idx_train]
    y_test = y[idx_test]

    ds = TensorDataset(
        torch.from_numpy(x_num_train),
        torch.from_numpy(x_cat_train),
        torch.from_numpy(y_train),
    )
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model = CreditMLP(len(NUMERIC), [len(categories[c]) for c in CATEGORICAL])

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nMelatih model...")
    for epoch in range(1, 31):
        model.train()
        total = 0.0
        for xb_num, xb_cat, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb_num, xb_cat), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)

        if epoch % 5 == 0:
            print(f"  epoch {epoch:2d} | loss {total / len(y_train):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(x_num_test), torch.from_numpy(x_cat_test)
        )
        proba = torch.sigmoid(logits).numpy()

    auc = roc_auc_score(y_test, proba)
    accuracy = ((proba >= 0.5).astype(int) == y_test).mean()

    print(f"\nHasil pada data uji ({len(y_test):,} nasabah):")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Accuracy : {accuracy:.4f}")

    torch.save(model.state_dict(), ARTIFACT / "model.pt")
    json.dump(
        {
            "auc": float(auc),
            "accuracy": float(accuracy),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "base_rate": float(y.mean()),
            "mean": mean.to_dict(),
            "std": std.to_dict(),
            "categories": {k: [int(v) for v in vs] for k, vs in categories.items()},
        },
        open(ARTIFACT / "metrics.json", "w"),
        indent=2,
    )
    print(f"\nModel tersimpan di {ARTIFACT / 'model.pt'}")


if __name__ == "__main__":
    main()
