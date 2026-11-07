"""
Melatih model prediksi gagal bayar pada dataset Taiwan Credit Card Default.

Jalankan:  python train.py
Output  :  artifacts/model.pt  dan  artifacts/metrics.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from model import CreditMLP

warnings.filterwarnings("ignore")  # redam warning deprecation engine xlrd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARTIFACT = HERE / "artifacts"
DATA = ROOT / "data" / "taiwan" / "default of credit card clients.xls"
SEED = 42

# Semua 23 fitur dipakai sebagai angka. Fitur kategorikal sudah berupa kode
# integer (SEX 1/2, EDUCATION 1..4, MARRIAGE 1..3) sehingga tidak perlu embedding.
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET = "default payment next month"


def load_data():
    df = pd.read_excel(DATA, header=1)
    # Rapikan kategori yang tidak terdefinisi di dokumentasi.
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ARTIFACT.mkdir(exist_ok=True)

    df = load_data()
    print(f"Data: {len(df):,} nasabah | gagal bayar: {df[TARGET].mean():.2%}")

    y = df[TARGET].values.astype(np.float32)
    X = df[FEATURES].values.astype(np.float32)

    # Split 80/20, stratified agar proporsi kelas sama.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Standardisasi: statistik HANYA dari data latih (hindari data leakage).
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    Xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(y_tr)

    model = CreditMLP(len(FEATURES))

    # Bobot kelas: kesalahan pada nasabah gagal bayar dihitung lebih berat.
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nMelatih model...")
    for epoch in range(1, 31):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if epoch % 5 == 0:
            print(f"  epoch {epoch:2d} | loss {total / len(Xt):.4f}")

    # Evaluasi pada data uji yang belum pernah dilihat.
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(torch.from_numpy(X_te))).numpy()

    auc = roc_auc_score(y_te, proba)
    acc = ((proba >= 0.5).astype(int) == y_te).mean()

    print(f"\nHasil pada data uji ({len(y_te):,} nasabah):")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Accuracy : {acc:.4f}")

    torch.save(model.state_dict(), ARTIFACT / "model.pt")
    json.dump(
        {
            "auc": float(auc),
            "accuracy": float(acc),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "base_rate": float(y.mean()),
            "features": FEATURES,
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        open(ARTIFACT / "metrics.json", "w"),
        indent=2,
    )
    print(f"\nModel tersimpan: {ARTIFACT / 'model.pt'}")


if __name__ == "__main__":
    main()
