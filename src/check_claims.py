"""Verifikasi klaim laporan: jumlah parameter & perbandingan ukuran model."""
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
from train import CATEGORICAL, DATA, NUMERIC, SEED, TARGET

torch.manual_seed(SEED)
np.random.seed(SEED)

df = pd.read_excel(DATA, header=1)
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
y = df[TARGET].values.astype(np.float32)
idx = np.arange(len(df))
i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
train, test = df.iloc[i_tr], df.iloc[i_te]

mean, std = train[NUMERIC].mean(), train[NUMERIC].std().replace(0, 1)
xn_tr = ((train[NUMERIC] - mean) / std).values.astype(np.float32)
xn_te = ((test[NUMERIC] - mean) / std).values.astype(np.float32)

cats = {c: sorted(train[c].unique()) for c in CATEGORICAL}
xc_tr = np.stack([train[c].map({v: i for i, v in enumerate(cats[c])}).values for c in CATEGORICAL], axis=1).astype(np.int64)
xc_te = np.stack([test[c].map({v: i for i, v in enumerate(cats[c])}).fillna(0).values for c in CATEGORICAL], axis=1).astype(np.int64)
ytr, yte = y[i_tr], y[i_te]

cards = [len(cats[c]) for c in CATEGORICAL]


class Variant(nn.Module):
    def __init__(self, n_num, cards, hidden, emb_dim=4):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cards])
        d = n_num + emb_dim * len(cards)
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xn, xc):
        parts = [xn] + [e(xc[:, i]) for i, e in enumerate(self.embeddings)]
        return self.net(torch.cat(parts, 1)).squeeze(1)


def run(hidden):
    m = Variant(len(NUMERIC), cards, hidden)
    n_params = sum(p.numel() for p in m.parameters())
    ds = TensorDataset(torch.from_numpy(xn_tr), torch.from_numpy(xc_tr), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    n_pos = ytr.sum()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(ytr) - n_pos) / n_pos))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(30):
        m.train()
        for a, b, c in dl:
            opt.zero_grad()
            crit(m(a, b), c).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.from_numpy(xn_te), torch.from_numpy(xc_te))).numpy()
    return n_params, roc_auc_score(yte, p)


print(f"{'arsitektur':<22} {'parameter':>12} {'AUC uji':>10}")
print("-" * 48)
for h in [[64, 32], [128, 64, 32], [512, 256, 128], [1024, 512, 256]]:
    np_, auc = run(h)
    print(f"{str(h):<22} {np_:>12,} {auc:>10.4f}")

saved = CreditMLP(len(NUMERIC), cards)
print(f"\nModel tersimpan ({'[64,32]'}): {sum(p.numel() for p in saved.parameters()):,} parameter")
