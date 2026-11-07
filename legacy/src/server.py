"""
server.py — Web server prediksi risiko kredit (tanpa PyTorch).

Inferensi dihitung manual dengan NumPy karena model hanya butuh
perkalian matriks dan ReLU. Ini membuat server ringan (~50 MB)
sehingga dapat berjalan di VPS kecil.

Jalankan:
    python server.py
"""
import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "artifacts"

NUMERIC = [
    "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]

app = Flask(__name__)

metrics = json.load(open(ARTIFACT / "metrics.json"))
W = dict(np.load(ARTIFACT / "model_weights.npz"))

MEAN = np.array([metrics["mean"][c] for c in NUMERIC], dtype=np.float32)
STD = np.array([metrics["std"][c] for c in NUMERIC], dtype=np.float32)
CAT_MAP = {
    c: {v: i for i, v in enumerate(metrics["categories"][c])} for c in CATEGORICAL
}


def forward(x_num: np.ndarray, x_cat: np.ndarray) -> float:
    """Forward pass MLP: concat(embedding, numerik) -> Linear -> ReLU -> Linear."""
    emb = [W[f"embeddings.{i}.weight"][x_cat[:, i]] for i in range(len(CATEGORICAL))]
    h = np.concatenate([x_num] + emb, axis=1).astype(np.float32)

    for layer in range(3):
        h = h @ W[f"net.{layer * 3}.weight"].T + W[f"net.{layer * 3}.bias"]
        if layer < 2:
            h = np.maximum(h, 0.0)

    return float(1.0 / (1.0 + np.exp(-h[0, 0])))


def predict(application: dict) -> float:
    x_num = (np.array([[application[c] for c in NUMERIC]], dtype=np.float32) - MEAN) / STD
    x_cat = np.array(
        [[CAT_MAP[c].get(application[c], 0) for c in CATEGORICAL]], dtype=np.int64
    )
    return forward(x_num, x_cat)


@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        p = predict(request.get_json(force=True))
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Input tidak valid: {exc}"}), 400
    return jsonify({"probability": p, "percent": round(p * 100, 1)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Server berjalan di http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port)
