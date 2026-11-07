"""
Web app sederhana: prediksi probabilitas gagal bayar (Flask + PyTorch).

Jalankan:  python app.py   ->  buka http://127.0.0.1:5000
"""
import json
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from model import CreditMLP

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "artifacts"

app = Flask(__name__)

metrics = json.load(open(ARTIFACT / "metrics.json"))
FEATURES = metrics["features"]
MEAN = np.array(metrics["mean"], dtype=np.float32)
STD = np.array(metrics["std"], dtype=np.float32)

# Model dimuat sekali saat server mulai.
model = CreditMLP(len(FEATURES))
model.load_state_dict(torch.load(ARTIFACT / "model.pt", map_location="cpu"))
model.eval()


def predict(application: dict) -> float:
    """Hitung probabilitas gagal bayar dari satu data nasabah."""
    x = np.array([[float(application[f]) for f in FEATURES]], dtype=np.float32)
    x = (x - MEAN) / STD
    x = np.clip(x, -5.0, 5.0)  # jaga input ekstrem agar tidak saturasi 0%/100%
    with torch.no_grad():
        logit = model(torch.from_numpy(x))
    return float(torch.sigmoid(logit).item())


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
    app.run(host="127.0.0.1", port=5000, debug=False)
