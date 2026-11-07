"""Uji end-to-end: jalankan server, kirim beberapa pengajuan, tampilkan hasil."""
import json
import threading
import time

import requests
from werkzeug.serving import make_server

import server

BASE = "http://127.0.0.1:5055"


def run_server():
    srv = make_server("127.0.0.1", 5055, server.app)
    srv.serve_forever()


CASES = [
    ("Nasabah disiplin", {
        "LIMIT_BAL": 500000, "AGE": 42, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 1,
        "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 20000, "BILL_AMT2": 18000, "BILL_AMT3": 16000,
        "BILL_AMT4": 15000, "BILL_AMT5": 14000, "BILL_AMT6": 13000,
        "PAY_AMT1": 20000, "PAY_AMT2": 18000, "PAY_AMT3": 16000,
        "PAY_AMT4": 15000, "PAY_AMT5": 14000, "PAY_AMT6": 13000,
    }),
    ("Nasabah rata-rata", {
        "LIMIT_BAL": 150000, "AGE": 35, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 50000, "BILL_AMT2": 48000, "BILL_AMT3": 46000,
        "BILL_AMT4": 44000, "BILL_AMT5": 42000, "BILL_AMT6": 40000,
        "PAY_AMT1": 8000, "PAY_AMT2": 7500, "PAY_AMT3": 7000,
        "PAY_AMT4": 6800, "PAY_AMT5": 6500, "PAY_AMT6": 6000,
    }),
    ("Nasabah berisiko", {
        "LIMIT_BAL": 30000, "AGE": 23, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 2,
        "PAY_0": 4, "PAY_2": 3, "PAY_3": 3, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
        "BILL_AMT1": 29000, "BILL_AMT2": 28500, "BILL_AMT3": 28000,
        "BILL_AMT4": 27500, "BILL_AMT5": 27000, "BILL_AMT6": 26500,
        "PAY_AMT1": 0, "PAY_AMT2": 500, "PAY_AMT3": 0,
        "PAY_AMT4": 800, "PAY_AMT5": 0, "PAY_AMT6": 1000,
    }),
]


def main():
    threading.Thread(target=run_server, daemon=True).start()
    time.sleep(3)

    page = requests.get(BASE + "/", timeout=10)
    print(f"Halaman utama: HTTP {page.status_code}, {len(page.text):,} karakter")
    assert page.status_code == 200 and "Prediksi Risiko Kredit" in page.text

    print("\n" + "=" * 62)
    print(f"{'Skenario':<24} {'Probabilitas':>14}   {'Kategori':<20}")
    print("=" * 62)

    results = []
    for name, payload in CASES:
        r = requests.post(BASE + "/api/predict", json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        p = data["probability"]

        if p < 0.20:
            cat = "Risiko Rendah"
        elif p < 0.45:
            cat = "Risiko Menengah"
        else:
            cat = "Risiko Tinggi"

        results.append(p)
        print(f"{name:<24} {data['percent']:>13.1f}%   {cat:<20}")

    print("=" * 62)

    assert results[0] < results[1] < results[2], (
        f"Urutan tidak masuk akal: {results}"
    )
    print("Urutan probabilitas benar: disiplin < rata-rata < berisiko")

    bad = requests.post(BASE + "/api/predict", json={"AGE": 30}, timeout=10)
    print(f"Input tidak lengkap -> HTTP {bad.status_code} (ditolak dengan benar)")
    assert bad.status_code == 400

    print("\nSemua pengujian lulus.")


if __name__ == "__main__":
    main()
