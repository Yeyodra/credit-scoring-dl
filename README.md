# Prediksi Risiko Kredit (Deep Learning)

Sistem sederhana untuk memprediksi **probabilitas gagal bayar** nasabah
menggunakan jaringan saraf tiruan (MLP). Dataset: **Taiwan Credit Card Default**
(UCI Machine Learning Repository).

## Hasil model (data uji 6.000 nasabah)
| Metrik | Nilai |
|---|---|
| AUC-ROC | **0,7701** |
| Accuracy | 74,70% |
| Data latih / uji | 24.000 / 6.000 |
| Parameter | 3.649 |

## Struktur
```
.
├── model.py            definisi MLP (23 -> 64 -> 32 -> 1)
├── train.py            latih + evaluasi, simpan ke artifacts/
├── app.py              web server Flask (PyTorch langsung)
├── templates/
│   └── index.html      antarmuka (6 masukan utama)
├── artifacts/
│   ├── model.pt        bobot model terlatih
│   └── metrics.json    parameter standardisasi + hasil evaluasi
├── data/taiwan/        dataset mentah (.xls)
├── requirements.txt
└── legacy/             arsip versi lama (embedding + deploy VPS) — tidak dipakai
```

## Cara menjalankan
```bash
pip install -r requirements.txt

# (opsional) latih ulang model -> artifacts/model.pt + metrics.json
python train.py

# jalankan web
python app.py
# buka http://127.0.0.1:5000
```

## Akses publik (Cloudflare Quick Tunnel)
```powershell
# terminal terpisah, setelah app.py jalan
.\tools\cloudflared.exe tunnel --url http://127.0.0.1:5000
```
URL `https://<acak>.trycloudflare.com` akan tercetak dan **berubah setiap restart**.

## Catatan
- Antarmuka hanya menampilkan 6 masukan utama; fitur riwayat tagihan/pembayaran
  lainnya diisi otomatis. Nilai masukan dibatasi agar tidak saturasi.
- `legacy/` berisi versi lama (arsitektur embedding + skrip deploy VPS) sebagai arsip.
