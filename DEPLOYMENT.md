# Panduan Deploy ke VPS (versi ringkas)

## Perlu training ulang? TIDAK.
Model yang sudah terlatih ikut di dalam folder **`artifacts/`**:
- `artifacts/model.pt` — bobot model
- `artifacts/metrics.json` — parameter standardisasi (mean/std) + hasil evaluasi

Selama dua file itu ikut di-upload, server langsung bisa jalan. **Training hanya
perlu diulang** kalau kamu mengganti dataset, mengubah arsitektur, atau ingin
melatih dari nol. Untuk sekadar menyalakan server, tidak perlu.

> Catatan: melatih di VPS kecil (1 vCPU / 2 GB RAM) memang bisa tapi lambat dan
> berisiko kehabisan memori. Praktik yang lazim: **latih di laptop, kirim hasilnya
> (folder `artifacts/`) ke VPS.**

---

## Prasyarat VPS
- Ubuntu 22.04 (atau distro Linux lain)
- Python 3.10+ dan `python3-venv`
- Akses SSH

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

---

## 1. Upload proyek ke VPS
Pilih salah satu:

**A. Via Git (kalau repo publik/ada akses):**
```bash
git clone https://github.com/Yeyodra/credit-scoring-dl.git
cd credit-scoring-dl
```

**B. Via ZIP (dari GDrive / scp):**
```bash
# dari komputer lokal:
scp credit-scoring-dl.zip ubuntu@VPS_IP:/home/ubuntu/
# di VPS:
cd /home/ubuntu
unzip credit-scoring-dl.zip
cd credit-scoring-dl
```

Pastikan folder `artifacts/` ada isinya (`model.pt`, `metrics.json`).

---

## 2. Setup environment & install
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Atau cukup jalankan skrip otomatis:
```bash
bash setup.sh
```

> **Hemat RAM (opsional):** kalau VPS-mu hanya 2 GB dan *tidak* akan melatih ulang,
> cukup install kebutuhan serving saja:
> ```bash
> pip install torch numpy flask
> ```
> (`pandas`, `scikit-learn`, `xlrd`, `openpyxl` hanya dipakai saat `train.py`.)

---

## 3. Jalankan server

**Mode development (cek cepat):**
```bash
python app.py
# buka http://127.0.0.1:5000
```

**Mode produksi (disarankan) — pakai gunicorn:**
```bash
pip install gunicorn
gunicorn -w 2 -b 127.0.0.1:8000 app:app
```
> Untuk VPS 2 GB, pakai `-w 1` atau `-w 2` agar tidak kehabisan RAM.

---

## 4. Akses dari internet

**Opsi A — Cloudflare Quick Tunnel (tanpa buka port, cocok VPS NAT):**
```bash
# unduh cloudflared (sekali)
wget -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared
./cloudflared tunnel --url http://127.0.0.1:8000
```
URL `https://<acak>.trycloudflare.com` akan muncul (berubah tiap restart).

**Opsi B — buka port langsung (kalau VPS punya IP publik):**
```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
# lalu buka port 8000 di firewall/security group
```

---

## 5. (Opsional) Auto-start pakai systemd
Buat `/etc/systemd/system/credit-app.service`:
```ini
[Unit]
Description=Credit Scoring App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/credit-scoring-dl
ExecStart=/home/ubuntu/credit-scoring-dl/venv/bin/gunicorn -w 2 -b 127.0.0.1:8000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now credit-app
sudo systemctl status credit-app
```

---

## 6. Update model di kemudian hari
Kalau model dilatih ulang di laptop:
```bash
python train.py          # hasilkan artifacts/model.pt + metrics.json
```
Lalu upload dua file itu ke VPS (menimpa yang lama), dan restart:
```bash
scp artifacts/model.pt artifacts/metrics.json ubuntu@VPS_IP:/home/ubuntu/credit-scoring-dl/artifacts/
sudo systemctl restart credit-app
```
Tidak perlu install ulang apa pun.

---

## Catatan RAM (penting untuk VPS 2 GB)
Versi ringkas ini memakai **PyTorch** langsung untuk inferensi. PyTorch cukup
berat (~2 GB disk, ~300–500 MB RAM saat jalan). Kalau VPS-mu sangat terbatas,
ada alternatif di folder **`legacy/`**: `legacy/src/server.py` memakai inferensi
**NumPy murni** (tanpa PyTorch, ~390 MB RAM) dengan hasil identik. Itu sebabnya
versi lama dibuat — pakai itu kalau RAM mepet.
