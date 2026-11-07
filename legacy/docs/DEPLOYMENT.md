# Panduan Deployment

Sistem berjalan di VPS dan dapat diakses publik melalui Cloudflare Tunnel.

---

## URL Akses

```
https://<url-tunnel-anda>.trycloudflare.com
```

Catatan: URL quick tunnel **berubah setiap kali service di-restart**.
Lihat bagian "Mendapatkan URL Terbaru" di bawah.

---

## Arsitektur Deployment

```
Internet
   │
   ▼
Cloudflare Edge (HTTPS)
   │
   ▼
cloudflared tunnel (di VPS, outbound only)
   │
   ▼
127.0.0.1:8000  ← gunicorn (2 worker)
   │
   ▼
Flask app (NumPy inference, tanpa PyTorch)
```

**Mengapa tunnel, bukan buka port:** VPS ini bertipe NAT — port masuk ditutup.
Cloudflare Tunnel bekerja dengan koneksi keluar dari server, sehingga tidak
memerlukan port terbuka maupun IP publik statis.

---

## Spesifikasi Server

| Komponen | Nilai |
|---|---|
| OS | Ubuntu 22.04 |
| CPU | 1 vCPU |
| RAM | 2 GB |
| Disk | 78 GB (terpakai 9,4 GB) |
| Python | 3.10 |

**Keputusan penting:** PyTorch **tidak** diinstal di server. Bobot model
diekspor ke NumPy (`model_weights.npz`) dan forward pass dihitung manual
dengan operasi matriks.

Alasannya:
- PyTorch memakan ~1 GB RAM/disk — terlalu berat untuk VPS 2 GB
- Model hanya butuh `Linear + ReLU + Sigmoid`, mudah direplikasi
- Hasil **identik**: selisih maksimum 1,2×10⁻⁷ (batas presisi float32),
  diverifikasi pada 200 input acak di `src/verify_parity.py`

Konsumsi RAM saat berjalan: ~394 MB dari 1963 MB.

---

## Service Systemd

Dua service berjalan otomatis saat boot:

| Service | Fungsi |
|---|---|
| `credit-app.service` | gunicorn menjalankan Flask di 127.0.0.1:8000 |
| `credit-tunnel.service` | cloudflared menghubungkan ke internet |

Keduanya `enabled` (auto-start) dan `Restart=always` (auto-restart jika crash).

### Perintah Operasional

```bash
# Status
systemctl status credit-app credit-tunnel

# Restart
sudo systemctl restart credit-app

# Lihat log
sudo journalctl -u credit-app -n 50
sudo journalctl -u credit-tunnel -n 50

# Hentikan sementara
sudo systemctl stop credit-app credit-tunnel
```

---

## Mendapatkan URL Terbaru

Quick tunnel menghasilkan domain acak setiap restart:

```bash
sudo journalctl -u credit-tunnel --no-pager -n 50 | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1
```

---

## Update Aplikasi

Dari komputer lokal:

```powershell
$pw = 'PASSWORD'
$hk = "SHA256:UkZEhpeWkvTYkyrN1OYoHCrjsCdp6ct6uVlMFPXmw+k"

pscp -P VPS_PORT -pw $pw -hostkey $hk "src\server.py" ubuntu@VPS_HOST:/home/ubuntu/credit-app/src/
pscp -P VPS_PORT -pw $pw -hostkey $hk "src\templates\index.html" ubuntu@VPS_HOST:/home/ubuntu/credit-app/src/templates/

plink -ssh -P VPS_PORT -l ubuntu -pw $pw -hostkey $hk VPS_HOST "sudo systemctl restart credit-app"
```

Jika model dilatih ulang, ekspor dulu bobotnya:

```bash
# di lokal
cd src
python train.py
python export_weights.py

# lalu upload artifacts/metrics.json dan artifacts/model_weights.npz
```

---

## Struktur di Server

```
/home/ubuntu/
├── venv/                        virtualenv (flask, numpy, gunicorn)
├── credit-app/
│   ├── src/
│   │   ├── server.py            aplikasi Flask
│   │   └── templates/
│   │       └── index.html
│   └── artifacts/
│       ├── metrics.json         parameter standardisasi + hasil evaluasi
│       └── model_weights.npz    bobot model (19 KB)
├── run-tunnel.sh
├── credit-tunnel.service
└── credit-app.service
```

---

## Keamanan

**Yang perlu segera dilakukan:**

1. **Ganti password VPS.** Password saat ini dikirim melalui chat sehingga
   harus dianggap sudah bocor.
   ```bash
   passwd
   ```

2. **Pindah ke SSH key dan matikan login password.** Ini menghilangkan
   risiko brute force sepenuhnya.
   ```bash
   # di lokal: buat key
   ssh-keygen -t ed25519

   # kirim public key ke server
   type $env:USERPROFILE\.ssh\id_ed25519.pub | plink -ssh -P VPS_PORT -l ubuntu -pw "PASSWORD" VPS_HOST "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

   # di server: matikan login password
   sudo sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
   sudo systemctl restart ssh
   ```

3. **Batasi akses SSH** melalui Cloudflare Access atau firewall jika
   memungkinkan.

---

## Batasan Deployment

1. **URL berubah setiap restart.** Untuk URL permanen diperlukan domain
   sendiri yang dikelola di dashboard Cloudflare (named tunnel).
2. **Tanpa HTTPS kustom.** Sertifikat mengikuti domain trycloudflare.com.
3. **Tanpa rate limiting.** Endpoint publik dapat dipanggil tanpa batas.
4. **Tanpa autentikasi.** Siapa pun yang memiliki URL dapat mengakses.
5. **Quick tunnel tidak untuk produksi.** Cloudflare menyatakan layanan ini
   untuk pengujian; untuk produksi gunakan named tunnel.

---

## Mengapa Tidak Training di VPS

VPS memiliki 1 vCPU dan 2 GB RAM. Melatih model pada 30.000 baris data
memerlukan waktu jauh lebih lama dan berisiko kehabisan memori.

Alur yang dipakai:
```
Lokal (Windows)          →  VPS (Ubuntu)
train.py (30 epoch)          server.py (inferensi saja)
export_weights.py    ───►    model_weights.npz
```

Training sekali di lokal, hasilnya di-deploy. Ini juga praktik yang lazim:
proses pelatihan dan penyajian dipisahkan.
