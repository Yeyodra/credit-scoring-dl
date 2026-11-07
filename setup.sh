#!/usr/bin/env bash
# Setup cepat proyek di VPS/Linux.
# Pakai: bash setup.sh
set -e
cd "$(dirname "$0")"

echo "==> Membuat virtualenv..."
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Upgrade pip..."
pip install --upgrade pip

echo "==> Install dependencies..."
pip install -r requirements.txt

if [ ! -f artifacts/model.pt ] || [ ! -f artifacts/metrics.json ]; then
  echo "!! artifacts/model.pt atau metrics.json TIDAK ADA."
  echo "   Jalankan 'python train.py' atau upload folder artifacts/ dari laptop."
else
  echo "==> Model terlatih ditemukan di artifacts/ (tidak perlu training ulang)."
fi

echo ""
echo "Selesai. Jalankan server dengan:"
echo "  source venv/bin/activate"
echo "  python app.py                      # dev  -> http://127.0.0.1:5000"
echo "  # atau produksi:"
echo "  pip install gunicorn && gunicorn -w 2 -b 127.0.0.1:8000 app:app"
