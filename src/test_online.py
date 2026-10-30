"""Uji situs online di browser sungguhan.

Pemakaian:
    python test_online.py https://<url-tunnel>.trycloudflare.com
"""
import sys
import time

from playwright.sync_api import sync_playwright

if len(sys.argv) < 2:
    raise SystemExit("Pemakaian: python test_online.py <URL>")

URL = sys.argv[1].rstrip("/")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1000, "height": 1000})

    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(URL, wait_until="networkidle", timeout=60000)

    page.fill("#LIMIT_BAL", "180000")
    page.fill("#AGE", "35")
    page.fill("#BILL_AMT1", "50000")
    page.fill("#PAY_AMT1", "8000")
    page.click("#btn")

    page.wait_for_selector(".pct", timeout=20000)
    time.sleep(1)

    pct = page.inner_text(".pct")
    verdict = page.inner_text(".verdict")

    print(f"URL              : {URL}")
    print(f"Angka PD tampil  : {pct}")
    print(f"Kategori         : {verdict}")
    print(f"Elemen terlihat  : {page.is_visible('.pct')}")
    print(f"Error JavaScript : {errors if errors else 'tidak ada'}")

    assert page.is_visible(".pct"), "Hasil tidak terlihat di situs online"
    assert "%" in pct, f"Angka tidak valid: {pct}"

    page.screenshot(path="../artifacts/bukti_online.png", full_page=True)
    print("\nLULUS: situs online menampilkan hasil prediksi")
    print("Screenshot: artifacts/bukti_online.png")

    browser.close()
