"""Uji render di browser sungguhan: klik tombol, cek hasil benar-benar tampil."""
import threading
import time

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

import server

PORT = 5077


def run():
    make_server("127.0.0.1", PORT, server.app).serve_forever()


threading.Thread(target=run, daemon=True).start()
time.sleep(3)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")

    page.fill("#LIMIT_BAL", "180000")
    page.fill("#AGE", "35")
    page.fill("#BILL_AMT1", "50000")
    page.fill("#PAY_AMT1", "8000")

    page.click("#btn")
    page.wait_for_selector(".pct", timeout=10000)
    time.sleep(1)

    pct = page.inner_text(".pct")
    verdict = page.inner_text(".verdict")
    visible = page.is_visible(".pct")

    print(f"Angka PD terlihat : {pct}")
    print(f"Elemen terlihat   : {visible}")
    print(f"Kategori          : {verdict}")

    assert visible, "HASIL MASIH TIDAK TERLIHAT"
    assert "%" in pct, f"Angka tidak valid: {pct}"

    page.screenshot(path="../artifacts/bukti_hasil.png", full_page=True)
    print("Screenshot: artifacts/bukti_hasil.png")

    page.select_option("#PAY_0", "4")
    page.fill("#LIMIT_BAL", "30000")
    page.fill("#PAY_AMT1", "0")
    page.click("#btn")
    time.sleep(2)
    pct2 = page.inner_text(".pct")
    verdict2 = page.inner_text(".verdict")
    print(f"\nSetelah diubah ke berisiko: {pct2} ({verdict2})")

    assert pct2 != pct, "Prediksi tidak berubah padahal input berubah"
    print("\nLULUS: hasil tampil dan berubah sesuai input")

    browser.close()
