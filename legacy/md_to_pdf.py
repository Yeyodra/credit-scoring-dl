"""
Konversi dokumen Markdown ke PDF.

Pemakaian:
    python md_to_pdf.py                    # semua dokumen di docs/
    python md_to_pdf.py docs/LAPORAN.md    # satu file

Memakai markdown (MD -> HTML) dan Playwright Chromium (HTML -> PDF).
"""
import sys
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"

CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-size: 8.5pt;
    color: #94a3b8;
  }
}

* { box-sizing: border-box; }

body {
  font-family: "Segoe UI", Calibri, Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.65;
  color: #1e293b;
  margin: 0;
}

h1 {
  font-size: 20pt;
  color: #0f2557;
  margin: 0 0 4pt;
  padding-bottom: 8pt;
  border-bottom: 3px solid #2563eb;
  page-break-after: avoid;
}

h2 {
  font-size: 14pt;
  color: #1e3a8a;
  margin: 22pt 0 8pt;
  padding-bottom: 4pt;
  border-bottom: 1px solid #cbd5e1;
  page-break-after: avoid;
}

h3 {
  font-size: 11.5pt;
  color: #1e40af;
  margin: 16pt 0 6pt;
  page-break-after: avoid;
}

h4 {
  font-size: 10.5pt;
  color: #334155;
  margin: 12pt 0 4pt;
  page-break-after: avoid;
}

p { margin: 0 0 8pt; text-align: justify; }

ul, ol { margin: 0 0 9pt; padding-left: 18pt; }
li { margin-bottom: 3.5pt; }
li > p { margin-bottom: 3pt; }

strong { color: #0f172a; font-weight: 600; }
em { color: #334155; }

code {
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 9pt;
  background: #f1f5f9;
  padding: 1pt 3.5pt;
  border-radius: 3px;
  color: #be123c;
}

pre {
  background: #0f172a;
  color: #e2e8f0;
  padding: 10pt 12pt;
  border-radius: 6px;
  font-size: 8.5pt;
  line-height: 1.5;
  overflow-x: auto;
  page-break-inside: avoid;
  margin: 0 0 10pt;
}

pre code {
  background: none;
  color: inherit;
  padding: 0;
  font-size: inherit;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 11pt;
  font-size: 9.5pt;
  page-break-inside: avoid;
}

th {
  background: #1e3a8a;
  color: #fff;
  text-align: left;
  padding: 6pt 8pt;
  font-weight: 600;
  font-size: 9pt;
}

td {
  padding: 5pt 8pt;
  border-bottom: 1px solid #e2e8f0;
  vertical-align: top;
}

tr:nth-child(even) td { background: #f8fafc; }

blockquote {
  margin: 0 0 10pt;
  padding: 8pt 12pt;
  background: #eff6ff;
  border-left: 3.5px solid #2563eb;
  color: #1e3a8a;
  font-size: 10pt;
}

blockquote p { margin: 0; }

hr {
  border: 0;
  border-top: 1px solid #e2e8f0;
  margin: 16pt 0;
}

a { color: #2563eb; text-decoration: none; }

h2 + p, h3 + p { margin-top: 0; }

/* Jangan pisahkan heading dari isi sesudahnya */
h1, h2, h3, h4 { break-after: avoid-page; }
"""


def convert(md_path: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )

    html = f"""<!DOCTYPE html>
<html lang="id">
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{body}</body>
</html>"""

    pdf_path = md_path.with_suffix(".pdf")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            display_header_footer=False,
            margin={"top": "18mm", "bottom": "20mm", "left": "16mm", "right": "16mm"},
        )
        browser.close()

    return pdf_path


def main() -> None:
    if len(sys.argv) > 1:
        targets = [Path(a) for a in sys.argv[1:]]
    else:
        targets = sorted(DOCS.glob("*.md"))

    for md_path in targets:
        if not md_path.exists():
            print(f"  lewati (tidak ada): {md_path}")
            continue
        pdf = convert(md_path)
        size = pdf.stat().st_size / 1024
        print(f"  {md_path.name:22s} -> {pdf.name:22s} ({size:6.1f} KB)")


if __name__ == "__main__":
    main()
