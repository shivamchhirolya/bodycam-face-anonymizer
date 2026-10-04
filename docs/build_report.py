#!/usr/bin/env python3
"""Render the technical design report to PDF."""
from __future__ import annotations

from pathlib import Path

from weasyprint import HTML

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
HTML_PATH = ROOT / "technical_report.html"
PDF_PATH = ROOT / "Shivam_Pronto_Bodycam_Face_Anonymizer_Report.pdf"
DEMO_PDF = PROJECT / "outputs" / "demo" / "Shivam_Pronto_Bodycam_Face_Anonymizer_Report.pdf"


def main() -> None:
    HTML(filename=str(HTML_PATH)).write_pdf(str(PDF_PATH))
    DEMO_PDF.parent.mkdir(parents=True, exist_ok=True)
    DEMO_PDF.write_bytes(PDF_PATH.read_bytes())
    print(f"wrote {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"copied {DEMO_PDF}")


if __name__ == "__main__":
    main()
