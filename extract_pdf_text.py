"""
Day 2: PDF Text Extraction Pipeline

Extracts text from contract PDFs so they can be fed into the NER model
(Day 3) and the fine-tuned classifier (Day 4-5) at inference time. This is
separate from the CUAD data used for training -- CUAD already comes as
text. This script handles real-world PDFs your system will receive later
(e.g. via the FastAPI upload endpoint built later in the project).

Strategy:
  - Try direct text extraction first (fast, works for almost all
    digitally-created contract PDFs).
  - Flag any page where extracted text looks suspiciously short relative
    to the page (a sign it's a scanned image, not real text).
  - OCR fallback (Tesseract) is OFF by default -- most contracts won't
    need it, and it requires installing a separate binary. Turn it on
    with --enable_ocr only if you actually have scanned PDFs.

Usage:
    python extract_pdf_text.py --input_dir ./data/contracts --output_dir ./extracted_data
    python extract_pdf_text.py --input_dir ./data/contracts --output_dir ./extracted_data --enable_ocr
"""

import argparse
import json
import os

import fitz  # PyMuPDF

MIN_CHARS_PER_PAGE = 40  # below this, a page is flagged as "likely scanned"


def extract_pdf(pdf_path: str, enable_ocr: bool = False):
    """Extract text page-by-page from a single PDF.
    Returns (full_text, per_page_info, num_likely_scanned_pages)."""
    doc = fitz.open(pdf_path)
    pages = []
    likely_scanned = 0

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        is_thin = len(text) < MIN_CHARS_PER_PAGE

        if is_thin and enable_ocr:
            text, used_ocr = _ocr_page(page), True
        else:
            used_ocr = False

        if is_thin and not used_ocr:
            likely_scanned += 1

        pages.append({
            "page_number": page_num + 1,
            "char_count": len(text),
            "likely_scanned": is_thin and not used_ocr,
            "used_ocr": used_ocr,
            "text": text,
        })

    doc.close()
    full_text = "\n\n".join(p["text"] for p in pages)
    return full_text, pages, likely_scanned


def _ocr_page(page):
    """OCR fallback for a single page. Requires pytesseract + the Tesseract
    binary installed separately (not just `pip install pytesseract`)."""
    try:
        import pytesseract
        from PIL import Image
        import io

        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img).strip()
    except ImportError:
        raise RuntimeError(
            "OCR fallback requires `pip install pytesseract pillow` AND the "
            "Tesseract binary installed separately. On Windows, download it "
            "from https://github.com/UB-Mannheim/tesseract/wiki, then either "
            "add it to PATH or set: "
            "pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'"
        )
    except Exception as e:
        print(f"  WARNING: OCR failed for this page ({e}). Leaving text empty.")
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./data/contracts",
                         help="Folder containing contract PDFs")
    parser.add_argument("--output_dir", default="./extracted_data")
    parser.add_argument("--enable_ocr", action="store_true",
                         help="Fall back to Tesseract OCR for pages with "
                              "little/no extractable text (requires Tesseract "
                              "binary installed separately)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: input folder not found: {args.input_dir}")
        return

    pdf_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {args.input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {args.input_dir}")
    if args.enable_ocr:
        print("OCR fallback: ENABLED (requires Tesseract binary installed)")
    else:
        print("OCR fallback: disabled (pages with little text will be flagged, not OCR'd)")

    summary = []
    for filename in pdf_files:
        pdf_path = os.path.join(args.input_dir, filename)
        print(f"\nProcessing: {filename}")
        try:
            full_text, pages, likely_scanned = extract_pdf(pdf_path, args.enable_ocr)
        except Exception as e:
            print(f"  ERROR reading {filename}: {e}")
            continue

        out_name = os.path.splitext(filename)[0] + ".json"
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "source_file": filename,
                "num_pages": len(pages),
                "total_chars": len(full_text),
                "likely_scanned_pages": likely_scanned,
                "pages": pages,
                "full_text": full_text,
            }, f, indent=2)

        status = "OK"
        if likely_scanned > 0:
            status = f"{likely_scanned}/{len(pages)} page(s) look scanned (little/no text extracted)"
        print(f"  {len(pages)} pages, {len(full_text)} chars -> {out_path}  [{status}]")

        summary.append({
            "file": filename,
            "pages": len(pages),
            "chars": len(full_text),
            "likely_scanned_pages": likely_scanned,
        })

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    scanned_total = sum(s["likely_scanned_pages"] for s in summary)
    for s in summary:
        flag = " <- check this one" if s["likely_scanned_pages"] > 0 else ""
        print(f"  {s['file']:<50} {s['pages']:>3} pages, {s['chars']:>7} chars{flag}")

    if scanned_total > 0 and not args.enable_ocr:
        print(f"\n{scanned_total} page(s) across your PDFs look like scanned images "
              f"with little/no extractable text. If that's expected (real scanned "
              f"contracts), re-run with --enable_ocr after installing Tesseract.")


if __name__ == "__main__":
    main()
