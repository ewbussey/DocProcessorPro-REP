"""
ocr_pdf_pipeline_vl.py
Scanned / handwritten PDF -> structured Markdown, using PaddleOCR-VL with
DPI-controlled rasterization via pypdfium2 (Apache-2.0 / BSD-3-Clause --
no AGPL entanglement, unlike PyMuPDF). PaddleOCR itself is Apache-2.0.

Setup (Python 3.9+ required for the doc-parser extra):
    pip install paddlepaddle                 # or paddlepaddle-gpu --
                                              # see paddlepaddle.org.cn/install/quick
    pip install "paddleocr[doc-parser]"      # PaddleOCR-VL extras
    pip install pypdfium2

Usage:
    python ocr_pdf_pipeline_vl.py input.pdf
    python ocr_pdf_pipeline_vl.py input.pdf -o output.md --dpi 300

Notes:
    - PaddleOCR-VL is a ~0.9B-parameter vision-language model -- far
      heavier per page than the general OCR pipeline. CPU-only inference
      on large batches (hundreds of pages) will be slow; use a GPU for
      volume work.
    - First run downloads model weights to ~/.paddlex with no progress
      bar in some modes -- if it looks stalled, check that directory's
      size before assuming it's hung.
    - Pages are rendered and OCR'd one at a time (not batched) to keep
      memory bounded on long documents and so partial output survives
      an interruption partway through -- relevant at the ~450-page scale.
    - Output is Markdown, not flat text: PaddleOCR-VL is a document-parsing
      model and preserves structure (headings, tables). Strip markdown
      syntax afterward if you need plain text instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from paddleocr import PaddleOCRVL


def render_page(pdf_page, dpi: int) -> np.ndarray:
    """Rasterize one pdfium page to an RGB numpy array at the given DPI.

    pypdfium2's render() scale is relative to the PDF's native 72-DPI unit,
    so scale = dpi / 72 reproduces the target resolution.
    """
    scale = dpi / 72.0
    image = pdf_page.render(scale=scale).to_pil().convert("RGB")
    return np.array(image)


def ocr_pdf_vl(pdf_path: Path, output_path: Path, dpi: int = 300) -> int:
    pipeline = PaddleOCRVL(
        use_doc_orientation_classify=True,  # auto-corrects rotated scans
        use_doc_unwarping=True,             # corrects skewed/warped pages
    )

    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)
    prev_ends_complete = True

    with open(output_path, "w", encoding="utf-8") as out:
        for i, page in enumerate(pdf):
            image = render_page(page, dpi)
            page.close()

            # A single image in -> a single Result out; looping (rather
            # than indexing) matches PaddleOCR's documented usage pattern.
            for res in pipeline.predict(image):
                md = res.markdown
                text = md["markdown_texts"]
                starts_new, ends_complete = md.get(
                    "page_continuation_flags", (True, True)
                )

                if i > 0:
                    # Only force a page break when the previous page ended
                    # a paragraph AND this page starts a new one -- avoids
                    # splitting a sentence that runs across a page boundary.
                    if prev_ends_complete and starts_new:
                        out.write(f"\n\n<!-- page {i + 1} -->\n\n")
                    else:
                        out.write(" ")
                out.write(text)
                prev_ends_complete = ends_complete

    pdf.close()
    return n_pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR a scanned/handwritten PDF with PaddleOCR-VL"
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .md path")
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization DPI (default: 300)")
    args = parser.parse_args()

    output_path = args.output or args.pdf.with_suffix(".md")
    n = ocr_pdf_vl(args.pdf, output_path, dpi=args.dpi)
    print(f"Processed {n} page(s) -> {output_path}")