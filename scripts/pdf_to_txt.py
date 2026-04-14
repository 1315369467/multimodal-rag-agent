"""
将 data/pdfs/ 目录下所有 PDF 文件提取为纯文本，
输出到 data/pdf_output/ 目录，每个 PDF 对应一个同名 .txt 文件。

使用 PyMuPDF (fitz) 作为解析后端。
"""

import sys
from pathlib import Path

import fitz  # PyMuPDF


PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"
OUT_DIR = Path(__file__).parent.parent / "data" / "pdf_output"


def extract_pdf_to_txt(pdf_path: Path, out_path: Path) -> None:
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)
    doc.close()

    full_text = "\n".join(pages_text)
    out_path.write_text(full_text, encoding="utf-8")
    print(f"  [OK] {pdf_path.name} -> {out_path.name}  ({len(full_text)} chars)")


def main() -> None:
    if not PDF_DIR.exists():
        print(f"PDF 目录不存在: {PDF_DIR}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("data/pdfs/ 中没有找到 PDF 文件。")
        return

    print(f"共找到 {len(pdf_files)} 个 PDF，输出到 {OUT_DIR}\n")
    for pdf_path in pdf_files:
        out_path = OUT_DIR / (pdf_path.stem + ".txt")
        extract_pdf_to_txt(pdf_path, out_path)

    print(f"\n完成，共处理 {len(pdf_files)} 个文件。")


if __name__ == "__main__":
    main()
