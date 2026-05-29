from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from io import BytesIO
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd
from PyPDF2 import PdfReader
from PyPDF2.errors import DependencyError


OCR_MIN_TEXT_CHARS = 2500
OCR_MAX_PAGES = 40
SUPPLEMENTARY_MAX_ROWS = 500
SUPPLEMENTARY_MAX_CHARS = 120_000


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(value: object) -> str:
    text = clean(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def text_similarity(a: object, b: object) -> float:
    left = normalize_text(a)
    right = normalize_text(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def contains_any(text: str, terms: Iterable[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def next_prefixed_id(existing: Iterable[object], prefix: str, width: int = 3) -> str:
    max_value = 0
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for item in existing:
        match = pattern.match(clean(item))
        if match:
            max_value = max(max_value, int(match.group(1)))
    return f"{prefix}{max_value + 1:0{width}d}"


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
        try:
            from drive_storage import upload_csv

            upload_csv(path)
        except Exception:
            pass


def table_to_evidence_text(df: pd.DataFrame, label: str, max_rows: int = SUPPLEMENTARY_MAX_ROWS) -> str:
    if df.empty:
        return f"\n\n--- SUPPLEMENTARY TABLE: {label} ---\n(empty table)"
    clipped = df.head(max_rows).copy()
    return f"\n\n--- SUPPLEMENTARY TABLE: {label} ---\n{clipped.to_csv(index=False)}"


def supplementary_table_text_from_bytes(data: bytes, name: str) -> str:
    suffix = Path(name).suffix.lower()
    buffer = BytesIO(data)
    if suffix == ".csv":
        df = pd.read_csv(buffer, nrows=SUPPLEMENTARY_MAX_ROWS)
        return table_to_evidence_text(df, name)
    if suffix in {".xlsx", ".xls"}:
        chunks: list[str] = []
        workbook = pd.ExcelFile(buffer)
        for sheet in workbook.sheet_names[:12]:
            df = pd.read_excel(workbook, sheet_name=sheet, nrows=SUPPLEMENTARY_MAX_ROWS)
            chunks.append(table_to_evidence_text(df, f"{name} / {sheet}"))
        return "\n".join(chunks)
    return ""


def extract_supplementary_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".xlsx", ".xls"}:
        return supplementary_table_text_from_bytes(path.read_bytes(), path.name)[:SUPPLEMENTARY_MAX_CHARS]
    if suffix == ".zip":
        chunks: list[str] = []
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member_name = info.filename
                member_suffix = Path(member_name).suffix.lower()
                if member_suffix not in {".csv", ".xlsx", ".xls"}:
                    continue
                if info.file_size > 25_000_000:
                    chunks.append(f"\n\n--- SUPPLEMENTARY FILE SKIPPED: {member_name} ---\nFile is larger than 25 MB.")
                    continue
                try:
                    chunks.append(supplementary_table_text_from_bytes(archive.read(info), member_name))
                except Exception as exc:
                    chunks.append(f"\n\n--- SUPPLEMENTARY FILE SKIPPED: {member_name} ---\nCould not parse table: {exc}")
                if len("\n".join(chunks)) >= SUPPLEMENTARY_MAX_CHARS:
                    break
        return "\n".join(chunks)[:SUPPLEMENTARY_MAX_CHARS]
    return ""


def extract_supplementary_links(text: str) -> list[dict[str, str]]:
    hint_pattern = re.compile(
        r"supplementary|supporting information|supporting data|additional data|data availability|"
        r"available at|available from|mendeley data|figshare|zenodo|dryad|repository|dataset",
        flags=re.IGNORECASE,
    )
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(r"https?://[^\s<>\]\)\"']+|10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, flags=re.IGNORECASE):
        start = max(0, match.start() - 260)
        end = min(len(text), match.end() + 260)
        context = re.sub(r"\s+", " ", text[start:end]).strip()
        url = match.group(0).rstrip(".,);]")
        if not hint_pattern.search(context):
            continue
        if not url.lower().startswith("http"):
            url = f"https://doi.org/{url}"
        if url in seen:
            continue
        seen.add(url)
        label = "Supplementary data"
        lower_context = context.lower()
        if "mendeley" in lower_context:
            label = "Mendeley Data"
        elif "figshare" in lower_context:
            label = "Figshare"
        elif "zenodo" in lower_context:
            label = "Zenodo"
        elif "dryad" in lower_context:
            label = "Dryad"
        rows.append({"label": label, "url": url, "context": context[:420]})
    return rows[:12]


def pdf_text_needs_ocr(text: str, page_count: int) -> bool:
    normalized = normalize_text(text)
    if page_count <= 0:
        return False
    if len(normalized) < OCR_MIN_TEXT_CHARS:
        return True
    words_per_page = len(normalized.split()) / max(1, page_count)
    table_markers = len(re.findall(r"\b(table|figure|wt|mpa|gpa|strength|modulus|absorption|composition)\b", normalized))
    return words_per_page < 80 or table_markers < 4


def ocr_pdf_text(path: Path, max_pages: int = OCR_MAX_PAGES) -> str:
    if not shutil.which("tesseract"):
        raise RuntimeError("OCR is needed for this PDF, but Tesseract is not installed. Install Tesseract, then upload it again.")
    try:
        import fitz
        import pytesseract
    except Exception as exc:
        raise RuntimeError("OCR Python packages are missing. Install requirements, then upload it again.") from exc

    doc = fitz.open(str(path))
    chunks: list[str] = []
    for page_index in range(min(len(doc), max_pages)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = None
        try:
            from PIL import Image

            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(image, config="--psm 6")
            if text.strip():
                chunks.append(f"\n\n[OCR page {page_index + 1}]\n{text}")
        finally:
            if image is not None:
                image.close()
    return "\n".join(chunks)


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            text = "\n".join(pages)
            if pdf_text_needs_ocr(text, len(reader.pages)):
                ocr_text = ocr_pdf_text(path)
                if ocr_text.strip():
                    return f"{text}\n\n--- OCR TEXT FROM PDF IMAGES ---\n{ocr_text}"
            return text
        except DependencyError as exc:
            raise RuntimeError("This PDF uses AES encryption. Install pycryptodome, then upload it again.") from exc
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".csv", ".xlsx", ".xls"}:
        df = read_table(path)
        return df.head(200).to_csv(index=False)
    return path.read_text(encoding="utf-8", errors="ignore")


def dataframe_with_columns(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns]
