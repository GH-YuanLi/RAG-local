import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pdf(path: str | Path) -> list[dict]:
    doc = fitz.open(str(path))
    chunks = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if text:
            chunks.append({
                "text": text,
                "metadata": {"source": str(path), "page": page_num},
            })
    doc.close()
    return chunks


def load_docx(path: str | Path) -> list[dict]:
    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not text:
        return []
    return [{"text": text, "metadata": {"source": str(path), "page": 1}}]


def load_text(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    return [{"text": text, "metadata": {"source": str(path), "page": 1}}]


_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_text,
}


def load_file(path: str | Path) -> list[dict]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _LOADERS:
        logger.warning("不支持的文件格式: %s", path)
        return []
    try:
        return _LOADERS[ext](path)
    except Exception as e:
        logger.error("加载文件失败 %s: %s", path, e)
        return []


def split_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", "。", "！", "？", ".", " "]

    if len(text) <= chunk_size:
        return [text]

    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    if len(part) > chunk_size:
                        chunks.extend(split_text(part, chunk_size, chunk_overlap, separators))
                        current = ""
                    else:
                        current = part
            if current:
                chunks.append(current)
            return _merge_small_chunks(chunks, chunk_size, chunk_overlap)

    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]


def _merge_small_chunks(chunks: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    if not chunks:
        return []
    merged = []
    current = chunks[0]
    for chunk in chunks[1:]:
        candidate = current + "\n" + chunk
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            merged.append(current)
            current = chunk
    if current:
        merged.append(current)
    return merged


def load_and_chunk(
    path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
) -> list[dict]:
    raw_chunks = load_file(path)
    if not raw_chunks:
        return []

    result = []
    for raw in raw_chunks:
        sub_chunks = split_text(raw["text"], chunk_size, chunk_overlap, separators)
        for i, text in enumerate(sub_chunks):
            result.append({
                "text": text,
                "metadata": {**raw["metadata"], "chunk_id": i},
            })
    logger.info("文件 %s 加载完成，生成 %d 个分块", path, len(result))
    return result
