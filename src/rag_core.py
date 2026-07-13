from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ0-9]+")

STOPWORDS = {
    "a",
    "al",
    "and",
    "ante",
    "como",
    "con",
    "cuando",
    "de",
    "del",
    "desde",
    "el",
    "en",
    "entre",
    "es",
    "esta",
    "este",
    "for",
    "is",
    "la",
    "las",
    "lo",
    "los",
    "mas",
    "no",
    "of",
    "on",
    "or",
    "para",
    "por",
    "que",
    "se",
    "sin",
    "the",
    "to",
    "un",
    "una",
    "y",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = TOKEN_RE.findall(normalized)
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def document_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("title", ""),
        record.get("summary", ""),
        record.get("content", ""),
    ]
    return clean_whitespace(". ".join(part for part in parts if part))


def chunk_text(text: str, max_chars: int = 1600, overlap_chars: int = 250) -> list[str]:
    text = clean_whitespace(text)
    if not text:
        return []
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            sentence_end = text.rfind(". ", start, end)
            if sentence_end > start + max_chars // 2:
                end = sentence_end + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def records_to_chunks(
    records: list[dict[str, Any]],
    max_chars: int = 1600,
    overlap_chars: int = 250,
) -> list[dict[str, Any]]:
    chunks = []
    for doc_idx, record in enumerate(records):
        text = document_text(record)
        for chunk_idx, chunk in enumerate(
            chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
        ):
            chunks.append(
                {
                    "chunk_id": f"doc-{doc_idx:04d}-chunk-{chunk_idx:03d}",
                    "doc_id": record.get("id", f"doc-{doc_idx:04d}"),
                    "title": record.get("title", ""),
                    "source": record.get("source", ""),
                    "published": record.get("published", ""),
                    "link": record.get("link", ""),
                    "text": chunk,
                }
            )
    return chunks


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def save_vector_index(
    output_dir: Path,
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    model_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "embeddings.npz", embeddings=embeddings)
    metadata = {
        "version": 1,
        "model_name": model_name,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_vector_index(index_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    embeddings = np.load(index_dir / "embeddings.npz")["embeddings"]
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    return embeddings, metadata


def search_embeddings(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    query_embedding = query_embedding.reshape(1, -1)
    query_embedding = l2_normalize(query_embedding)[0]
    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "score": float(scores[idx]),
            **chunks[int(idx)],
        }
        for idx in top_indices
    ]


def build_rag_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    context_blocks = []
    for idx, hit in enumerate(hits, start=1):
        source = hit.get("source") or "unknown"
        title = hit.get("title") or "untitled"
        link = hit.get("link") or "no-link"
        context_blocks.append(
            f"[{idx}] {title}\nFuente: {source}\nURL: {link}\nTexto: {hit['text']}"
        )

    context = "\n\n".join(context_blocks)
    return (
        "Responde en español usando solo el contexto recuperado. "
        "Incluye las fuentes relevantes entre corchetes, por ejemplo [1]. "
        "Si el contexto no permite responder, dilo claramente.\n\n"
        f"Pregunta: {question}\n\n"
        f"Contexto recuperado:\n{context}\n\n"
        "Respuesta:"
    )


def extractive_answer(question: str, hits: list[dict[str, Any]], max_sentences: int = 4) -> str:
    if not hits:
        return "No encontre documentos relevantes en el indice."

    query_terms = set(tokenize(question))
    candidates = []
    for hit_idx, hit in enumerate(hits, start=1):
        for sentence in SENTENCE_RE.split(hit["text"]):
            sentence = sentence.strip()
            if not sentence:
                continue
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap:
                candidates.append((overlap, hit["score"], -hit_idx, hit_idx, sentence))

    if not candidates:
        return (
            "Recupere noticias relacionadas, pero no encontre una frase directa "
            "para construir una respuesta extractiva."
        )

    candidates.sort(reverse=True)
    selected = []
    seen = set()
    for _, _, _, hit_idx, sentence in candidates:
        if sentence in seen:
            continue
        selected.append(f"{sentence} [{hit_idx}]")
        seen.add(sentence)
        if len(selected) >= max_sentences:
            break

    return " ".join(selected)
