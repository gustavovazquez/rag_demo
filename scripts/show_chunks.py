"""Muestra en qué chunks se divide una noticia.

Antes de indexar, cada noticia se convierte en texto (`document_text`) y se
parte en fragmentos solapados (`chunk_text`). Esos chunks son la unidad que se
codifica y se recupera en el RAG. Este script deja ver ese paso:

    - lista las noticias disponibles (--list);
    - para una noticia, muestra su texto completo y cómo queda dividido en
      chunks, resaltando la zona de SOLAPAMIENTO entre chunks consecutivos.

Con el corpus de RSS por defecto las noticias son cortas, así que cada una suele
generar un único chunk (max_chars=1600). Para ver el mecanismo de división y
solapamiento en acción, bajá el tamaño de chunk:

    python scripts/show_chunks.py --list
    python scripts/show_chunks.py --index 0
    python scripts/show_chunks.py --search inteligencia
    python scripts/show_chunks.py --index 0 --max-chars 200 --overlap 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rag_core import (  # noqa: E402
    chunk_text,
    document_text,
    normalize_text,
    read_jsonl,
)


def overlap_len(prev: str, curr: str, max_probe: int = 400) -> int:
    """Longitud del sufijo de `prev` que coincide con el prefijo de `curr`.

    Sirve para resaltar cuánto texto se repite entre dos chunks consecutivos.
    La comparación se hace sobre texto normalizado (sin tildes ni mayúsculas)
    para ser robusta al recorte de espacios que hace chunk_text.
    """
    a = normalize_text(prev)
    b = normalize_text(curr)
    limit = min(len(a), len(b), max_probe)
    for k in range(limit, 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def list_news(records: list[dict], max_chars: int, overlap_chars: int) -> None:
    print(f"{'idx':>4}  {'chunks':>6}  {'chars':>6}  fuente / título")
    print("-" * 78)
    for idx, record in enumerate(records):
        text = document_text(record)
        n_chunks = len(chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars))
        source = (record.get("source") or "?")[:22]
        title = (record.get("title") or "(sin título)")[:60]
        print(f"{idx:>4}  {n_chunks:>6}  {len(text):>6}  {source:22} | {title}")


def find_indices(records: list[dict], term: str) -> list[int]:
    needle = normalize_text(term)
    matches = []
    for idx, record in enumerate(records):
        haystack = normalize_text(
            f"{record.get('title', '')} {record.get('summary', '')}"
        )
        if needle in haystack:
            matches.append(idx)
    return matches


def show_one(record: dict, idx: int, max_chars: int, overlap_chars: int) -> None:
    text = document_text(record)
    chunks = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)

    print("=" * 78)
    print(f"NOTICIA idx={idx}")
    print("=" * 78)
    print(f"Título   : {record.get('title') or '(sin título)'}")
    print(f"Fuente   : {record.get('source') or '?'}")
    print(f"Fecha    : {record.get('published') or '?'}")
    print(f"Link     : {record.get('link') or '?'}")
    print()
    print(f"Texto del documento ({len(text)} caracteres):")
    print("-" * 78)
    print(text)
    print("-" * 78)
    print(
        f"\nParámetros de chunking: max_chars={max_chars}, overlap_chars={overlap_chars}"
    )
    print(f"La noticia se divide en {len(chunks)} chunk(s):\n")

    prev = None
    for chunk_idx, chunk in enumerate(chunks):
        ov = overlap_len(prev, chunk) if prev is not None else 0
        print("·" * 78)
        header = f"CHUNK {chunk_idx}  ({len(chunk)} caracteres"
        if ov:
            header += f", {ov} de solapamiento con el chunk anterior"
        header += ")"
        print(header)
        print("·" * 78)
        if ov:
            # Marca la parte inicial repetida del chunk anterior.
            print("«…solapa:» " + chunk[:ov])
            print("«nuevo :» " + chunk[ov:])
        else:
            print(chunk)
        print()
        prev = chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Muestra en qué chunks se divide una noticia.",
    )
    parser.add_argument(
        "--news",
        type=Path,
        default=PROJECT_DIR / "data" / "news.jsonl",
        help="Archivo de noticias producido por download_news.py.",
    )
    parser.add_argument("--index", type=int, help="Índice de la noticia a mostrar.")
    parser.add_argument(
        "--search",
        help="Muestra las noticias cuyo título/resumen contienen este término.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista todas las noticias con su número de chunks.",
    )
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--overlap", dest="overlap_chars", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.news.exists():
        raise SystemExit(
            f"No se encontró {args.news}. Corré primero: "
            "python scripts/download_news.py"
        )

    records = read_jsonl(args.news)
    if not records:
        raise SystemExit("El archivo de noticias está vacío.")

    if args.list:
        list_news(records, args.max_chars, args.overlap_chars)
        return

    if args.search:
        matches = find_indices(records, args.search)
        if not matches:
            raise SystemExit(f"Ninguna noticia coincide con: {args.search!r}")
        print(f"{len(matches)} noticia(s) coinciden con {args.search!r}: "
              f"{', '.join(map(str, matches))}\n")
        for idx in matches:
            show_one(records[idx], idx, args.max_chars, args.overlap_chars)
        return

    idx = args.index if args.index is not None else 0
    if not 0 <= idx < len(records):
        raise SystemExit(
            f"Índice fuera de rango: {idx}. Hay {len(records)} noticias (0..{len(records) - 1})."
        )
    show_one(records[idx], idx, args.max_chars, args.overlap_chars)


if __name__ == "__main__":
    main()
