from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rag_core import l2_normalize, read_jsonl, records_to_chunks, save_vector_index  # noqa: E402


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode downloaded news documents and build a local vector index."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_DIR / "data" / "news.jsonl",
        help="Input JSONL file produced by download_news.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "index",
        help="Directory where embeddings and metadata will be written.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformers model name.",
    )
    parser.add_argument("--chunk-chars", type=int, default=1600)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. "
            "Install it with: pip install -r requirements.txt"
        ) from exc

    records = read_jsonl(args.input)
    chunks = records_to_chunks(
        records,
        max_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )
    if not chunks:
        raise SystemExit(f"No text chunks found in {args.input}")

    model = SentenceTransformer(args.model)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = l2_normalize(embeddings)

    save_vector_index(
        output_dir=args.output_dir,
        embeddings=embeddings,
        chunks=chunks,
        model_name=args.model,
    )

    print(f"Read {len(records)} documents from {args.input}")
    print(f"Encoded {len(chunks)} chunks with {args.model}")
    print(f"Wrote vector index to {args.output_dir}")


if __name__ == "__main__":
    main()
