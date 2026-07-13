from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rag_core import extractive_answer, load_vector_index, search_embeddings  # noqa: E402


DEFAULT_QUESTIONS = [
    "Que noticias recientes hablan de inteligencia artificial?",
    "Que temas internacionales aparecen en las noticias descargadas?",
    "Hay noticias sobre economia o empresas?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample questions against the RAG.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "index",
        help="Directory produced by build_index.py.",
    )
    parser.add_argument("--top-k", type=int, default=3)
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

    embeddings, metadata = load_vector_index(args.index_dir)
    model = SentenceTransformer(metadata["model_name"])

    for question in DEFAULT_QUESTIONS:
        query_embedding = model.encode(question, convert_to_numpy=True)
        hits = search_embeddings(
            query_embedding=query_embedding,
            embeddings=embeddings,
            chunks=metadata["chunks"],
            top_k=args.top_k,
        )

        print("=" * 80)
        print(f"QUESTION: {question}")
        print("\nANSWER:")
        print(extractive_answer(question, hits))
        print("\nSOURCES:")
        for idx, hit in enumerate(hits, start=1):
            print(f"[{idx}] {hit['score']:.3f} | {hit['title']} | {hit['link']}")
        print()


if __name__ == "__main__":
    main()
