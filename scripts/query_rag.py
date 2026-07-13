from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rag_core import (  # noqa: E402
    build_rag_prompt,
    extractive_answer,
    load_vector_index,
    search_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local news RAG index.")
    parser.add_argument("question", help="Question to ask the RAG.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "index",
        help="Directory produced by build_index.py.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the final RAG prompt assembled from retrieved documents.",
    )
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
    model_name = metadata["model_name"]
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(args.question, convert_to_numpy=True)

    hits = search_embeddings(
        query_embedding=query_embedding,
        embeddings=embeddings,
        chunks=metadata["chunks"],
        top_k=args.top_k,
    )

    print("\nQUESTION")
    print(args.question)

    print("\nEXTRACTIVE ANSWER")
    print(extractive_answer(args.question, hits))

    print("\nRETRIEVED NEWS")
    for idx, hit in enumerate(hits, start=1):
        title = hit.get("title") or "(no title)"
        source = hit.get("source") or "(unknown source)"
        published = hit.get("published") or "(unknown date)"
        link = hit.get("link") or "(no link)"
        print(f"[{idx}] score={hit['score']:.3f} | {source} | {published}")
        print(f"    {title}")
        print(f"    {link}")

    if args.show_prompt:
        print("\nRAG PROMPT")
        print(build_rag_prompt(args.question, hits))


if __name__ == "__main__":
    main()
