"""Muestra, paso a paso, cómo un RAG genera una respuesta.

Un RAG combina tres etapas: Recuperación (Retrieval), Aumentación
(Augmentation) y Generación (Generation). El resto de scripts del ejemplo se
detienen en la recuperación y arman el prompt, pero nunca dejan ver cómo se
transforma la pregunta en una respuesta. Este script hace visible ese recorrido
completo:

    PASO 1  Pregunta del usuario
    PASO 2  Se codifica la pregunta en un vector (embedding)
    PASO 3  Se busca por similitud coseno contra el índice
    PASO 4  Se arma el contexto con los chunks recuperados
    PASO 5  Se construye el prompt RAG (contexto + pregunta)
    PASO 6  Ese prompt queda listo para enviar a un LLM

El PASO 6 NO llama a ningún modelo: deja el prompt planteado, que es
exactamente lo que se le pasaría a un LLM (por ejemplo Claude o GPT) para
obtener la respuesta final. Como ilustración del mecanismo, además se muestra
una respuesta "extractiva" armada con frases del propio contexto, para dejar en
claro que la respuesta siempre se apoya en lo recuperado.

Ejemplo:

    python scripts/generate_answer.py "¿Qué noticias hablan de inteligencia artificial?"
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rag_core import (  # noqa: E402
    build_rag_prompt,
    extractive_answer,
    load_vector_index,
    search_embeddings,
)


# --------------------------------------------------------------------------- #
# Utilidades de presentación
# --------------------------------------------------------------------------- #
def paso(numero: int, titulo: str) -> None:
    """Imprime un encabezado de paso para separar visualmente cada etapa."""
    print()
    print("=" * 78)
    print(f"PASO {numero}  |  {titulo}")
    print("=" * 78)


def preview_vector(vector, n: int = 8) -> str:
    """Devuelve una vista corta del vector de embedding para inspección."""
    head = ", ".join(f"{v:+.3f}" for v in vector[:n])
    return f"[{head}, ...]  (dim={vector.shape[0]})"


# --------------------------------------------------------------------------- #
# Backends de generación
# --------------------------------------------------------------------------- #
def generar_extractivo(question: str, hits: list[dict]) -> str:
    """Respuesta 'de juguete': arma la respuesta con frases del contexto.

    No es un modelo de lenguaje: selecciona frases de los chunks recuperados
    que comparten términos con la pregunta. Es determinista y no inventa texto,
    pero tampoco redacta ni razona. Sirve para entender que la respuesta SIEMPRE
    se apoya en el contexto recuperado. La generación "de verdad" la haría un LLM
    a partir del prompt del PASO 5.
    """
    return extractive_answer(question, hits)


# --------------------------------------------------------------------------- #
# Pipeline principal
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Muestra paso a paso cómo un RAG genera una respuesta.",
    )
    parser.add_argument("question", help="Pregunta para el RAG.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "index",
        help="Directorio producido por build_index.py.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Chunks a recuperar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- PASO 1: la pregunta ------------------------------------------------ #
    paso(1, "Pregunta del usuario")
    print(args.question)

    # --- PASO 2: pregunta -> embedding -------------------------------------- #
    paso(2, "Se codifica la pregunta como vector (embedding)")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Falta la dependencia sentence-transformers. "
            "Instálala con: pip install -r requirements.txt"
        ) from exc

    embeddings, metadata = load_vector_index(args.index_dir)
    model_name = metadata["model_name"]
    print(f"Modelo de embeddings : {model_name}")
    print(f"Índice cargado       : {metadata['num_chunks']} chunks, "
          f"matriz {embeddings.shape}")

    encoder = SentenceTransformer(model_name)
    query_embedding = encoder.encode(args.question, convert_to_numpy=True)
    print(f"Vector de la pregunta: {preview_vector(query_embedding)}")

    # --- PASO 3: recuperación por similitud coseno -------------------------- #
    paso(3, "Recuperación: similitud coseno pregunta vs. cada chunk")
    hits = search_embeddings(
        query_embedding=query_embedding,
        embeddings=embeddings,
        chunks=metadata["chunks"],
        top_k=args.top_k,
    )
    print(f"Top-{args.top_k} chunks más parecidos:\n")
    for idx, hit in enumerate(hits, start=1):
        title = hit.get("title") or "(sin título)"
        source = hit.get("source") or "(fuente desconocida)"
        print(f"  [{idx}] score={hit['score']:.3f} | {source}")
        print(f"      {title}")

    # --- PASO 4: se arma el contexto ---------------------------------------- #
    paso(4, "Aumentación: los chunks recuperados forman el CONTEXTO")
    for idx, hit in enumerate(hits, start=1):
        fragmento = textwrap.shorten(hit["text"], width=220, placeholder=" ...")
        print(f"  [{idx}] {fragmento}\n")

    # --- PASO 5: se construye el prompt ------------------------------------- #
    paso(5, "Se construye el prompt RAG (instrucciones + contexto + pregunta)")
    prompt = build_rag_prompt(args.question, hits)
    print(prompt)

    # --- PASO 6: el prompt queda listo para el LLM -------------------------- #
    paso(6, "El prompt queda planteado para enviar a un LLM")
    print("El prompt del PASO 5 es lo que se le pasaría a un LLM (Claude, GPT,")
    print("etc.) para redactar la respuesta final. Aquí no se llama a ningún")
    print("modelo: la generación real es ese paso externo.\n")

    print("Como ilustración del mecanismo, una respuesta EXTRACTIVA armada solo")
    print("con frases del contexto recuperado (no redacta ni razona):\n")
    print("-" * 78)
    print(generar_extractivo(args.question, hits))
    print("-" * 78)
    print()
    print("Nota: las marcas [n] apuntan al chunk n del contexto recuperado,")
    print("de modo que cada afirmación queda trazada a su fuente.")


if __name__ == "__main__":
    main()
