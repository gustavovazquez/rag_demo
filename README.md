# RAG basico con noticias reales

Este ejemplo arma un RAG minimo con documentos reales descargados desde feeds RSS/Atom.

El flujo es:

1. Descargar noticias reales a `data/news.jsonl`.
2. Dividir las noticias en chunks.
3. Codificar cada chunk con `sentence-transformers`.
4. Guardar un indice vectorial local.
5. Hacer preguntas, recuperar noticias relevantes y construir el prompt RAG.

## Instalacion

Desde esta carpeta:

```bash
pip install -r requirements.txt
```

## 1. Descargar noticias

```bash
python scripts/download_news.py
```

Por defecto descarga items desde BBC Mundo, DW Espanol y Google News sobre inteligencia artificial / machine learning.

Tambien se pueden pasar feeds manualmente:

```bash
python scripts/download_news.py ^
  --feed "https://feeds.bbci.co.uk/mundo/rss.xml" ^
  --feed "https://news.google.com/rss/search?q=inteligencia%20artificial&hl=es-419&gl=US&ceid=US:es-419"
```

Salida esperada:

```text
data/news.jsonl
```

Cada linea es un documento con `title`, `summary`, `source`, `published` y `link`.
El ejemplo usa el texto expuesto por cada feed RSS/Atom. Algunos feeds publican solo titulo
y resumen; otros incluyen contenido mas largo.

## 2. Construir el indice vectorial

```bash
python scripts/build_index.py
```

El modelo por defecto es:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Salida esperada:

```text
data/index/embeddings.npz
data/index/metadata.json
```

## 3. Consultar el RAG

```bash
python scripts/query_rag.py "Que noticias hablan de inteligencia artificial?" --show-prompt
```

El script imprime:

- una respuesta extractiva simple basada en los chunks recuperados;
- las noticias recuperadas con score, fuente, fecha y URL;
- opcionalmente, el prompt RAG completo para pasar a un LLM.

## 4. Correr consultas de ejemplo

```bash
python scripts/demo_queries.py
```

## 5. Ver cómo se genera la respuesta (paso a paso)

```bash
python scripts/generate_answer.py "Que noticias hablan de inteligencia artificial?"
```

Este script es didactico: hace visible cada etapa del RAG (pregunta →
embedding → recuperacion por coseno → contexto → prompt).

El ultimo paso deja el prompt RAG **planteado**, que es exactamente lo que se
le enviaria a un LLM (Claude, GPT, etc.) para redactar la respuesta final. No
llama a ningun modelo ni requiere API key. Como ilustracion del mecanismo,
tambien muestra una respuesta "extractiva" armada con frases del contexto.

## Estructura

```text
rag_basico/
  data/
    news.jsonl              # noticias descargadas
    index/
      embeddings.npz        # matriz de embeddings normalizados
      metadata.json         # chunks y metadatos
  scripts/
    download_news.py        # baja documentos reales desde RSS/Atom
    build_index.py          # codifica documentos y crea indice vectorial
    query_rag.py            # hace consultas al RAG
    demo_queries.py         # consultas predefinidas
    generate_answer.py      # muestra paso a paso como se genera la respuesta
  src/
    rag_core.py             # funciones compartidas
```

## Nota didactica

Este ejemplo no usa una base vectorial externa como FAISS, Chroma o Pinecone. Para mantenerlo basico, guarda embeddings en NumPy y calcula similitud coseno con producto punto. Para corpus grandes, conviene reemplazar esa parte por una base vectorial.
