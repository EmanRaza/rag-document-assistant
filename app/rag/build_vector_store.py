from pathlib import Path
import json

import numpy as np

from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import FAISSVectorStore


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.json"
)

VECTOR_STORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
)

INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"


def main():
    print("Loading chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks)} chunks.")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    print("\nCreating embeddings...")

    embeddings = embedding_model.encode(texts)

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    dimension = embeddings.shape[1]

    print(
        f"Embedding dimension: {dimension}"
    )

    print("\nBuilding FAISS index...")

    vector_store = FAISSVectorStore(
        dimension=dimension
    )

    vector_store.add(embeddings)

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save(INDEX_FILE)

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nVector store created successfully.")
    print(f"Index: {INDEX_FILE}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()