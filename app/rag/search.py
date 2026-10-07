from pathlib import Path
import json

from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import FAISSVectorStore


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
    / "index.faiss"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
    / "metadata.json"
)


def search_documents(
    query: str,
    top_k: int = 5,
):
    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    embedding_model = EmbeddingModel()

    query_embedding = embedding_model.encode(
        [query]
    )

    dimension = query_embedding.shape[1]

    vector_store = FAISSVectorStore.load(
        INDEX_FILE,
        dimension,
    )

    scores, indices = vector_store.search(
        query_embedding,
        top_k=top_k,
    )

    results = []

    for score, index in zip(scores, indices):
        if index == -1:
            continue

        result = metadata[int(index)].copy()
        result["score"] = float(score)

        results.append(result)

    return results


if __name__ == "__main__":
    query = input("Enter your question: ")

    results = search_documents(
        query,
        top_k=5,
    )

    print("\nTop results:\n")

    for i, result in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document: {result['document']}")
        print(f"Page: {result['page']}")
        print(f"Text: {result['text'][:500]}")
        print("-" * 70)