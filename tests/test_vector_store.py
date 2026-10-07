import numpy as np

from app.rag.vector_store import FAISSVectorStore


def test_faiss_add_and_search():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype="float32",
    )

    store = FAISSVectorStore(
        dimension=3
    )

    store.add(embeddings)

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype="float32",
    )

    scores, indices = store.search(
        query,
        top_k=2,
    )

    assert len(indices) == 2
    assert indices[0] == 0
    assert scores[0] > scores[1]