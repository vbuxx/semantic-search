"""Create and save the Gemini embedding and FAISS index for the FAQ CSV."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from semantic_search import (  # noqa: E402
    build_faiss_index,
    embed_texts,
    get_client,
    load_knowledge_base,
    make_retrievable_texts,
    save_faiss_index,
)


def main():
    data_path = PROJECT_DIR / "data" / "ecommerce_faq_knowledge_base.csv"
    index_path = PROJECT_DIR / "artifacts" / "ecommerce_faq.faiss"
    df = load_knowledge_base(data_path)
    embeddings = embed_texts(make_retrievable_texts(df), client=get_client())
    index = build_faiss_index(embeddings)
    metadata_path = save_faiss_index(index, df, index_path)
    print(f"FAQ: {len(df)} | shape embedding: {embeddings.shape} | vektor FAISS: {index.ntotal}")
    print(f"Index: {index_path}\nMetadata: {metadata_path}")


if __name__ == "__main__":
    main()
