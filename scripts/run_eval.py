"""Jalankan 5 query uji assignment dan cetak hasil retrieval + jawaban LLM."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from semantic_search import (  # noqa: E402
    answer_question,
    get_client,
    get_min_similarity,
    load_faiss_index,
    load_knowledge_base,
)


CASES = [
    {
        "query": "Bagaimana cara melacak paket saya?",
        "jenis": "Langsung",
        "target_faq_id": "10",
    },
    {
        "query": "Apakah bisa bayar pakai cicilan?",
        "jenis": "Langsung",
        "target_faq_id": "20",
    },
    {
        "query": "Barang saya sudah melewati perkiraan tiba, harus bagaimana?",
        "jenis": "Parafrasa",
        "target_faq_id": "12",
    },
    {
        "query": "Saya menerima barang yang salah, bagaimana cara mengajukan pengembalian?",
        "jenis": "Parafrasa",
        "target_faq_id": "54",
    },
    {
        "query": "Bisakah teknisi memasang mesin cuci yang saya beli di rumah?",
        "jenis": "Di luar knowledge base",
        "target_faq_id": None,
    },
]


def main():
    df = load_knowledge_base(PROJECT_DIR / "data" / "ecommerce_faq_knowledge_base.csv")
    index = load_faiss_index(PROJECT_DIR / "artifacts" / "ecommerce_faq.faiss", df)
    client = get_client()
    threshold = get_min_similarity()
    print(f"MIN_SIMILARITY={threshold:.2f}\n")

    for case in CASES:
        result = answer_question(case["query"], df, index, client=client)
        print("=" * 80)
        print(f"Jenis : {case['jenis']}")
        print(f"Query : {case['query']}")
        print(f"Target: {case['target_faq_id'] or 'tidak ada'}")
        print("Retrieval:")
        context_ids = {item["faq_id"] for item in result["context_results"]}
        for item in result["search_results"]:
            used = "context" if item["faq_id"] in context_ids else "di bawah ambang"
            print(
                f"  FAQ {item['faq_id']} | {item['similarity']:.4f} | {used} | {item['question']}"
            )
        print("Jawaban:")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
