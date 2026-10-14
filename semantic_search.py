"""Pipeline FAQ: Gemini embeddings, FAISS, lalu jawaban Gemini dari context retrieval."""

import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).with_name(".env"))

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_DIMENSION = 768
FAQ_COLUMNS = ("faq_id", "category", "question", "answer")


def load_knowledge_base(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(FAQ_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")
    if df.empty:
        raise ValueError("Knowledge base kosong.")
    if df["faq_id"].duplicated().any():
        raise ValueError("faq_id harus unik.")
    return df


def make_retrievable_texts(df):
    """Gabungkan question + answer jadi satu unit retrieval, plus prefix embedding-2."""
    texts = []
    for row in df.itertuples():
        body = f"Pertanyaan: {row.question}\nJawaban: {row.answer}"
        texts.append(f"title: {row.question} | text: {body}")
    return texts


def prepare_query(query):
    return f"task: question answering | query: {query.strip()}"


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY belum diatur.")
    return genai.Client(api_key=api_key)


def embed_texts(texts, client=None):
    """Embed satu teks per request.

    gemini-embedding-2 menggabungkan list contents jadi satu vektor agregat,
    jadi tidak bisa di-batch seperti embedding-001.
    """
    if not texts:
        raise ValueError("Tidak ada teks untuk dibuat embedding.")
    client = client or get_client()
    vectors = []
    for text in texts:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
        )
        if not response.embeddings:
            raise RuntimeError("Gemini tidak mengembalikan embedding.")
        vectors.append(response.embeddings[0].values)

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (len(texts), EMBEDDING_DIMENSION):
        raise ValueError(f"Shape embedding tidak sesuai: {matrix.shape}")
    return matrix


def build_faiss_index(embeddings):
    vectors = np.asarray(embeddings, dtype=np.float32).copy()
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_faiss_index(index, df, path):
    if index.ntotal != len(df):
        raise ValueError("Jumlah vektor FAISS harus sama dengan jumlah FAQ.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "faq_ids": df["faq_id"].tolist(),
                "dimension": index.d,
                "embedding_model": EMBEDDING_MODEL,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata_path


def load_faiss_index(path, df):
    path = Path(path)
    metadata_path = path.with_suffix(".metadata.json")
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Index belum ada. Jalankan scripts/build_index.py dulu.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["faq_ids"] != df["faq_id"].tolist():
        raise ValueError("Urutan FAQ berubah; bangun ulang index.")
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("Model embedding berubah; bangun ulang index.")
    index = faiss.read_index(str(path))
    if index.ntotal != len(df):
        raise ValueError("Jumlah vektor index tidak cocok dengan knowledge base.")
    return index


def semantic_search(query, df, index, client=None, top_k=3):
    if not query or not query.strip():
        raise ValueError("Query tidak boleh kosong.")
    client = client or get_client()
    query_vector = embed_texts([prepare_query(query)], client=client)
    faiss.normalize_L2(query_vector)
    scores, positions = index.search(query_vector, min(top_k, len(df)))
    results = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        row = df.iloc[int(position)]
        results.append(
            {
                "faq_id": row["faq_id"],
                "category": row["category"],
                "question": row["question"],
                "answer": row["answer"],
                "similarity": float(score),
            }
        )
    return results


def get_min_similarity(value=None):
    raw = os.getenv("MIN_SIMILARITY", "0.65") if value is None else value
    threshold = float(raw)
    if not -1 <= threshold <= 1:
        raise ValueError("MIN_SIMILARITY harus antara -1 dan 1.")
    return threshold


def build_prompt(query, search_results, min_similarity=None):
    threshold = get_min_similarity(min_similarity)
    relevant = [item for item in search_results if item["similarity"] >= threshold]
    if relevant:
        context = "\n\n".join(
            f"FAQ {item['faq_id']} ({item['category']})\n"
            f"Pertanyaan: {item['question']}\n"
            f"Jawaban: {item['answer']}"
            for item in relevant
        )
    else:
        context = "Tidak ada entri FAQ yang melewati ambang relevansi."

    return f"""Persona:
Kamu adalah asisten customer service e-commerce. Jawabanmu ramah, singkat, dan hanya berdasarkan FAQ yang diberikan.

Task:
Jawab pertanyaan pengguna hanya dari Context. Jangan menambah kebijakan, syarat, atau janji yang tidak tertulis di Context. Jika Context tidak menjawab pertanyaan, katakan bahwa informasi tersebut belum tersedia di knowledge base dan sarankan pengguna menghubungi customer service.

Context:
{context}

Pertanyaan pengguna:
{query.strip()}

Format:
Jawaban dalam Bahasa Indonesia, 1-3 kalimat. Jangan menyebut fakta yang tidak ada di Context.
"""


def generate_answer(query, search_results, client=None, min_similarity=None):
    client = client or get_client()
    prompt = build_prompt(query, search_results, min_similarity)
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini tidak mengembalikan teks jawaban.")
            return response.text.strip()
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise last_error


def answer_question(query, df, index, client=None, top_k=3, min_similarity=None):
    client = client or get_client()
    results = semantic_search(query, df, index, client=client, top_k=top_k)
    threshold = get_min_similarity(min_similarity)
    context_results = [item for item in results if item["similarity"] >= threshold]
    answer = generate_answer(query, results, client=client, min_similarity=threshold)
    return {
        "answer": answer,
        "search_results": results,
        "context_results": context_results,
    }
