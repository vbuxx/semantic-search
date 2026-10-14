# Semantic Search & LLM Application

Prototype asisten customer service e-commerce. Pertanyaan pelanggan dicari ke FAQ lewat semantic search (Gemini Embeddings + FAISS), lalu Gemini menyusun jawaban hanya dari FAQ yang relevan.

## 1. Problem statement

### Objective

Menjawab pertanyaan berulang seputar pemesanan, pengiriman, pembayaran, pengembalian, akun, promo, produk, dan customer service tanpa mengharuskan pelanggan memakai kata yang sama persis dengan FAQ.

### Sumber knowledge base

Dataset memakai file resmi assignment: `data/ecommerce_faq_knowledge_base.csv`. CSV asli tidak diubah.

- 59 entri
- Kolom: `faq_id`, `category`, `question`, `answer`

| Kategori | Entri |
| --- | ---: |
| Pemesanan | 9 |
| Pengiriman | 9 |
| Pembayaran | 8 |
| Pengembalian | 8 |
| Akun | 7 |
| Produk | 7 |
| Promo | 6 |
| Customer Service | 5 |

### Batasan dan asumsi

- Sistem hanya menjawab dari FAQ yang terambil. Tidak memakai pengetahuan umum model.
- Setiap baris FAQ adalah satu unit retrieval (question + answer). Chunking dokumen panjang tidak dipakai.
- Jika tidak ada hasil di atas ambang cosine similarity, jawaban harus menyatakan informasi belum tersedia dan mengarahkan ke customer service.
- Prototype ini bukan sistem RAG lengkap dan tidak menggantikan penanganan kasus pelanggan yang sebenarnya.

## 2. Alur sistem

1. **Ingest.** Pandas membaca CSV, cek kolom, dan hitung distribusi kategori.
2. **Prepare.** Pertanyaan dan jawaban digabung di memori menjadi satu teks per FAQ.
3. **Embed.** Gemini `gemini-embedding-2` membuat vektor 768 dimensi. Satu teks per request, karena model ini menggabungkan banyak input menjadi satu embedding.
4. **Store.** FAISS `IndexFlatIP` menyimpan vektor yang dinormalisasi (inner product = cosine similarity). Index ada di `artifacts/`.
5. **Retrieve.** Query di-embed dengan prefix `task: question answering`, lalu FAISS mengembalikan top-3 beserta skor.
6. **Generate.** Hanya hasil dengan cosine similarity ≥ `MIN_SIMILARITY` (default `0.65`) masuk Context. Prompt memakai struktur Persona, Task, Context, Format.

## 3. Hasil pengujian

Lima query diuji: dua langsung, dua parafrasa, satu di luar knowledge base. Ringkasan:

| Query | Jenis | Target | FAQ teratas | Skor | Jawaban |
| --- | --- | --- | --- | ---: | --- |
| Bagaimana cara melacak paket saya? | Langsung | FAQ 10 | FAQ 10 | 0.7908 | Sesuai FAQ 10 (resi di Pesanan Saya) |
| Apakah bisa bayar pakai cicilan? | Langsung | FAQ 20 | FAQ 20 | 0.8342 | Sesuai FAQ 20 (kartu kredit tertentu) |
| Barang saya sudah melewati perkiraan tiba, harus bagaimana? | Parafrasa | FAQ 12 | FAQ 12 | 0.8149 | Sesuai FAQ 12 (hubungi CS + resi setelah +3 hari) |
| Saya menerima barang yang salah, bagaimana cara mengajukan pengembalian? | Parafrasa | FAQ 54 | FAQ 54 | 0.8272 | Retrieval tepat; jawaban lebih mengikuti prosedur umum FAQ 24 |
| Bisakah teknisi memasang mesin cuci yang saya beli di rumah? | Di luar KB | Tidak ada | FAQ 15 | 0.6017 | Ditolak; di bawah ambang, LLM menyatakan info belum tersedia |

Analisis lengkap ada di [documentation/retrieval_test_results.md](documentation/retrieval_test_results.md).

Retrieval akurat pada keempat query yang ada di FAQ, termasuk dua parafrasa. Query di luar cakupan tidak lolos ambang. Ambang `0.65` dipilih setelah uji awal `0.50` masih memasukkan FAQ tidak relevan (~0.59–0.60) ke Context.

## 4. Keterbatasan

- Bergantung pada Gemini API dan koneksi internet. Saat uji, `gemini-3.8-flash` sempat 503, jadi generasi memakai `gemini-3.5-flash-lite`.
- Ambang similarity harus dikalibrasi ulang jika knowledge base atau model embedding berubah.
- Top-3 sering membawa FAQ tetangga (misalnya pelacakan paket ikut menarik FAQ keterlambatan). Prompt menahan LLM agar tidak menambah kebijakan, tapi context yang terlalu lebar tetap bisa membuat jawaban kurang spesifik.
- Satu FAQ = satu unit. Tidak ada chunking, hybrid search, atau reranker.
- Jawaban bisa berbeda sedikit antar request meskipun retrieval-nya sama.

## 5. Cara menjalankan

Python 3.10 atau lebih baru.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Isi `GEMINI_API_KEY` di file `.env`. File ini dimuat otomatis saat pipeline di-import dan sudah dikecualikan dari Git.

Bangun embedding dan index:

```bash
python scripts/build_index.py
```

Script menulis `artifacts/ecommerce_faq.faiss` dan `artifacts/ecommerce_faq.metadata.json`. Index bisa dibangun ulang dari CSV asli. Jangan commit API key.

Ulangi lima query uji:

```bash
python scripts/run_eval.py
```

Atau buka notebook dari root project, lalu jalankan semua cell dari atas ke bawah:

```bash
jupyter notebook notebook/starter_notebook_semantic_search.ipynb
```

Opsional:

Atur `GEMINI_EMBEDDING_MODEL`, `GEMINI_GENERATION_MODEL`, atau `MIN_SIMILARITY` di `.env` bila perlu; contoh komentarnya tersedia di `.env.example`.

Model embedding mengikuti [dokumentasi Gemini Embeddings](https://ai.google.dev/gemini-api/docs/embeddings).
