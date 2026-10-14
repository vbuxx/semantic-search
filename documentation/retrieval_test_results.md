# Hasil Uji Retrieval dan Jawaban LLM

**Eksekusi:** 14 Oktober 2026  
**Model embedding:** `gemini-embedding-2` (768 dimensi)  
**Model generasi:** `gemini-3.5-flash-lite`  
**Index:** FAISS `IndexFlatIP`, 59 vektor  
**MIN_SIMILARITY:** `0.65` (setelah kalibrasi; uji awal memakai `0.50`)

Query disusun sesuai brief: 2 langsung, 2 parafrasa, 1 di luar knowledge base.

## Ringkasan

| Query | Jenis | Target | Retrieval aktual (top-3) | Context yang dipakai | Jawaban LLM | Penilaian |
| --- | --- | --- | --- | --- | --- | --- |
| Bagaimana cara melacak paket saya? | Langsung | FAQ 10 | FAQ 10 (0.7908), FAQ 12 (0.7259), FAQ 6 (0.7259) | 10, 12, 6 | Resi muncul di Pesanan Saya setelah dikirim, lalu dilacak di website kurir. | Retrieval tepat. Jawaban hanya memakai FAQ 10, tidak terseret FAQ 12/6. |
| Apakah bisa bayar pakai cicilan? | Langsung | FAQ 20 | FAQ 20 (0.8342), FAQ 22 (0.7370), FAQ 17 (0.7323) | 20, 22, 17 | Cicilan tersedia untuk kartu kredit tertentu dengan nominal minimal dari bank. | Retrieval tepat. Jawaban tidak mengarang daftar metode bayar dari FAQ 17. |
| Barang saya sudah melewati perkiraan tiba, harus bagaimana? | Parafrasa | FAQ 12 | FAQ 12 (0.8149), FAQ 14 (0.7122), FAQ 9 (0.7000) | 12, 14, 9 | Jika lebih dari 3 hari dari estimasi, hubungi CS dengan nomor resi. | Parafrasa berhasil. Kata "perkiraan tiba" tetap ketemu FAQ keterlambatan. |
| Saya menerima barang yang salah, bagaimana cara mengajukan pengembalian? | Parafrasa | FAQ 54 | FAQ 54 (0.8272), FAQ 24 (0.7803), FAQ 28 (0.7138) | 54, 24, 28 | Ajukan lewat Pesanan Saya + foto, maks. 3 hari sejak diterima. | Retrieval tepat (FAQ 54 peringkat 1). Jawaban lebih mengikuti prosedur umum FAQ 24, bukan label komplain "Barang Tidak Sesuai" di FAQ 54. |
| Bisakah teknisi memasang mesin cuci yang saya beli di rumah? | Di luar KB | Tidak ada | FAQ 15 (0.6017), FAQ 20 (0.5905), FAQ 11 (0.5894) | kosong | Informasi belum tersedia; sarankan hubungi CS. | Perilaku yang diharapkan. FAQ teratas tidak relevan dan gagal ambang. |

## Analisis per query

### 1. Langsung — pelacakan paket

FAQ 10 jelas di puncak. FAQ 12 dan 6 ikut masuk karena masih satu tema pengiriman/status, skornya ~0.73. Itu wajar untuk top-k=3, tapi berarti Context tidak sepenuhnya "bersih". Di uji ini LLM tetap menjawab soal resi, tidak menambahkan prosedur keterlambatan dari FAQ 12.

### 2. Langsung — cicilan

FAQ 20 skornya paling tinggi di antara kelima query (0.83). Neighbor-nya FAQ pembayaran lain. Jawaban mengikuti syarat cicilan di FAQ 20 dan tidak menjanjikan metode yang tidak ditanya.

### 3. Parafrasa — paket lewat estimasi

Query tidak memakai kata "estimasi" atau "belum sampai", tapi memakai "melewati perkiraan tiba". FAQ 12 tetap peringkat 1 dengan 0.81. Ini bukti retrieval menangkap makna, bukan keyword matching. Jawaban meniru FAQ 12, termasuk syarat "lebih dari 3 hari", dan tidak menambah janji pengiriman.

### 4. Parafrasa — barang salah

Retrieval bagus: FAQ 54 (barang salah) lalu FAQ 24 (prosedur pengembalian). Jawaban LLM menggabungkan alur pengembalian 3 hari dari FAQ 24. Fakta itu ada di Context, jadi bukan halusinasi, tapi kurang spesifik dibanding FAQ 54 yang meminta komplain "Barang Tidak Sesuai" plus foto barang yang diterima. Ini celah prompt/context: dua FAQ mirip dalam satu Context, model memilih yang lebih prosedural.

### 5. Di luar knowledge base — pemasangan mesin cuci

Tidak ada FAQ pemasangan. Skor teratas hanya ~0.60 ke FAQ pengiriman/pembayaran yang tidak nyambung. Dengan ambang 0.65, Context kosong dan LLM menolak menjawab. Ini yang diinginkan brief.

## Kalibrasi ambang

Uji pertama memakai `MIN_SIMILARITY=0.50`. Query mesin cuci saat itu tetap memasukkan FAQ 15/20/11 ke Context karena skornya 0.59–0.60. LLM tetap menolak karena Context tidak menyebut pemasangan, tapi itu bergantung pada disiplin prompt, bukan filter retrieval.

Skor query yang memang ada di FAQ berada di 0.79–0.83 untuk hit utama, dan neighbor relevan di 0.70–0.78. Query out-of-scope di ~0.59–0.60. Ambang `0.65` memisahkan keduanya tanpa membuang target FAQ. Nilai ini hanya berlaku untuk dataset dan model embedding ini.

## Kesimpulan

- Semantic search akurat pada 4/4 query yang ada jawabannya, termasuk parafrasa.
- 1/1 query di luar cakupan ditolak setelah ambang dikalibrasi.
- Kualitas jawaban LLM bagus jika Context berisi satu FAQ yang jelas. Jika dua FAQ dekat masuk bersama, jawaban bisa bergeser ke prosedur yang lebih umum.
- Filter similarity perlu, tidak cukup mengandalkan instruksi prompt saja.
