# Individual Reflection — Lab 18

**Tên:** Đỗ Thị Thanh Bình - 2A202600717
**Module phụ trách:** M1, M2, M3, M4, M5 (solo)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** M1 (Chunking), M2 (Hybrid Search), M3 (Reranking), M4 (RAGAS Eval), M5 (Enrichment)
- **Các hàm/class chính đã viết:**
  - M1: `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`
  - M2: `BM25Search`, `DenseSearch`, `HybridSearch`, `reciprocal_rank_fusion()`
  - M3: `CrossEncoderReranker.rerank()`
  - M4: `evaluate_ragas()`, `failure_analysis()`, `save_report()`
  - M5: `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `extract_metadata()`, `_enrich_single_call()`, `enrich_chunks()`
- **Số tests pass:** 37/37 (103s — model load lần đầu mất ~60s)

## 2. Kiến thức học được

- **Khái niệm mới nhất:** HyQA (Hypothesis Question Answering) — thay vì index raw text, generate câu hỏi mà chunk có thể trả lời rồi index câu hỏi đó. Giúp bridge vocabulary gap giữa query của user và ngôn ngữ trong document, cải thiện context_recall từ 0.72 lên 0.88.

- **Điều bất ngờ nhất:** BM25 hybrid search làm giảm precision so với dense-only. Mặc dù hybrid thường được coi là tốt hơn, nhưng BM25 keyword-matching thêm noise vào retrieved chunks — CrossEncoder reranker không đủ để lọc hết trong top-5. Dense-only với semantic similarity cho context_precision cao hơn (0.9361 vs 0.8989).

- **Kết nối với bài giảng:**
  - Slide về Chunking Strategies → hierarchical parent-child (M1)
  - Slide về Hybrid Search & RRF → M2 BM25+Dense fusion
  - Slide về Reranking → CrossEncoder M3
  - Slide về RAG Evaluation → RAGAS 4 metrics M4
  - Slide về Contextual Enrichment & HyDE/HyQA → M5

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** RAGAS 0.4.3 có nhiều breaking changes so với docs online (vốn viết cho 0.1.x):
  - `to_pandas()` không include input columns → KeyError `'question'`
  - `answer_relevancy.embeddings = None` by default → metric trả về 0.0
  - `LangchainLLMWrapper` deprecated → phải dùng `llm_factory`
  - Mặc định `is_async=True` gây rate limit burst → phải dùng `batch_size=4`

- **Cách giải quyết:** Đọc source code của ragas 0.4.3 trực tiếp, dùng `inspect.signature(evaluate)` để kiểm tra tham số thực tế, test từng fix riêng lẻ.

- **Thời gian debug:** ~3 giờ cho RAGAS bugs, ~2 giờ tìm ra dense-only tốt hơn hybrid cho precision.

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:**
  1. **Corpus audit trước**: Kiểm tra 20 câu hỏi test_set có đáp án trong corpus không trước khi optimize pipeline. Ít nhất 3-4 câu hiện tại "unanswerable" vì corpus thiếu data (chu kỳ đổi mật khẩu, mức phạt tạm ứng...).
  2. **Thử dense-only sớm hơn**: Không assume hybrid luôn tốt hơn dense-only — test riêng từng configuration từ đầu.
  3. **Cache evaluation results**: Lưu RAGAS results theo từng config để so sánh, không phải re-run toàn bộ.

- **Module nào muốn thử tiếp:**
  - **Parent retrieval**: Dùng child chunks để retrieval (precision cao), trả về parent chunks (2048 chars) cho LLM (recall cao hơn).
  - **Confidence-based rejection**: Nếu max reranker score < threshold → từ chối trả lời thay vì hallucinate.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 5 (solo nên không có teamwork thực sự) |
| Problem solving | 5 |
