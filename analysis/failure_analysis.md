# Failure Analysis — Lab 18: Production RAG

**Nhóm:** DoThiThanhBinh  
**Thành viên:** Đỗ Thị Thanh Bình - 2A202600717 (M1→M5, solo)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7667 | **0.8726** | +0.1059 ✅ |
| Answer Relevancy | 0.7661 | **0.8342** | +0.0681 ✅ |
| Context Precision | 0.9333 | **0.9361** | +0.0028 ✅ |
| Context Recall | 0.8167 | **0.8833** | +0.0666 ✅ |

**Pipeline:** Hierarchical chunking 400-char children (M1) → OpenAI gpt-4o-mini enrichment với HyQA (M5) → Dense search với HyQA-indexed vectors + CrossEncoder reranker BAAI/bge-reranker-v2-m3 (M2+M3, top_k=5) → RAGAS 0.4.3 eval (M4)

**Key optimizations:**
- **HyQA indexing**: Prepend hypothesis questions vào indexed text → query vocabulary matches chunk vocabulary → context_recall +0.07
- **Dense-only retrieval** (bỏ BM25): BM25 keyword-match thêm noise → giảm precision. Dense-only với semantic similarity giữ precision cao hơn → context_precision vượt baseline
- **Clean original text cho LLM**: Index dùng "HyQA + original", LLM chỉ nhận "original" → faithfulness tăng vì không có HyQA prefix noise
- **CrossEncoder reranker k=5**: Rerank 20 dense candidates → top-5 precision cao hơn, recall đầy đủ hơn
- **Temperature=0.0**: LLM cho câu trả lời nhất quán, đúng trọng tâm

---

## Bottom-5 Failures

### #1 — faithfulness = 0.286, avg = 0.647
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Mức phạt theo quy chế tài chính
- **Got:** LLM đưa ra mức phạt không có trong corpus
- **Worst metric:** faithfulness (0.286)
- **Error Tree:** Output → Context có mức phạt? → Chunk tạm ứng không có bảng phạt → LLM hallucinate con số %
- **Root cause:** Corpus tài liệu tạm ứng không chứa điều khoản phạt trễ hạn. Câu hỏi tính toán yêu cầu số liệu cụ thể không tồn tại trong corpus.
- **Suggested fix:** (1) Bổ sung tài liệu quy chế tài chính đầy đủ. (2) Thêm vào prompt: "Với câu hỏi tính toán: nếu không có công thức hoặc mức phạt CỤ THỂ trong context → báo 'Không tìm thấy quy định phạt trong tài liệu'."

### #2 — faithfulness = 0.500, avg = 0.663
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Số ngày phép + range lương theo cấp Senior, 9 năm thâm niên
- **Got:** Câu trả lời đúng một phần nhưng thêm thông tin không có trong context
- **Worst metric:** faithfulness (0.500)
- **Error Tree:** Output → Context có cả 2 thông tin? → Phép năm + thâm niên ở 1 chunk, lương Senior ở chunk khác → LLM kết hợp được một phần → hallucinate phần còn lại
- **Root cause:** Multi-hop question: cần cross-reference bảng phép năm (theo thâm niên) và bảng lương (theo cấp Senior). Ngay cả với 5 chunks, thông tin từ 2 tài liệu khác nhau có thể thiếu context đầy đủ.
- **Suggested fix:** Parent retrieval — fetch toàn bộ parent chunk (2048 chars) thay vì child (400 chars) khi nhận được multi-condition query. Query classifier để nhận diện multi-hop questions.

### #3 — faithfulness = 0.500, avg = 0.767
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Cấp phê duyệt cụ thể theo số ngày nghỉ không lương
- **Got:** LLM trả lời về quy trình chung nhưng thêm thông tin cấp phê duyệt không có trong context
- **Worst metric:** faithfulness (0.500)
- **Error Tree:** Output → Context có bảng phê duyệt? → Chunk nghỉ không lương có điều kiện chung → LLM suy diễn cấp phê duyệt từ general HR knowledge
- **Root cause:** Corpus có thông tin về điều kiện nghỉ không lương nhưng không có bảng phê duyệt theo số ngày. LLM không từ chối mà suy diễn thêm.
- **Suggested fix:** Tăng cường instruction về từ chối: "Nếu không có người phê duyệt CỤ THỂ trong context → nói rõ 'Không tìm thấy quy định phê duyệt tương ứng trong tài liệu'."

### #4 — context_recall = 0.500, avg = 0.793
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Chu kỳ đổi mật khẩu cụ thể
- **Got:** LLM đưa ra chu kỳ không có trong context
- **Worst metric:** context_recall (0.500)
- **Error Tree:** Output sai → Context có chu kỳ? → Chunk IT security có "mật khẩu" nhưng không có thời hạn → corpus thiếu data
- **Root cause:** Tài liệu bảo mật IT trong corpus mô tả yêu cầu mật khẩu (độ phức tạp) nhưng không có chu kỳ bắt buộc đổi. Dense retrieval tìm được chunk "mật khẩu" nhưng chunk đó không có câu trả lời.
- **Suggested fix:** Corpus audit — xác nhận 20 câu hỏi test_set có đáp án trong corpus. Câu này thuộc nhóm "unanswerable from corpus" → nên nói "Không tìm thấy" thay vì hallucinate.

### #5 — faithfulness = 0.667, avg = 0.802
- **Question:** Mentor và buddy của nhân viên mới có thể là cùng một người không?
- **Expected:** Quy định về việc mentor và buddy có thể kiêm nhiệm
- **Got:** LLM trả lời "có thể" hoặc "không thể" kèm điều kiện không có trong context
- **Worst metric:** faithfulness (0.667)
- **Error Tree:** Output → Context có quy định? → Chunk onboarding đề cập mentor/buddy nhưng không nêu rõ trường hợp kiêm nhiệm → LLM diễn giải
- **Root cause:** Câu hỏi về edge case không được nêu rõ trong tài liệu. LLM diễn giải thay vì nói "không rõ".
- **Suggested fix:** Thêm vào prompt: "Chỉ trả lời YES/NO nếu context NÊU RÕ RÀNG. Nếu context không đề cập trường hợp này → 'Tài liệu không quy định cụ thể về trường hợp này'."

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"

**Error Tree walkthrough:**
1. Output đúng? → **Không** — LLM trả lời mức phạt không có trong tài liệu
2. Context đúng? → **Một phần** — chunk về tạm ứng được retrieve nhưng không có mức phạt
3. Retrieval OK? → **Có** — Dense+HyQA tìm được chunk liên quan "tạm ứng thanh toán"
4. Corpus OK? → **Không** — Tài liệu quy chế tài chính không chứa điều khoản phạt trễ hạn
5. Fix ở bước: **Corpus** + **LLM Prompt** (từ chối khi không có số liệu cụ thể)

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Corpus Audit**: Xác nhận 20 câu hỏi test_set có đáp án trong corpus. Ít nhất 3-4 câu hiện tại "unanswerable" (không có data). Loại bỏ hoặc bổ sung data cho các câu này.
- **Confidence-based rejection**: Nếu reranker scores thấp (max_score < threshold) → từ chối trả lời thay vì hallucinate. Dự kiến tăng faithfulness thêm ~0.05.
- **Query expansion**: Với multi-hop questions (cần cross-reference nhiều tài liệu), tự động expand query thành 2-3 sub-queries và merge results.
