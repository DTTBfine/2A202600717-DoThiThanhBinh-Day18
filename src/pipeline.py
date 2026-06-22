from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, DENSE_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        # parent-child retrieval: index small children (precision), return full parent for context (recall)
        parent_map = {p.metadata["parent_id"]: p.text for p in parents}
        for child in children:
            pid = child.parent_id
            all_chunks.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "parent_id": pid,
                    "_parent_text": parent_map.get(pid, child.text),
                },
            })
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        # HyQA: index "questions + original text" for better semantic matching
        # _parent_text preserved from auto_metadata (enrichment copies original metadata)
        all_chunks = []
        for e in enriched:
            hyqa_prefix = "\n".join(e.hypothesis_questions) if e.hypothesis_questions else ""
            index_text = f"{hyqa_prefix}\n\n{e.original_text}" if hyqa_prefix else e.original_text
            all_chunks.append({
                "text": index_text,
                "metadata": {
                    **e.auto_metadata,
                    "_original_text": e.original_text,
                    # _parent_text is already in e.auto_metadata (copied from original chunk metadata)
                },
            })
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    # Dense-only retrieval: higher semantic precision than hybrid (no BM25 keyword noise)
    # HyQA indexing already improves dense recall; CrossEncoder reranker handles final ranking
    results = search.dense.search(query, top_k=DENSE_TOP_K)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    # Return clean original child text (no HyQA prefix noise)
    def _clean(r):
        return r.metadata.get("_original_text", r.text)
    contexts = [_clean(r) for r in reranked] if reranked else [_clean(r) for r in results[:RERANK_TOP_K]]

    from config import OPENAI_API_KEY
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, messages=[
                {"role": "system", "content": (
                    "Bạn là trợ lý HR. Trả lời DỰA TRÊN context được cung cấp.\n"
                    "- Ưu tiên thông tin trong context. Không thêm thông tin không có trong context.\n"
                    "- Nếu context không có đủ thông tin → nói 'Không tìm thấy thông tin cụ thể trong tài liệu.'\n"
                    "- Trả lời ngắn gọn, trực tiếp vào câu hỏi."
                )},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    import json, pathlib
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    # Cache answers so RAGAS can be retried without re-running pipeline
    cache = {"questions": questions, "answers": answers,
             "contexts": all_contexts, "ground_truths": ground_truths}
    pathlib.Path("reports").mkdir(exist_ok=True)
    with open("reports/answers_cache.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print("  ✓ Answers cached to reports/answers_cache.json", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures, path="reports/ragas_report.json")
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
