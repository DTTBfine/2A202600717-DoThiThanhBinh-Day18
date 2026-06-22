from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        import os

        # Configure RAGAS LLM explicitly with llm_factory (ragas 0.4.x recommended API)
        # batch_size=1 prevents parallel bursts that trigger rate limits
        api_key = os.getenv("OPENAI_API_KEY", "")
        ragas_llm = None
        try:
            from openai import OpenAI as _OAI
            from ragas.llms import llm_factory
            ragas_llm = llm_factory("gpt-4o-mini", client=_OAI(api_key=api_key))
        except Exception:
            pass

        try:
            from langchain_openai import OpenAIEmbeddings
            if api_key:
                answer_relevancy.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        except Exception:
            pass

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        metrics_to_use = [faithfulness, context_precision, context_recall]
        has_answer_relevancy = answer_relevancy.embeddings is not None
        if has_answer_relevancy:
            metrics_to_use.append(answer_relevancy)

        eval_kwargs = {"metrics": metrics_to_use, "raise_exceptions": False, "batch_size": 4}
        if ragas_llm is not None:
            eval_kwargs["llm"] = ragas_llm
        result = evaluate(dataset, **eval_kwargs)
        df = result.to_pandas()

        # ragas 0.4.x does not include input columns in to_pandas() result
        rows = df.to_dict("records")
        per_question = [
            EvalResult(
                question=str(questions[i]),
                answer=str(answers[i]),
                contexts=list(contexts[i]),
                ground_truth=str(ground_truths[i]),
                faithfulness=float(rows[i].get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(rows[i].get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(rows[i].get("context_precision", 0.0) or 0.0),
                context_recall=float(rows[i].get("context_recall", 0.0) or 0.0),
            )
            for i in range(min(len(rows), len(questions)))
        ]

        available = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if c in df.columns]
        agg = df[available].mean()
        return {
            "faithfulness": float(agg.get("faithfulness", 0.0)),
            "answer_relevancy": float(agg.get("answer_relevancy", 0.0)),
            "context_precision": float(agg.get("context_precision", 0.0)),
            "context_recall": float(agg.get("context_recall", 0.0)),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — câu trả lời không bám sát context",
                         "Tighten system prompt, lower temperature, add citation requirement"),
        "context_recall": ("Missing relevant chunks — retrieval bỏ sót thông tin quan trọng",
                           "Improve chunking strategy, add BM25 or expand dense search top-k"),
        "context_precision": ("Too many irrelevant chunks — noise trong retrieved context",
                              "Add cross-encoder reranking, tighter top-k, or metadata filter"),
        "answer_relevancy": ("Answer doesn't address the question — câu trả lời lạc đề",
                             "Improve prompt template, ensure question is in context window"),
    }

    if not eval_results:
        return []

    scored = []
    for r in eval_results:
        avg = (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(metrics, key=lambda m: metrics[m])
        scored.append((avg, r, worst_metric, metrics[worst_metric]))

    scored.sort(key=lambda x: x[0])
    bottom = scored[:bottom_n]

    results = []
    for avg, r, worst_metric, worst_score in bottom:
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        results.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "avg_score": round(avg, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return results


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
