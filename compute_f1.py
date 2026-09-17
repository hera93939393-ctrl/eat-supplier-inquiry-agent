"""compute_f1.py — data/eval_results.json으로부터 카테고리 분류의 accuracy/macro-F1과
답변 통과율(①/②)을 계산한다. evaluate.py 실행 후 한 번 돌리는 보조 스크립트."""
import json
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "data" / "eval_results.json"
CATEGORIES = ["document_review", "site_inspection", "procedure_timeline", "eligibility_restriction", "out_of_scope"]


def compute_classification_metrics(results: list[dict]) -> dict:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    correct = 0
    for r in results:
        expected, actual = r["expected_category"], r["actual_category"]
        if expected == actual:
            correct += 1
            tp[expected] += 1
        else:
            fn[expected] += 1
            fp[actual] += 1

    accuracy = correct / len(results)

    per_class = {}
    f1_scores = []
    for c in CATEGORIES:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "n": tp[c] + fn[c]}
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}


def main():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    n = len(results)

    cls_metrics = compute_classification_metrics(results)
    answer_pass = sum(r["answer_ok"] for r in results)

    print(f"① 의도 분류 정확도: {cls_metrics['accuracy']:.0%} ({sum(r['tool_call_ok'] for r in results)}/{n} 건), macro F1 {cls_metrics['macro_f1']:.3f}")
    print("  카테고리별:")
    for c, m in cls_metrics["per_class"].items():
        print(f"    {c}: precision={m['precision']:.2f} recall={m['recall']:.2f} f1={m['f1']:.2f} (n={m['n']})")
    print()
    print(f"② 1턴 답변 통과율: {answer_pass}/{n} ({answer_pass / n:.0%})")


if __name__ == "__main__":
    main()
