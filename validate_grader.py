"""validate_grader.py — 채점기(judge_answer) 자체를 검증한다.

"평가셋을 채점하기 전에, 채점기가 정답은 정답으로, 오답은 오답으로 제대로 가르는가"를
사람이 직접 만든 모범답안/오답 쌍으로 확인한다. 실제 모델 응답이 아니라 손으로 쓴 텍스트를
넣는다 — 모델 성능이 아니라 채점기 자체의 신뢰도를 보는 테스트이기 때문이다.
"""
import json
from pathlib import Path

from evaluate import judge_answer

GOLDENSET_PATH = Path(__file__).parent / "data" / "goldenset.json"

# (문항 id, 라벨, 답변 텍스트, 기대되는 answer_ok)
CASES = [
    (
        6, "모범답안(정답)",
        "등기부등본에 '법적 효력이 없다'는 문구가 있으면 승인이 불가합니다.",
        True,
    ),
    (
        6, "오답(금지사항 위반 — 반대로 답함)",
        "네, 그 문구가 있어도 괜찮습니다. 그대로 제출하시면 됩니다.",
        False,
    ),
    (
        6, "오답(필수사실 누락 — 회피만 함)",
        "그 부분은 제가 가진 자료로는 확인이 어렵습니다.",
        False,
    ),
    (
        19, "모범답안(정답)",
        "다른 사람 명의로 회사를 여러 개 운영하는 위장업체의 경우, 대표자는 입찰방해죄로, "
        "명의를 빌려준 사람은 입찰방해방조죄로 형사 처벌을 받을 수 있습니다.",
        True,
    ),
    (
        19, "오답(필수사실 누락 — 형사처벌 언급 없이 축소)",
        "이용약관 위반으로 일정기간 플랫폼 이용이 제한될 수 있습니다.",
        False,
    ),
]


def main():
    goldenset = json.loads(GOLDENSET_PATH.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in goldenset["eval_set"]}

    results = []
    for item_id, label, answer_text, expected_ok in CASES:
        item = by_id[item_id]
        judge = judge_answer(answer_text, item["must_include"], item["must_not_include"])
        actual_ok = bool(judge.get("has_all_facts")) and not bool(judge.get("violates_forbidden"))
        passed = actual_ok == expected_ok
        results.append(
            {
                "id": item_id,
                "label": label,
                "answer": answer_text,
                "expected_ok": expected_ok,
                "actual_ok": actual_ok,
                "grader_correct": passed,
                "judge_reason": judge.get("reason", ""),
            }
        )

    out_path = Path(__file__).parent / "data" / "grader_validation.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    n_pass = sum(r["grader_correct"] for r in results)
    with open("grader_validation_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"채점기 자체 검증: {n_pass}/{len(results)} 케이스에서 예상대로 판정함\n\n")
        for r in results:
            mark = "OK" if r["grader_correct"] else "FAIL"
            f.write(
                f"[{mark}] #{r['id']} {r['label']}\n"
                f"  답변: {r['answer']}\n"
                f"  기대={r['expected_ok']} 실제={r['actual_ok']} (사유: {r['judge_reason']})\n\n"
            )
    print(f"{n_pass}/{len(results)} passed - see grader_validation_summary.txt")


if __name__ == "__main__":
    main()
