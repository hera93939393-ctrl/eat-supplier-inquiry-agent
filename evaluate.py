"""evaluate.py — 도구호출 적절성 / 답변 적절성 두 지표를 측정한다.

지표 1 (tool_call_appropriateness): 실제로 선택한 카테고리가 기대 카테고리와 정확히 일치하면 1점.
지표 2 (answer_appropriateness): 필수사실을 모두 담고, 금지사항을 하나도 어기지 않으면 1점.
    표현의 일치가 아니라 "사실이 담겼는지"를 LLM 판정기(judge)로 확인한다.
"""
import json
from pathlib import Path

from agent import _safe_chat, get_client, run

GOLDENSET_PATH = Path(__file__).parent / "data" / "goldenset.json"
RESULTS_PATH = Path(__file__).parent / "data" / "eval_results.json"

JUDGE_PROMPT = """아래 "답변"이 "반드시 담아야 할 사실"을 모두 담고 있는지, "말하면 안 되는 것"을
하나라도 포함하는지 판정해라. 표현이 똑같을 필요는 없다 — 같은 사실을 의미하면 충족한 것으로 본다.

반드시 담아야 할 사실:
{must_include}

말하면 안 되는 것:
{must_not_include}

답변:
{answer}

반드시 아래 JSON 형식으로만 답해라:
{{"has_all_facts": true|false, "violates_forbidden": true|false, "reason": "한 문장 이유"}}
"""


def judge_answer(answer: str, must_include: list[str], must_not_include: list[str]) -> dict:
    client = get_client()
    prompt = JUDGE_PROMPT.format(
        must_include="\n".join(f"- {f}" for f in must_include),
        must_not_include="\n".join(f"- {f}" for f in must_not_include) or "(없음)",
        answer=answer,
    )
    content = _safe_chat(client, json_mode=True, messages=[{"role": "user", "content": prompt}])
    if content is None:
        return {"has_all_facts": False, "violates_forbidden": True, "reason": "judge API 오류"}
    try:
        return json.loads(content)
    except (json.JSONDecodeError, AttributeError):
        return {"has_all_facts": False, "violates_forbidden": True, "reason": "judge 응답 파싱 실패"}


def evaluate_item(item: dict) -> dict:
    result = run(item["question"])
    tool_call_ok = result["category"] == item["expected_category"]

    judge = judge_answer(result["answer"], item["must_include"], item["must_not_include"])
    answer_ok = bool(judge.get("has_all_facts")) and not bool(judge.get("violates_forbidden"))

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_category": item["expected_category"],
        "actual_category": result["category"],
        "answer": result["answer"],
        "grounded": result["grounded"],
        "tool_call_ok": tool_call_ok,
        "answer_ok": answer_ok,
        "judge_reason": judge.get("reason", ""),
    }


def evaluate_all() -> list[dict]:
    goldenset = json.loads(GOLDENSET_PATH.read_text(encoding="utf-8"))
    results = [evaluate_item(item) for item in goldenset["eval_set"]]
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def print_summary(results: list[dict]) -> None:
    n = len(results)
    tool_score = sum(r["tool_call_ok"] for r in results)
    answer_score = sum(r["answer_ok"] for r in results)

    print(f"도구호출 적절성: {tool_score}/{n} ({tool_score / n:.0%})")
    print(f"답변 적절성:     {answer_score}/{n} ({answer_score / n:.0%})")
    print()
    print("틀린 문항:")
    for r in results:
        if not r["tool_call_ok"] or not r["answer_ok"]:
            print(f"  #{r['id']} [{r['category']}] {r['question']}")
            print(f"    카테고리: 기대={r['expected_category']} 실제={r['actual_category']}")
            print(f"    답변 적절성: {r['answer_ok']} ({r['judge_reason']})")


if __name__ == "__main__":
    results = evaluate_all()
    print_summary(results)
