"""agent.py — ①카테고리 판정 → ②근거조립 → ③답변 → ④검증, LangGraph 파이프라인.

확신이 없으면(out_of_scope로 판정되면) 근거조립/답변 단계를 건너뛰고 바로 이첩 메시지로 넘긴다.
"""
import json
from typing import Optional, TypedDict

import ollama
from langgraph.graph import END, StateGraph

from context import assemble_context
from prompts import (
    ANSWER_SYSTEM_PROMPT,
    CATEGORY_DESCRIPTIONS,
    CLASSIFY_SYSTEM_PROMPT,
    ESCALATION_MESSAGE,
    FEWSHOT_EXAMPLES,
    VERIFY_SYSTEM_PROMPT,
)

SERVER_HOST = "http://100.74.107.35:11434"
MODEL = "qwen3.5:9b"

VALID_CATEGORIES = list(CATEGORY_DESCRIPTIONS.keys())


def get_client() -> ollama.Client:
    return ollama.Client(host=SERVER_HOST)


def _safe_chat(client: ollama.Client, **kwargs) -> Optional[dict]:
    """ollama 호출을 감싸서, 서버가 500(예: token repeat limit)을 던지면 한 번 재시도하고
    그래도 실패하면 None을 반환한다 — 평가셋 30문항을 도는 중 한 건이 실패해도 전체가
    멈추지 않게 하기 위함이다."""
    for attempt in range(2):
        try:
            return client.chat(**kwargs)
        except ollama.ResponseError:
            if attempt == 1:
                return None
    return None


class AgentState(TypedDict):
    query: str
    category: Optional[str]
    context: str
    answer: str
    grounded: Optional[bool]
    verify_reason: str


def classify(state: AgentState) -> AgentState:
    categories_text = "\n".join(f"- {k}: {v}" for k, v in CATEGORY_DESCRIPTIONS.items())
    examples_text = "\n".join(
        f'  질문: "{ex["question"]}" -> category: "{ex["category"]}"' for ex in FEWSHOT_EXAMPLES
    )
    system = CLASSIFY_SYSTEM_PROMPT.format(categories=categories_text, examples=examples_text)

    client = get_client()
    response = _safe_chat(
        client,
        model=MODEL,
        format="json",
        think=False,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f'사용자 질문: "{state["query"]}"\n'
                    '반드시 다음 형식의 JSON으로만 답해라: {"category": "<카테고리 키>"}'
                ),
            },
        ],
    )
    category = None
    if response is not None:
        try:
            parsed = json.loads(response["message"]["content"])
            category = parsed.get("category")
        except (json.JSONDecodeError, AttributeError):
            category = None

    if category not in VALID_CATEGORIES:
        category = "out_of_scope"

    return {**state, "category": category}


def assemble(state: AgentState) -> AgentState:
    if state["category"] == "out_of_scope":
        return {**state, "context": ""}
    return {**state, "context": assemble_context(state["category"])}


def answer(state: AgentState) -> AgentState:
    if state["category"] == "out_of_scope":
        return {**state, "answer": ESCALATION_MESSAGE}

    client = get_client()
    prompt = ANSWER_SYSTEM_PROMPT.format(context=state["context"], question=state["query"])
    response = _safe_chat(
        client, model=MODEL, think=False, messages=[{"role": "user", "content": prompt}]
    )
    if response is None:
        return {**state, "answer": "일시적인 오류로 답변을 생성하지 못했습니다. 다시 시도해 주세요."}
    return {**state, "answer": response["message"]["content"]}


def verify(state: AgentState) -> AgentState:
    if state["category"] == "out_of_scope":
        return {**state, "grounded": True, "verify_reason": "이첩 메시지는 검증 대상이 아님"}

    client = get_client()
    prompt = VERIFY_SYSTEM_PROMPT.format(context=state["context"], answer=state["answer"])
    response = _safe_chat(
        client, model=MODEL, format="json", think=False, messages=[{"role": "user", "content": prompt}]
    )
    if response is None:
        return {**state, "grounded": False, "verify_reason": "API 오류로 검증 실패"}

    try:
        parsed = json.loads(response["message"]["content"])
        grounded = bool(parsed.get("grounded"))
        reason = parsed.get("reason", "")
    except (json.JSONDecodeError, AttributeError):
        grounded, reason = False, "검증 응답 파싱 실패"

    return {**state, "grounded": grounded, "verify_reason": reason}


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("assemble", assemble)
    graph.add_node("answer", answer)
    graph.add_node("verify", verify)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "assemble")
    graph.add_edge("assemble", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


_compiled_graph = _build_graph()


def run(query: str) -> AgentState:
    """①분류 → ②근거조립 → ③답변 → ④검증을 한 번에 실행한다."""
    initial: AgentState = {
        "query": query,
        "category": None,
        "context": "",
        "answer": "",
        "grounded": None,
        "verify_reason": "",
    }
    return _compiled_graph.invoke(initial)


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "사업자등록증 낼 때 뭘 확인해요?"
    result = run(query)
    print(json.dumps(result, ensure_ascii=False, indent=2))
