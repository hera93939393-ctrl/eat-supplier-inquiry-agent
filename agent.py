"""agent.py — ①카테고리 판정 → ②근거조립 → ③답변 → ④검증, LangGraph 파이프라인.

확신이 없으면(out_of_scope로 판정되면) 근거조립/답변 단계를 건너뛰고 바로 이첩 메시지로 넘긴다.
LLM은 OpenAI API(GPT)를 쓴다 — 이 프로젝트의 근거 문서는 전부 공개 안내자료라 외부 API 사용에
개인정보·보안 문제가 없고, 과제 특성상 로컬 모델을 고집할 필요가 없어 편의성과 성능을 택했다.
"""
import json
import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI

from context import assemble_context
from prompts import (
    ANSWER_SYSTEM_PROMPT,
    CATEGORY_DESCRIPTIONS,
    CLASSIFY_SYSTEM_PROMPT,
    ESCALATION_MESSAGE,
    FEWSHOT_EXAMPLES,
    VERIFY_SYSTEM_PROMPT,
)

load_dotenv()

MODEL = "gpt-4o"

VALID_CATEGORIES = list(CATEGORY_DESCRIPTIONS.keys())


def get_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _safe_chat(client: OpenAI, *, messages: list[dict], json_mode: bool = False) -> Optional[str]:
    """OpenAI 호출을 감싸서 실패하면 한 번 재시도하고, 그래도 실패하면 None을 반환한다 —
    평가셋 30문항을 도는 중 한 건(네트워크 오류 등)이 실패해도 전체가 멈추지 않게 하기 위함이다.

    temperature=0으로 고정한다 — 이걸 안 하면 같은 프롬프트도 실행마다 결과가 흔들려서,
    프롬프트를 바꾼 효과인지 단순 무작위성인지 구분할 수 없어진다."""
    kwargs = {"model": MODEL, "temperature": 0, "messages": messages}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(2):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception:
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
    content = _safe_chat(
        client,
        json_mode=True,
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
    if content is not None:
        try:
            category = json.loads(content).get("category")
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
    system = ANSWER_SYSTEM_PROMPT.format(context=state["context"])
    content = _safe_chat(
        client,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": state["query"]},
        ],
    )
    if content is None:
        return {**state, "answer": "일시적인 오류로 답변을 생성하지 못했습니다. 다시 시도해 주세요."}
    return {**state, "answer": content}


def verify(state: AgentState) -> AgentState:
    if state["category"] == "out_of_scope":
        return {**state, "grounded": True, "verify_reason": "이첩 메시지는 검증 대상이 아님"}

    client = get_client()
    prompt = VERIFY_SYSTEM_PROMPT.format(context=state["context"], answer=state["answer"])
    content = _safe_chat(client, json_mode=True, messages=[{"role": "user", "content": prompt}])
    if content is None:
        return {**state, "grounded": False, "verify_reason": "API 오류로 검증 실패"}

    try:
        parsed = json.loads(content)
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
