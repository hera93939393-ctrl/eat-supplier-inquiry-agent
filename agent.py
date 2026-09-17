"""agent.py — ①-a 주제 판정(4개 중 강제선택) → ①-b 범위 판단(넘길지, 독립 판단) →
②근거조립 → ③답변 → ④검증, LangGraph 파이프라인.

"카테고리를 고르는 일"과 "확신이 없을 때 넘기는 판단"을 의도적으로 서로 다른 LLM 호출로 분리했다
— classify_category()는 out_of_scope라는 선택지 자체를 모르는 채로 4개 주제 중 하나를 강제로
고르고, check_scope()는 그 결과를 전혀 보지 않은 채 질문만 보고 "이걸 이 문서들로 답할 수 있는
성격인가"만 독립적으로 판단한다. 두 판단이 갈리면(주제는 골랐지만 범위 밖으로 판정) 최종적으로
범위 판단이 우선한다.

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
    SCOPE_CHECK_PROMPT,
    TOPIC_DESCRIPTIONS,
    VERIFY_SYSTEM_PROMPT,
)

load_dotenv()

MODEL = "gpt-4o"

VALID_CATEGORIES = list(CATEGORY_DESCRIPTIONS.keys())
VALID_TOPICS = list(TOPIC_DESCRIPTIONS.keys())


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
    topic: Optional[str]
    in_scope: Optional[bool]
    scope_reason: str
    category: Optional[str]
    context: str
    answer: str
    grounded: Optional[bool]
    verify_reason: str


def classify_category(state: AgentState) -> AgentState:
    """①-a: 4개 주제 중 하나를 강제로 고른다. out_of_scope라는 선택지 자체가 없다 —
    "넘길지 말지"는 이 함수가 몰라야 진짜 분리다."""
    categories_text = "\n".join(f"- {k}: {v}" for k, v in TOPIC_DESCRIPTIONS.items())
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
                    '반드시 다음 형식의 JSON으로만 답해라: {"category": "<4개 주제 중 하나>"}'
                ),
            },
        ],
    )
    topic = None
    if content is not None:
        try:
            topic = json.loads(content).get("category")
        except (json.JSONDecodeError, AttributeError):
            topic = None

    if topic not in VALID_TOPICS:
        topic = VALID_TOPICS[0]  # 강제 선택이 실패해도 out_of_scope로 새지 않게 안전한 기본값

    return {**state, "topic": topic}


def check_scope(state: AgentState) -> AgentState:
    """①-b: classify_category의 결과를 전혀 보지 않고, 질문만 보고 "넘겨야 하는가"를 독립
    판단한다. 이 판단이 최종 범위 밖 여부를 결정한다(주제는 그럴듯해도 여기서 걸리면 넘긴다)."""
    client = get_client()
    prompt = SCOPE_CHECK_PROMPT.format(question=state["query"])
    content = _safe_chat(client, json_mode=True, messages=[{"role": "user", "content": prompt}])

    in_scope, reason = False, "API 오류로 범위 판단 실패(안전하게 이첩)"
    if content is not None:
        try:
            parsed = json.loads(content)
            in_scope = bool(parsed.get("in_scope"))
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, AttributeError):
            in_scope, reason = False, "범위 판단 응답 파싱 실패(안전하게 이첩)"

    return {**state, "in_scope": in_scope, "scope_reason": reason}


def resolve_category(state: AgentState) -> AgentState:
    """두 독립 판단을 합친다 — 범위 판단이 false면 주제가 뭐였든 out_of_scope가 최종 결과다."""
    category = state["topic"] if state["in_scope"] else "out_of_scope"
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
    graph.add_node("classify_category", classify_category)
    graph.add_node("check_scope", check_scope)
    graph.add_node("resolve_category", resolve_category)
    graph.add_node("assemble", assemble)
    graph.add_node("answer", answer)
    graph.add_node("verify", verify)
    graph.set_entry_point("classify_category")
    graph.add_edge("classify_category", "check_scope")
    graph.add_edge("check_scope", "resolve_category")
    graph.add_edge("resolve_category", "assemble")
    graph.add_edge("assemble", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


_compiled_graph = _build_graph()


def run(query: str) -> AgentState:
    """①-a 주제판정 → ①-b 범위판단 → ①-c 합치기 → ②근거조립 → ③답변 → ④검증을 실행한다."""
    initial: AgentState = {
        "query": query,
        "topic": None,
        "in_scope": None,
        "scope_reason": "",
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
