"""app.py — 공급업체 문의 응대 에이전트 데모 화면 (Streamlit)."""
import streamlit as st

from agent import run

st.set_page_config(page_title="eaT 공급업체 챗봇", page_icon="📋", layout="centered")

CATEGORY_STYLE = {
    "document_review": {"label": "서류등록", "bg": "#DFF6E8", "fg": "#1E8A4C", "border": "#7ED9A8"},
    "site_inspection": {"label": "현장심사", "bg": "#EDE4FB", "fg": "#6D3FC4", "border": "#B8A6E8"},
    "procedure_timeline": {"label": "절차·일정", "bg": "#FFF6D9", "fg": "#B8860B", "border": "#FFD966"},
    "eligibility_restriction": {"label": "자격제한", "bg": "#FDE4E9", "fg": "#C43F6D", "border": "#F5A8BE"},
    "out_of_scope": {"label": "범위 밖(이첩)", "bg": "#EDEDED", "fg": "#666666", "border": "#CCCCCC"},
}

st.markdown(
    """
    <style>
    .stApp { background: #FAFAF7; }
    .eat-hero {
        background: linear-gradient(135deg, #DFF6E8 0%, #EDE4FB 50%, #FFF6D9 100%);
        border-radius: 20px;
        padding: 28px 28px 22px 28px;
        margin-bottom: 22px;
        border: 1px solid #00000010;
    }
    .eat-hero h1 {
        font-size: 1.7rem;
        font-weight: 800;
        white-space: nowrap;
        margin: 0 0 8px 0;
        color: #1A1A1A;
    }
    .eat-hero p { margin: 0; color: #444; font-size: 0.95rem; }
    .eat-pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
    .eat-pill {
        display: inline-block; padding: 5px 12px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 700; border: 1.5px solid;
    }
    .eat-card {
        background: #FFFFFF; border-radius: 18px; padding: 20px 22px;
        margin-bottom: 16px; box-shadow: 0 2px 10px #00000010; border: 1px solid #00000008;
    }
    .eat-q { font-weight: 800; font-size: 1.02rem; margin-bottom: 6px; color: #1A1A1A; }
    .eat-a { font-size: 0.96rem; line-height: 1.55; color: #262626; margin-bottom: 14px; }
    .eat-badge-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .eat-verify-ok { background: #DFF6E8; color: #1E8A4C; border: 1.5px solid #7ED9A8; }
    .eat-verify-bad { background: #FDE4E9; color: #C43F6D; border: 1.5px solid #F5A8BE; }
    .eat-reason { margin-top: 10px; font-size: 0.85rem; color: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="eat-hero">
      <h1>📋 eaT 공급업체 문의 챗봇</h1>
      <p>공공급식통합플랫폼 공급업체 등록 안내 — 근거 문서 기반 답변 + 검증</p>
      <div class="eat-pill-row">
    """
    + "".join(
        f'<span class="eat-pill" style="background:{s["bg"]};color:{s["fg"]};border-color:{s["border"]}">{s["label"]}</span>'
        for s in CATEGORY_STYLE.values()
    )
    + "</div></div>",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("질문을 입력하세요", placeholder="예: 냉장고 온도는 몇 도로 유지해야 하나요?")

if st.button("질문하기", type="primary") and question:
    with st.spinner("분류하고 근거를 찾는 중..."):
        result = run(question)
    st.session_state.history.insert(0, result)

for result in st.session_state.history:
    style = CATEGORY_STYLE.get(result["category"], CATEGORY_STYLE["out_of_scope"])
    verify_cls = "eat-verify-ok" if result["grounded"] else "eat-verify-bad"
    verify_label = "✅ 근거 확인됨" if result["grounded"] else "⚠️ 근거 부족 의심"

    st.markdown(
        f"""
        <div class="eat-card">
          <div class="eat-q">Q. {result['query']}</div>
          <div class="eat-a">{result['answer']}</div>
          <div class="eat-badge-row">
            <span class="eat-pill" style="background:{style['bg']};color:{style['fg']};border-color:{style['border']}">{style['label']}</span>
            <span class="eat-pill {verify_cls}">{verify_label}</span>
          </div>
          <div class="eat-reason">{result.get('verify_reason', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if result["context"]:
        with st.expander("근거로 쓴 문서 내용 보기"):
            st.text(result["context"])
