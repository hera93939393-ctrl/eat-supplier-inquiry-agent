"""app.py — 공급업체 문의 응대 에이전트 데모 화면 (Streamlit)."""
import streamlit as st

from agent import run

st.set_page_config(page_title="공급업체 문의 응대 에이전트", page_icon="📋")
st.title("공공급식통합플랫폼 공급업체 문의 챗봇")
st.caption("서류등록 / 현장심사 / 절차·일정 / 자격제한 4개 카테고리 + 범위 밖 이첩")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("질문을 입력하세요", placeholder="예: 냉장고 온도는 몇 도로 유지해야 하나요?")

if st.button("질문하기") and question:
    with st.spinner("분류하고 근거를 찾는 중..."):
        result = run(question)
    st.session_state.history.insert(0, result)

for result in st.session_state.history:
    with st.container(border=True):
        st.markdown(f"**Q. {result['query']}**")
        st.markdown(f"**A.** {result['answer']}")

        cols = st.columns(2)
        with cols[0]:
            st.caption(f"분류된 카테고리: `{result['category']}`")
        with cols[1]:
            badge = "✅ 근거 확인됨" if result["grounded"] else "⚠️ 근거 부족 의심"
            st.caption(f"검증 결과: {badge}")
        if result.get("verify_reason"):
            st.caption(f"검증 사유: {result['verify_reason']}")

        if result["context"]:
            with st.expander("근거로 쓴 문서 내용 보기"):
                st.text(result["context"])
