# eaT 공급업체 문의 응대 에이전트

공공급식통합플랫폼 공급업체 등록 문의(서류등록/현장심사/절차·일정/자격제한)에 실제 공개
안내자료를 근거로 답하고, 문서에 없는 내용은 지어내지 않고 담당 부서로 넘기는 라우팅 에이전트.
①카테고리 판정 → ②근거 조립 → ③답변 → ④검증 4단계 LangGraph 파이프라인으로 구현했다.

**자세한 설계 근거, 실험 기록(11라운드), 실패 분석은 → [REPORT.md](./REPORT.md) 참고.**

## 최종 결과

| 지표 | 결과 |
|---|---|
| 의도 분류 정확도 | 100% (30/30), macro F1 1.000 |
| 1턴 답변 통과율 | 100% (30/30) |

## 빠른 실행

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env

python agent.py "냉장고 온도는 몇 도로 유지해야 하나요?"   # 단발 질문
python evaluate.py                                          # 평가셋 30문항 채점
python compute_f1.py                                        # 정확도·macro F1·혼동행렬
python validate_grader.py                                   # 채점기 자체 검증
streamlit run app.py                                        # 데모 화면
```

## 구조

```
docs/            근거 문서 4종 (카테고리별로 미리 쪼개둠)
data/            평가셋(goldenset.json)과 채점/검증 결과
prompts.py       분류·답변·검증 프롬프트
context.py       카테고리 → 근거 문서 조립
agent.py         LangGraph 파이프라인
evaluate.py      두 지표(①②) 측정
compute_f1.py    정확도·macro F1·혼동행렬
validate_grader.py  채점기 자체 검증
app.py           Streamlit 데모
experiments/     라운드별 실험 결과 스냅샷
REPORT.md        전체 설계·실험·회고 리포트
```
