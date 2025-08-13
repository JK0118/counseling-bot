# -*- coding: utf-8 -*-
import sys
import os

# Set encoding for stdout and stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# app.py
import os
from datetime import datetime
import streamlit as st
from openai import OpenAI

# ---------------- 기본 설정 ----------------
st.set_page_config(page_title="Student Counseling Chatbot (GPT-4o)", page_icon="🧸", layout="centered")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

MODEL = "gpt-4o"  # 비용/속도 필요시 "gpt-4o-mini"
SYSTEM_PROMPT = """
너는 한국 초등학생을 상담하는 마음건강 챗봇이다.
원칙:
- 첫 인사는 “안녕! 나는 너의 마음을 도와주는 친구야. 무엇이 힘드니?”로 시작한다.
- 말투는 따뜻하고 쉬운 표현을 사용한다(짧은 문장, 어려운 용어 피하기).
- 흐름: 상황 묻기 → 느낌 확인 → 몸/생각 반응 묻기 → 작게 실천할 방법 정하기 → 마무리 약속.
- 학생이 쉽게 답하도록 선택지를 제안하거나 한 문장으로 답하도록 돕는다.
- 위험 신호(자해·타해 의도, 지속적 폭력/협박, 극심한 절망 등) 감지 시:
  (1) 공감, (2) 즉시 믿을 만한 어른에게 알리기 권유, (3) 연락처: 112(긴급) / 1388(청소년상담) / 1393(자살예방, 24시간).
- 의료/법률 조언은 하지 않고 전문기관·보호자·학교 연계를 권한다.
- 응답은 3~6문장 내외로 간결하게, 초등학생 수준으로.
"""

RISK_KEYWORDS = [
    "죽고", "죽고싶", "자해", "극단", "해치", "살고싶지", "사라지고", "숨고싶",
    "계속 때려", "협박", "위협", "따돌", "왕따"
]
def detect_risk(text: str) -> bool:
    return any(k in text for k in RISK_KEYWORDS)

def gpt_stream(messages):
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.7, stream=True
    )
    for chunk in resp:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ---------------- 세션 상태 ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "started" not in st.session_state:
    st.session_state.started = False
if "risk_flag" not in st.session_state:
    st.session_state.risk_flag = False
if "queued_user" not in st.session_state:
    st.session_state.queued_user = None    # 사이드바 예시 버튼으로 들어온 사용자 발화
if "run_assistant" not in st.session_state:
    st.session_state.run_assistant = False # 예시 클릭 시 즉시 응답 트리거

# ---------------- 사이드바(대화창 외부로 이동) ----------------
with st.sidebar:
    st.markdown("### 💡 답하기가 어려우면 눌러보세요")
    cols = st.columns(1)
    examples = [
        "학교에서 있었던 일 때문에 속상해요.",
        "친구랑 싸워서 마음이 복잡해요.",
        "집에서 요즘 자주 걱정돼요.",
        "요즘 잠이 잘 안 와요."
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            # 예시 발화를 '방금 입력한 사용자 메시지'로 큐잉하고 어시스턴트 응답 트리거
            st.session_state.queued_user = ex
            st.session_state.run_assistant = True

    st.divider()
    st.markdown("### 🗂️ 상담 요약 보기/내려받기")
    # 최근 30줄 요약
    lines = []
    for m in st.session_state.messages[-30:]:
        who = "상담봇" if m["role"] == "assistant" else "학생"
        lines.append(f"{who}: {m['content']}")
    summary_text = (
        f"[상담 요약 - {datetime.now().strftime('%Y-%m-%d %H:%M')}]\n"
        + "\n".join(lines)
        + "\n\n안전 체크: "
        + ("⚠️ 추가 보호자/전문기관 연계 필요" if st.session_state.risk_flag else "보고된 범위 내 위험 신호 없음")
    )
    st.code(summary_text, language="markdown")
    st.download_button("요약 텍스트 내려받기", summary_text, file_name="상담요약.txt", use_container_width=True)

    st.divider()
    st.caption("긴급: 112 / 청소년상담: 1388 / 자살예방: 1393(24시간)")

# ---------------- 메인 타이틀 ----------------
st.markdown("<h2 style='text-align:center'>🧸 학생 심리 상담 챗봇 (GPT-4o)</h2>", unsafe_allow_html=True)

# ---------------- 첫 인사(대화 시작 시 1회) ----------------
if not st.session_state.started:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "안녕! 나는 너의 마음을 도와주는 친구야. **무엇이 힘드니?**"
    })
    st.session_state.started = True

# ---------------- 큐잉된 예시 발화 처리(사이드바 버튼) ----------------
if st.session_state.queued_user:
    # 1) 사용자 메시지 즉시 표시
    with st.chat_message("user"):
        st.markdown(st.session_state.queued_user)
    st.session_state.messages.append({"role": "user", "content": st.session_state.queued_user})

    # 2) 위험 감지
    risk_now = detect_risk(st.session_state.queued_user)
    st.session_state.risk_flag = st.session_state.risk_flag or risk_now

    # 3) 시스템 프롬프트 구성
    system_prompt = SYSTEM_PROMPT + ("\n\n[안전모드] 사용자의 메시지에서 위기 징후가 감지됨. 안전 안내를 최우선으로 제공." if risk_now else "")

    payload = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    # 4) 스트리밍 응답 즉시 출력
    with st.chat_message("assistant"):
        stream_area = st.empty()
        acc = ""
        for token in gpt_stream(payload):
            acc += token
            stream_area.markdown(acc)
    st.session_state.messages.append({"role": "assistant", "content": acc})

    # 5) 큐 초기화
    st.session_state.queued_user = None
    st.session_state.run_assistant = False

# ---------------- 과거 메시지 렌더링 ----------------
if st.session_state.messages:
    st.markdown("---")
    st.markdown("### 📚 대화 기록")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# ---------------- 사용자 입력(실시간 상호작용 보장) ----------------
st.markdown("---")
st.markdown("### 💬 새로운 대화")
user_text = st.chat_input("여기에 답을 적어줘…")
if user_text:
    # 1) 사용자 메시지 즉시 화면에 출력
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.messages.append({"role": "user", "content": user_text})

    # 2) 위험 감지
    risk_now = detect_risk(user_text)
    st.session_state.risk_flag = st.session_state.risk_flag or risk_now

    # 3) 시스템 프롬프트 구성
    system_prompt = SYSTEM_PROMPT + ("\n\n[안전모드] 사용자의 메시지에서 위기 징후가 감지됨. 안전 안내를 최우선으로 제공." if risk_now else "")

    payload = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    # 4) 어시스턴트 스트리밍 응답을 즉시 같은 프레임에 출력
    with st.chat_message("assistant"):
        stream_area = st.empty()
        acc = ""
        for token in gpt_stream(payload):
            acc += token
            stream_area.markdown(acc)
    st.session_state.messages.append({"role": "assistant", "content": acc})
