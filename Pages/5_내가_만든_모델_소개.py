import pandas as pd
import streamlit as st


# ============================================================
# NEURO FORCE | 05. 내가 만든 모델 소개
#
# - CSV, 데이터셋, 외부 API를 읽지 않습니다.
# - 사용자가 직접 다섯 문항과 전후 비교표를 작성합니다.
# - 입력 내용은 현재 Streamlit 세션의 화면에서만 사용합니다.
# - 파일 저장, 다운로드, 이메일 전송 기능이 없습니다.
# - main.py에서 페이지 설정과 사이드바 메뉴를 담당합니다.
# ============================================================

QUESTIONS = [
    ("무슨 데이터입니까", "data", "어떤 데이터이며, 무엇을 기록한 자료인지 적어 보세요."),
    ("무엇을 보고 맞힙니까", "features", "모델이 사용하는 입력 속성과 예측하려는 대상을 설명해 보세요."),
    ("무엇을 고쳤습니까", "changes", "전처리나 클래스 무게 조절 등 바꾼 부분을 적어 보세요."),
    ("이 모델을 검진센터가 사용해도 됩니까", "hospital", "그렇게 판단한 이유와 실제 의료 현장에서의 한계를 설명해 보세요."),
    ("이 모델이 못 하는 것 한 가지", "limits", "모델이 해결하지 못하는 문제를 구체적으로 한 가지 적어 보세요."),
]

METRIC_NAMES = [
    "정확도",
    "재현율",
    "F1",
    "찾아낸 환자",
    "헛짚은 사람",
]

RED = "#F34E67"
BLUE = "#386BFA"
GREEN = "#19B58A"
YELLOW = "#FFC83D"
PINK = "#E968C6"


# ============================================================
# 01. 이 페이지 전용 전대물 디자인
# HTML은 st.html()로만 출력합니다.
# ============================================================

st.html("""
<style>
.nf5-hero {
    position: relative;
    overflow: hidden;
    padding: 40px 42px 35px;
    border: 1px solid #DDE5FD;
    border-radius: 22px;
    background: linear-gradient(116deg, #F2F6FF 0%, #F0FCF8 48%, #FFF0F9 100%);
    margin: 12px 0 29px;
}
.nf5-hero:after {
    content: "✦";
    position: absolute;
    right: 25px;
    top: -64px;
    font-size: 220px;
    color: #E968C61D;
    pointer-events: none;
}
.nf5-kicker {
    position: relative;
    z-index: 1;
    color: #386BFA;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: .16em;
}
.nf5-title {
    position: relative;
    z-index: 1;
    margin: 15px 0 16px;
    color: #202943;
    font-size: clamp(35px, 5vw, 61px);
    line-height: 1.16;
    letter-spacing: -.065em;
    font-weight: 900;
}
.nf5-title span {
    background: linear-gradient(95deg, #F34E67, #E968C6, #386BFA);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.nf5-lead {
    position: relative;
    z-index: 1;
    max-width: 720px;
    color: #53627E;
    font-size: 15px;
    line-height: 1.85;
}
.nf5-bars {
    display: flex;
    gap: 6px;
    max-width: 280px;
    margin-top: 26px;
}
.nf5-bars i {
    flex: 1;
    height: 8px;
    border-radius: 50px;
}
.nf5-section {
    display: flex;
    gap: 15px;
    align-items: baseline;
    border-top: 1px solid #DDE4F2;
    margin: 28px 0 7px;
    padding-top: 18px;
}
.nf5-section b {
    font-size: 12px;
    color: #386BFA;
}
.nf5-section span {
    font-size: 23px;
    font-weight: 850;
    letter-spacing: -.04em;
    color: #202943;
}
.nf5-description {
    color: #69758B;
    font-size: 14px;
    line-height: 1.8;
    margin: 0 0 22px;
}
.nf5-pill {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 6px;
    background: #EEF3FF;
    color: #386BFA;
    font-size: 11px;
    letter-spacing: .08em;
    font-weight: 850;
}
.nf5-question {
    padding: 15px 19px;
    background: #FFFFFF;
    border: 1px solid #E1E7F3;
    border-left: 4px solid var(--stripe);
    border-radius: 0 11px 11px 0;
    margin: 21px 0 9px;
    color: #202943;
    font-size: 16px;
    font-weight: 800;
}
.nf5-help {
    color: #69758B;
    font-size: 12px;
    margin-bottom: 9px;
}
.nf5-note {
    background: #FFF9E9;
    border: 1px solid #F1E2B7;
    padding: 16px 19px;
    border-radius: 12px;
    color: #62563D;
    font-size: 13px;
    line-height: 1.85;
    margin: 17px 0 24px;
}
.nf5-report-head {
    padding: 25px 27px;
    border-radius: 15px;
    background: linear-gradient(112deg, #EBF1FF, #FCEEFF);
    border: 1px solid #DFE5FB;
    color: #202943;
    font-size: 25px;
    font-weight: 900;
    letter-spacing: -.05em;
    margin-bottom: 21px;
}
.nf5-report-head small {
    display: block;
    font-size: 11px;
    color: #687998;
    font-weight: 800;
    letter-spacing: .15em;
    margin-bottom: 9px;
}
.nf5-report-card {
    padding: 20px 23px;
    margin: 0 0 13px;
    border-radius: 13px;
    background: #FFFFFF;
    border: 1px solid #DEE6F3;
    border-top: 4px solid var(--stripe);
    box-shadow: 0 6px 20px #354B770A;
}
.nf5-report-title {
    color: #202943;
    font-size: 15px;
    font-weight: 850;
    margin-bottom: 12px;
}
.nf5-report-text {
    color: #46536B;
    font-size: 14px;
    line-height: 1.9;
    white-space: normal;
    overflow-wrap: anywhere;
}
.nf5-unfilled {
    color: #9A6678;
    font-style: italic;
    font-size: 13px;
}
.nf5-footer {
    margin-top: 47px;
    padding-top: 20px;
    border-top: 1px solid #DFE5F1;
    font-size: 11px;
    letter-spacing: .1em;
    color: #8691A3;
}
@media(max-width: 700px) {
    .nf5-hero { padding: 27px 23px; }
    .nf5-title { font-size: 36px; }
    .nf5-report-head { padding: 20px; }
}
</style>
""")


# ============================================================
# 02. 표시용 함수
# 사용자 글은 HTML 특수문자를 이스케이프하여 출력합니다.
# ============================================================

def escape_text(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def section(number, title, description):
    st.html(
        '<div class="nf5-section">'
        f'<b>{escape_text(number)}</b>'
        f'<span>{escape_text(title)}</span>'
        '</div>'
        f'<div class="nf5-description">{escape_text(description)}</div>'
    )


def is_written(value):
    return value is not None and bool(str(value).strip())


def preview_card(number, title, content, color):
    if is_written(content):
        rendered = escape_text(content.strip()).replace("\n", "<br>")
    else:
        rendered = '<span class="nf5-unfilled">적지 않았음</span>'

    st.html(
        f'<div class="nf5-report-card" style="--stripe:{color}">'
        f'<div class="nf5-report-title">{number}. {escape_text(title)}</div>'
        f'<div class="nf5-report-text">{rendered}</div>'
        '</div>'
    )


COLORS = [RED, BLUE, GREEN, YELLOW, PINK]


# ============================================================
# 03. 상단 제목
# ============================================================

st.html("""
<div class="nf5-hero">
    <div class="nf5-kicker">06 / NEURO FORCE · FINAL REPORT</div>
    <div class="nf5-title">
        나만의 모델,<br>
        <span>최종 소개 작전!</span>
    </div>
    <div class="nf5-lead">
        앞 페이지에서 탐구한 결과를 직접 정리하는 마지막 미션입니다.
        다섯 가지 질문에 답하고, 고치기 전과 고친 뒤의 수치를 기록한 다음
        한 장으로 모아 확인하세요.
    </div>
    <div class="nf5-bars">
        <i style="background:#F34E67"></i>
        <i style="background:#386BFA"></i>
        <i style="background:#19B58A"></i>
        <i style="background:#FFC83D"></i>
        <i style="background:#E968C6"></i>
    </div>
</div>
""")


# ============================================================
# 04. 두 탭
# 두 탭의 내용은 같은 실행에서 모두 계산되므로,
# 미리보기가 입력한 내용을 바로 읽을 수 있습니다.
# ============================================================

write_tab, preview_tab = st.tabs([
    "✍️ 채워 넣기",
    "📋 한 장으로 모아 보기",
])


with write_tab:
    section(
        "01",
        "다섯 가지 질문에 답하기",
        "각 질문 아래에 자유롭게 여러 줄로 작성하세요.",
    )

    for number, (title, name, hint) in enumerate(QUESTIONS, start=1):
        st.html(
            f'<div class="nf5-question" style="--stripe:{COLORS[number - 1]}">'
            f'{number:02d} / {escape_text(title)}'
            '</div>'
            f'<div class="nf5-help">{escape_text(hint)}</div>'
        )

        st.text_area(
            title,
            key=f"nf5_answer_{name}",
            height=125,
            placeholder="여기에 직접 작성하세요.",
            label_visibility="collapsed",
        )

    section(
        "02",
        "고치기 전과 고친 뒤 비교표",
        "전처리 페이지에서 확인한 값을 직접 옮겨 적으세요. 자동으로 데이터를 읽지 않습니다.",
    )

    st.html(
        '<div class="nf5-note">'
        '<b>입력 방법</b> · 각 빈 칸을 클릭해 직접 적으세요. '
        '비율은 예: 92.5%, 사람 수는 예: 18명처럼 자유롭게 적을 수 있습니다. '
        '앞 페이지의 수치를 자동으로 가져오거나 전송하지 않습니다.'
        '</div>'
    )

    # 언제나 동일한 빈 표를 기본 입력으로 사용합니다.
    # st.data_editor가 이 페이지의 세션 안에서 편집 상태를 기억합니다.
    empty_table = pd.DataFrame({
        "지표": METRIC_NAMES,
        "고치기 전": [""] * len(METRIC_NAMES),
        "고친 뒤": [""] * len(METRIC_NAMES),
    })

    edited_table = st.data_editor(
        empty_table,
        key="nf5_metrics_editor",
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["지표"],
        column_config={
            "지표": st.column_config.TextColumn(
                "지표",
                width="medium",
            ),
            "고치기 전": st.column_config.TextColumn(
                "고치기 전 · 직접 입력",
                width="large",
            ),
            "고친 뒤": st.column_config.TextColumn(
                "고친 뒤 · 직접 입력",
                width="large",
            ),
        },
    )

    answers_count = sum(
        is_written(st.session_state.get(f"nf5_answer_{name}", ""))
        for _, name, _ in QUESTIONS
    )

    metrics_count = sum(
        is_written(value)
        for column in ("고치기 전", "고친 뒤")
        for value in edited_table[column]
    )

    completed = answers_count + metrics_count
    total_fields = len(QUESTIONS) + 2 * len(METRIC_NAMES)

    st.progress(
        completed / total_fields,
        text=f"작성 진행률 · {completed} / {total_fields}칸",
    )

    st.info(
        "작성 내용은 이 화면의 현재 Streamlit 세션에만 유지됩니다. "
        "파일 저장이나 외부 전송 기능은 없으며, "
        "브라우저 연결이 끊기거나 새 세션이 시작되면 내용이 사라질 수 있습니다."
    )


with preview_tab:
    st.html(
        '<div class="nf5-report-head">'
        '<small>NEURO FORCE / FINAL MISSION SHEET</small>'
        '내가 만든 모델 소개'
        '</div>'
    )

    section(
        "01",
        "모델을 소개하는 다섯 가지 답변",
        "작성한 문장을 원래 질문 순서대로 보여 줍니다.",
    )

    for number, (title, name, _) in enumerate(QUESTIONS, start=1):
        answer = st.session_state.get(f"nf5_answer_{name}", "")
        preview_card(number, title, answer, COLORS[number - 1])

    section(
        "02",
        "모델을 고친 전후 비교",
        "작성하지 않은 값은 '적지 않았음'으로 표시합니다.",
    )

    # 입력 화면의 편집 결과를 미리보기로만 가져옵니다.
    # 디스크 파일, CSV, 외부 데이터베이스에는 저장하지 않습니다.
    preview_table = edited_table.copy()

    for column in ("고치기 전", "고친 뒤"):
        preview_table[column] = preview_table[column].map(
            lambda value: (
                str(value).strip()
                if pd.notna(value) and is_written(value)
                else "적지 않았음"
            )
        )

    st.dataframe(
        preview_table,
        hide_index=True,
        use_container_width=True,
    )

    st.html(
        '<div class="nf5-note">'
        '<b>한 장 요약 완료</b> · 이 화면은 입력 내용을 정리해 보여 주기만 합니다. '
        '저장·다운로드·공유 버튼이나 외부 전송 기능은 제공하지 않습니다.'
        '</div>'
    )


st.html(
    '<div class="nf5-footer">'
    'NEURO FORCE · MY MODEL INTRODUCTION · EDUCATIONAL USE ONLY'
    '</div>'
)
