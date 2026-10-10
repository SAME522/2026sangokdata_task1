
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="데이터 소개 | NEURO LAB",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

SOURCE = """출처: Kaggle Stroke Prediction Dataset
게시자: fedesoriano
원본 주소: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
원본 설명: (Confidential Source) - Use only for educational purposes
용도: 교육용 데이터 분석 실습"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --paper: #F8F8F5;
    --ink: #222622;
    --sub: #697068;
    --line: #E2E4DE;
    --blue: #3559D5;
    --soft: #F0F1ED;
}
.stApp {
    background: var(--paper);
    color: var(--ink);
}
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', 'Manrope', sans-serif;
}
[data-testid="stHeader"] {
    background: var(--paper);
}
[data-testid="stSidebar"] {
    background: #F0F1ED;
    border-right: 1px solid #DFE2DA;
}
[data-testid="stSidebar"] * {
    color: #303630;
}
.block-container {
    max-width: 1240px;
    padding-top: 2.3rem;
    padding-bottom: 5rem;
}
h1, h2, h3, p {
    color: var(--ink);
}
h1, h2, h3 {
    letter-spacing: -.045em;
}
[data-testid="stMarkdownContainer"] p {
    line-height: 1.75;
}
.topline {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    border-bottom: 1px solid #D7DBD3;
    padding-bottom: 18px;
    margin-bottom: 54px;
}
.brand {
    font-size: 15px;
    font-weight: 800;
    letter-spacing: -.7px;
}
.edition {
    color: #777D75;
    font-size: 11px;
    letter-spacing: 1.5px;
    font-weight: 700;
}
.kicker {
    color: var(--blue);
    font-size: 11px;
    letter-spacing: 2.4px;
    font-weight: 800;
    margin-bottom: 19px;
}
.display {
    font-size: clamp(44px, 6vw, 83px);
    line-height: 1.12;
    letter-spacing: -.075em;
    font-weight: 800;
    max-width: 920px;
    margin-bottom: 23px;
}
.display span {
    color: var(--blue);
}
.lead {
    max-width: 650px;
    color: #666E66;
    font-size: 16px;
    line-height: 1.95;
}
.hero-rule {
    height: 1px;
    background: #DADDD6;
    margin: 45px 0 28px;
}
.section-head {
    display: flex;
    align-items: baseline;
    gap: 18px;
    border-top: 1px solid #D7DBD3;
    padding-top: 20px;
    margin-top: 76px;
    margin-bottom: 15px;
}
.section-no {
    color: var(--blue);
    font-size: 12px;
    font-weight: 800;
}
.section-name {
    font-size: clamp(25px, 3vw, 36px);
    font-weight: 800;
    letter-spacing: -.055em;
}
.section-desc {
    color: #6D756C;
    font-size: 14px;
    margin-bottom: 25px;
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E4DE;
    border-radius: 5px;
    padding: 25px 23px;
    min-height: 145px;
}
[data-testid="stMetricLabel"] {
    color: #70776E;
    font-size: 13px;
}
[data-testid="stMetricValue"] {
    color: #242923;
    font-size: clamp(26px, 2.9vw, 43px);
    font-weight: 800;
    letter-spacing: -.055em;
}
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
    border: 1px solid #E0E3DB;
    border-radius: 5px;
    overflow: hidden;
}
.stButton button,
.stDownloadButton button {
    border-radius: 5px;
    font-weight: 700;
    border: 1px solid #C9D0C6;
}
.stButton button:hover,
.stDownloadButton button:hover {
    border-color: #3559D5;
    color: #3559D5;
}
[data-testid="stProgressBar"] > div > div {
    background: #3559D5;
}
.note {
    background: #EDF0E9;
    border-left: 3px solid #3559D5;
    padding: 17px 20px;
    color: #4E594F;
    font-size: 13px;
    line-height: 1.8;
    margin-top: 20px;
}
.footer {
    margin-top: 85px;
    border-top: 1px solid #D7DBD3;
    padding-top: 20px;
    color: #8B9189;
    font-size: 11px;
    letter-spacing: 1px;
}
@media(max-width:700px) {
    .block-container {padding-top: 1rem;}
    .topline {margin-bottom: 34px;}
    .display {font-size: 43px;}
    .section-head {margin-top: 48px;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    required = ["id", "age", "avg_glucose_level", "bmi", "stroke"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"필수 열 누락: {', '.join(missing)}")
    return data


def section(number, name, description):
    st.markdown(
        f"""
        <div class="section-head">
            <span class="section-no">{number}</span>
            <span class="section-name">{name}</span>
        </div>
        <div class="section-desc">{description}</div>
        """,
        unsafe_allow_html=True,
    )


def describe_values(series):
    values = series.dropna()
    if values.empty:
        return "값 없음"

    if pd.api.types.is_numeric_dtype(series) and values.nunique() > 2:
        return f"{values.min():g} — {values.max():g}"

    items = sorted(values.astype(str).unique())
    return f"{len(items)}가지 · " + " / ".join(items)


with st.sidebar:
    st.markdown("### NEURO / LAB")
    st.caption("데이터를 읽는 연습")
    st.divider()
    st.markdown("**01 — 데이터 소개**")
    st.caption("현재 페이지")
    st.markdown("**02 — 데이터 탐색**")
    st.caption("위쪽 기본 페이지 메뉴에서 이동")
    st.divider()
    st.caption("EDUCATIONAL EDITION · 2026")


st.markdown("""
<div class="topline">
    <span class="brand">NEURO / LAB</span>
    <span class="edition">DATA STUDY — VOL. 01</span>
</div>
<div class="kicker">01 / UNDERSTAND THE DATA</div>
<div class="display">
    건강 데이터를<br>
    <span>읽는 새로운 방식.</span>
</div>
<div class="lead">
    숫자는 그 자체로 답을 말하지 않습니다.
    데이터를 관찰하고, 의미를 이해하고,
    더 좋은 질문을 만드는 것부터 시작합니다.
</div>
<div class="hero-rule"></div>
""", unsafe_allow_html=True)

try:
    df = load_data()
except Exception as exc:
    st.error("데이터를 불러올 수 없습니다.")
    with st.expander("오류 상세"):
        st.code(str(exc))
    if st.button("다시 시도"):
        load_data.clear()
        st.rerun()
    st.stop()

total = len(df)
stroke_count = int(df["stroke"].eq(1).sum())
stroke_rate = stroke_count / total * 100 if total else 0

st.caption("DATASET AT A GLANCE")

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 관측 인원", f"{total:,}")
c2.metric("데이터 변수", f"{df.shape[1]:02d}")
c3.metric("뇌졸중 경험자", f"{stroke_count:,}")
c4.metric("뇌졸중 경험 비율", f"{stroke_rate:.2f}%")

st.markdown("""
<div class="note">
    이 데이터의 <b>stroke</b> 열은 뇌졸중 경험 여부를 나타냅니다.
    1은 경험 있음, 0은 경험 없음을 뜻합니다.
    비율은 전체 인원 중 1로 기록된 사람의 비중입니다.
</div>
""", unsafe_allow_html=True)


# =========================================================
# 데이터 사전
# =========================================================

section(
    "02",
    "변수를 이해하는 시간",
    "열 이름의 의미를 교재에서 찾아 직접 작성해 보세요.",
)

if "dictionary_meanings" not in st.session_state:
    st.session_state.dictionary_meanings = {
        col: "" for col in df.columns
    }

dictionary = pd.DataFrame([
    {
        "열 이름": col,
        "우리말 뜻": st.session_state.dictionary_meanings.get(col, ""),
        "값의 종류": describe_values(df[col]),
        "빈 값 개수": int(df[col].isna().sum()),
    }
    for col in df.columns
])

edited = st.data_editor(
    dictionary,
    key="dictionary_editor",
    hide_index=True,
    use_container_width=True,
    height=460,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn(width="medium"),
        "우리말 뜻": st.column_config.TextColumn(
            "직접 작성하는 우리말 뜻",
            width="large",
        ),
        "값의 종류": st.column_config.TextColumn(width="large"),
        "빈 값 개수": st.column_config.NumberColumn(format="%d"),
    },
)

for _, row in edited.iterrows():
    value = row["우리말 뜻"]
    st.session_state.dictionary_meanings[row["열 이름"]] = (
        "" if pd.isna(value) else str(value)
    )

completed = sum(
    bool(value.strip())
    for value in st.session_state.dictionary_meanings.values()
)

st.progress(
    completed / len(df.columns),
    text=f"작성 진행률  {completed} / {len(df.columns)}",
)

st.download_button(
    "작성한 데이터 사전 저장 ↗",
    data=edited.to_csv(index=False).encode("utf-8-sig"),
    file_name="neuro_dictionary.csv",
    mime="text/csv",
)


# =========================================================
# 데이터 미리보기
# =========================================================

section(
    "03",
    "원본을 먼저 살펴보기",
    "어떤 형태로 데이터가 기록되어 있는지 확인합니다.",
)

st.dataframe(
    df.head(5),
    hide_index=True,
    use_container_width=True,
)

missing_bmi = int(df["bmi"].isna().sum())

st.markdown(
    f"""
    <div class="note">
        <b>관찰 포인트</b><br>
        BMI 열에는 빈 값이 <b>{missing_bmi:,}개</b> 있습니다.
        데이터가 비어 있다는 사실도 분석에 중요한 정보입니다.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("모든 열의 결측치 개수 확인"):
    st.dataframe(
        df.isna().sum().rename("빈 값 개수").reset_index(
            names="열 이름"
        ),
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# 출처
# =========================================================

section(
    "04",
    "데이터의 출처",
    "어디서 온 데이터인지 기록하고 한계를 이해합니다.",
)

source = st.text_area(
    "출처 기록",
    value=SOURCE,
    height=175,
    key="source_text",
)

st.download_button(
    "출처 메모 저장 ↗",
    data=source.encode("utf-8-sig"),
    file_name="neuro_source.txt",
    mime="text/plain",
)

st.markdown("""
<div class="note">
    <b>데이터 윤리</b><br>
    원본 데이터의 수집 과정과 대표성은 충분히 확인되지 않았습니다.
    이 자료는 교육용 분석에만 사용하며,
    분석 결과를 실제 의료 진단이나 치료 판단에 적용하지 않습니다.
</div>
""", unsafe_allow_html=True)

section(
    "05",
    "다음은, 패턴을 발견할 차례",
    "기본 페이지 메뉴에서 '탐색'을 선택하면 그래프와 분석표를 볼 수 있습니다.",
)

st.markdown("""
<div class="footer">
    NEURO / LAB &nbsp;&nbsp;·&nbsp;&nbsp;
    A STUDY IN DATA &nbsp;&nbsp;·&nbsp;&nbsp;
    EDUCATIONAL USE ONLY
</div>
""", unsafe_allow_html=True)
