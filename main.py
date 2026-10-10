"""뇌졸중 예측 실습실 — 첫 화면: 데이터 소개와 데이터 사전 작성."""

import pandas as pd
import streamlit as st


APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
EXPECTED_COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi",
    "smoking_status", "stroke",
]
BINARY_COLUMNS = {"hypertension", "heart_disease", "stroke"}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* 본문 너비와 여백 */
    .block-container {
        max-width: 1260px;
        padding-top: 2.1rem;
        padding-bottom: 4rem;
    }
    [data-testid="stHeader"] { background: transparent; }
    .hero {
        background: linear-gradient(122deg, #132f52 0%, #1b5473 52%, #087e87 100%);
        color: #ffffff;
        padding: 2.25rem 2.35rem;
        border-radius: 22px;
        box-shadow: 0 16px 38px rgba(13, 53, 78, 0.16);
        margin-bottom: 1.6rem;
    }
    .hero-kicker {
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.15em;
        color: #abeae8;
        margin-bottom: 0.65rem;
    }
    .hero-title {
        font-size: clamp(1.7rem, 3.2vw, 2.6rem);
        line-height: 1.28;
        font-weight: 800;
        letter-spacing: -0.055em;
        margin: 0 0 0.75rem;
        color: #ffffff;
    }
    .hero-description {
        font-size: 1rem;
        line-height: 1.8;
        color: #e0f2f5;
        margin: 0;
        max-width: 800px;
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        color: #0c8c94;
        margin: 1.9rem 0 0.3rem;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0 0 0.4rem;
        color: inherit;
    }
    .section-desc {
        opacity: 0.72;
        font-size: 0.94rem;
        margin: 0 0 1.05rem;
    }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(103, 145, 159, 0.22);
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 7px 20px rgba(10, 55, 74, 0.045);
        min-height: 132px;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 700;
        opacity: 0.8;
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.75rem, 2.4vw, 2.4rem);
        font-weight: 800;
        letter-spacing: -0.055em;
    }
    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
        border-radius: 12px;
        overflow: hidden;
    }
    div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 42px;
    }
    .footnote {
        font-size: 0.84rem;
        opacity: 0.68;
        margin-top: 0.5rem;
    }
    @media (max-width: 720px) {
        .block-container { padding-top: 1rem; }
        .hero { padding: 1.6rem 1.3rem; border-radius: 17px; }
        .hero-description { font-size: 0.94rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_stroke_data() -> pd.DataFrame:
    """원격 UTF-8 CSV를 읽고 실습에 필요한 열 구성을 확인한다."""
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    missing_columns = [name for name in EXPECTED_COLUMNS if name not in data.columns]
    if missing_columns:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing_columns))
    return data


def describe_value_type(series: pd.Series, column_name: str) -> str:
    """학습자가 알기 쉬운 값 종류 설명."""
    if column_name == "id":
        return "식별자 · 정수"
    if column_name in BINARY_COLUMNS:
        return "이진형 · 0 / 1"
    if pd.api.types.is_numeric_dtype(series):
        return "수치형 · 정수" if pd.api.types.is_integer_dtype(series) else "수치형 · 실수"
    count = series.nunique(dropna=True)
    return f"범주형 · {count:,}종"


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">DATA SCIENCE · INTRODUCTION</div>
      <h1 class="hero-title">🧠 뇌졸중 예측 실습실</h1>
      <p class="hero-description">
        사람들의 건강·생활 습관 데이터를 살펴보고, 뇌졸중 발생 여부를 예측하는
        데이터 과학 실습의 첫걸음입니다. 오늘은 먼저 데이터의 규모, 열의 의미,
        빈 값을 확인해 봅니다.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("실습 데이터를 불러오는 중입니다..."):
        df = load_stroke_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 CSV 주소를 확인해 주세요.")
    with st.expander("오류 상세 보기"):
        st.code(str(exc))
    if st.button("🔄 데이터 다시 불러오기"):
        load_stroke_data.clear()
        st.rerun()
    st.stop()

person_count = len(df)
column_count = len(df.columns)
stroke_count = int(df["stroke"].eq(1).sum())
stroke_rate = (stroke_count / person_count * 100) if person_count else 0.0

st.markdown('<div class="section-label">01 · OVERVIEW</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">한눈에 보는 데이터</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="medium")
with col1:
    st.metric("👥 전체 사람 수", f"{person_count:,}명")
with col2:
    st.metric("🧾 전체 열 개수", f"{column_count:,}개")
with col3:
    st.metric("🧠 뇌졸중 발생 (stroke = 1)", f"{stroke_count:,}명")
with col4:
    st.metric("📊 뇌졸중 발생 비율", f"{stroke_rate:.2f}%")

st.caption("💡 발생 비율 = (stroke가 1인 사람 수 ÷ 전체 사람 수) × 100")

st.markdown('<div class="section-label">02 · DATA DICTIONARY</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">열 이름과 값 알아보기</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    '교재를 참고해 <b>우리말 뜻</b> 칸을 직접 채워 보세요. 다른 칸은 원본 데이터에서 자동으로 계산됩니다.'
    '</div>',
    unsafe_allow_html=True,
)

# 편집 중 화면이 다시 실행되어도 입력한 우리말 뜻을 세션 내에서 유지한다.
if "column_meanings" not in st.session_state:
    st.session_state.column_meanings = {name: "" for name in df.columns}

column_info = pd.DataFrame(
    [
        {
            "열 이름": name,
            "우리말 뜻": st.session_state.column_meanings.get(name, ""),
            "값의 종류": describe_value_type(df[name], name),
            "빈 값 개수": int(df[name].isna().sum()),
        }
        for name in df.columns
    ]
)

edited_info = st.data_editor(
    column_info,
    key="column_dictionary_editor",
    hide_index=True,
    use_container_width=True,
    height=464,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_order=["열 이름", "우리말 뜻", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", width="medium"),
        "우리말 뜻": st.column_config.TextColumn(
            "✏️ 우리말 뜻 (직접 입력)",
            width="large",
            help="교재를 보고 여기에 직접 입력하세요.",
            max_chars=120,
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="medium"),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수", format="%d", width="small"
        ),
    },
)

for row in edited_info.to_dict("records"):
    value = row.get("우리말 뜻", "")
    st.session_state.column_meanings[row["열 이름"]] = "" if pd.isna(value) else str(value)

completed_count = sum(bool(v.strip()) for v in st.session_state.column_meanings.values())
st.progress(
    completed_count / column_count if column_count else 0.0,
    text=f"우리말 뜻 작성 현황 · {completed_count}/{column_count}개 완료",
)

st.download_button(
    "⬇️ 작성한 열 설명표 CSV 저장",
    data=edited_info.to_csv(index=False).encode("utf-8-sig"),
    file_name="stroke_열설명표.csv",
    mime="text/csv",
    help="입력한 우리말 뜻을 포함한 표를 저장합니다. 엑셀에서도 한글이 읽히도록 저장됩니다.",
)

missing_bmi = int(df["bmi"].isna().sum())
if missing_bmi:
    st.info(
        f"🔎 **빈 값 관찰** · `bmi` 열에는 빈 값이 **{missing_bmi:,}개** 있습니다. "
        "빈 값은 임의로 채우지 않고 그대로 표시합니다."
    )

st.markdown('<div class="section-label">03 · PREVIEW</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">데이터 처음 5줄</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">원본 데이터의 열 순서와 값을 바꾸지 않고 맨 앞 다섯 행만 보여 줍니다. '
    '화면이 좁으면 표를 좌우로 스크롤하세요.</div>',
    unsafe_allow_html=True,
)
st.dataframe(df.head(5), hide_index=True, use_container_width=True, height=232)

st.markdown('<div class="section-label">04 · SOURCE</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">데이터 출처 기록</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">교재에 나온 데이터 출처를 아래에 직접 적어 보세요.</div>',
    unsafe_allow_html=True,
)
source_note = st.text_area(
    "데이터 출처",
    key="data_source_note",
    placeholder="교재를 확인한 후 출처를 입력하세요.",
    height=118,
    label_visibility="collapsed",
)
left, right = st.columns([1, 2], gap="medium")
with left:
    st.download_button(
        "⬇️ 출처 메모 TXT 저장",
        data=source_note.encode("utf-8-sig"),
        file_name="stroke_데이터출처.txt",
        mime="text/plain",
        disabled=not source_note.strip(),
    )
with right:
    st.caption("작성한 내용은 현재 접속 중인 세션에 보관됩니다. 종료하거나 새로고침하기 전에 파일로 저장해 주세요.")

st.divider()
st.caption("🧠 뇌졸중 예측 실습실 · 데이터 이해부터 시작하는 머신러닝 실습")
