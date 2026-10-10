
"""뇌졸중 예측 실습실 | 01 데이터 소개"""

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type", "avg_glucose_level",
    "bmi", "smoking_status", "stroke",
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1260px; padding-top: 2rem; padding-bottom: 4rem;}
    [data-testid="stHeader"] {background: transparent;}
    .hero {
        position: relative; overflow: hidden;
        padding: 2.6rem 2.7rem; border-radius: 25px;
        background: radial-gradient(circle at 92% 12%, #287c98 0, transparent 38%),
                    linear-gradient(120deg, #101a3a 0%, #18486a 60%, #087e83 100%);
        box-shadow: 0 16px 38px rgba(14, 43, 76, .18);
        color: #fff; margin-bottom: 1.4rem;
    }
    .hero h1 {font-size: clamp(2rem, 3.5vw, 3rem); letter-spacing: -.055em;
              line-height: 1.22; color: #fff; margin: .5rem 0 .8rem;}
    .hero p {color: #e2f2fa; line-height: 1.85; max-width: 780px; margin: 0;}
    .eyebrow {font-size: .77rem; letter-spacing: .15em; font-weight: 800;
              color: #80ede3;}
    .tag {display: inline-block; margin: 1.2rem .4rem 0 0; padding: .35rem .8rem;
          background: #ffffff1e; border: 1px solid #ffffff45;
          border-radius: 999px; color: #fff; font-size: .78rem;}
    .section-kicker {font-size: .76rem; font-weight: 800; letter-spacing: .14em;
                     color: #0c9b95; margin: 2.05rem 0 .3rem;}
    .section-title {font-size: 1.5rem; font-weight: 800; letter-spacing: -.04em;
                    margin: 0 0 .4rem;}
    .section-sub {opacity: .76; font-size: .94rem; margin-bottom: 1.1rem;}
    [data-testid="stMetric"] {background: var(--secondary-background-color);
        border: 1px solid rgba(118,153,170,.22); border-radius: 18px;
        padding: 1.1rem 1.2rem; min-height: 126px;}
    [data-testid="stMetricValue"] {font-weight: 800; letter-spacing: -.04em;}
    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
        border-radius: 14px; overflow: hidden;}
    div.stDownloadButton > button, div.stButton > button {border-radius: 12px;}
    @media(max-width: 700px) {
        .block-container {padding-top: 1rem;}
        .hero {padding: 1.6rem 1.4rem; border-radius: 19px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    missing = [col for col in COLUMNS if col not in data.columns]
    if missing:
        raise ValueError("필수 열 누락: " + ", ".join(missing))
    return data


def value_kind(series, name):
    if name == "id":
        return "식별자 · 정수"
    if name in ("hypertension", "heart_disease", "stroke"):
        return "이진형 · 0 / 1"
    if pd.api.types.is_numeric_dtype(series):
        return "수치형 · 정수" if pd.api.types.is_integer_dtype(series) else "수치형 · 실수"
    return f"범주형 · {series.nunique(dropna=True)}종"


with st.sidebar:
    st.markdown("### 🧠 실습실 메뉴")
    st.caption("소개 화면에서 열을 익힌 다음 탐색 화면에서 그래프를 분석하세요.")
    st.page_link("main.py", label="01 · 데이터 소개", icon="🏠")
    st.page_link("pages/1_탐색.py", label="02 · 데이터 탐색", icon="🔎")
    st.divider()
    st.caption("데이터는 원격 CSV에서 UTF-8로 읽습니다.")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">STROKE DATA LAB / CHAPTER 01</div>
      <h1>🧠 뇌졸중 예측 실습실</h1>
      <p>건강·생활 습관 데이터로 배우는 데이터 과학.<br>
      먼저 데이터를 이해하고, 다음 페이지에서 분포와 집단별 차이를 직접 살펴보세요.</p>
      <span class="tag">01 데이터 소개</span>
      <span class="tag">02 시각화 탐색</span>
      <span class="tag">직접 작성하는 데이터 사전</span>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("UTF-8 데이터를 불러오는 중..."):
        df = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결 또는 데이터 주소를 확인하세요.")
    with st.expander("자세한 오류"):
        st.code(str(exc))
    if st.button("🔄 다시 시도"):
        load_data.clear()
        st.rerun()
    st.stop()

n = len(df)
num_stroke = int(df["stroke"].eq(1).sum())
rate = num_stroke / n * 100 if n else 0.0

st.markdown('<div class="section-kicker">01 / OVERVIEW</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">데이터 한눈에 보기</div>', unsafe_allow_html=True)
a, b, c, d = st.columns(4)
a.metric("👥 전체 사람 수", f"{n:,}명")
b.metric("🧾 열 개수", f"{len(df.columns):,}개")
c.metric("🧠 stroke = 1", f"{num_stroke:,}명")
d.metric("📈 발생 비율", f"{rate:.2f}%")
st.caption("계산 방법: 뇌졸중 발생 비율 = (stroke가 1인 사람 수 ÷ 전체 사람 수) × 100")

st.markdown('<div class="section-kicker">02 / DATA DICTIONARY</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">직접 완성하는 데이터 사전</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">교재를 보고 <b>우리말 뜻</b> 칸을 채워 보세요. '
    '나머지 항목은 원본 데이터에서 자동으로 계산됩니다.</div>',
    unsafe_allow_html=True,
)

if "column_meanings" not in st.session_state:
    st.session_state.column_meanings = {col: "" for col in df.columns}

table = pd.DataFrame([
    {
        "열 이름": col,
        "우리말 뜻": st.session_state.column_meanings.get(col, ""),
        "값의 종류": value_kind(df[col], col),
        "빈 값 개수": int(df[col].isna().sum()),
    }
    for col in df.columns
])

edited = st.data_editor(
    table,
    key="column_dictionary_editor",
    hide_index=True,
    use_container_width=True,
    height=465,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", width="medium"),
        "우리말 뜻": st.column_config.TextColumn("✏️ 우리말 뜻", width="large"),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="medium"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
)

for record in edited.to_dict("records"):
    val = record["우리말 뜻"]
    st.session_state.column_meanings[record["열 이름"]] = "" if pd.isna(val) else str(val)

complete = sum(bool(v.strip()) for v in st.session_state.column_meanings.values())
st.progress(complete / len(df.columns), text=f"열 뜻 작성 진행률 · {complete}/{len(df.columns)}개")
st.download_button(
    "⬇️ 열 설명표 CSV 저장",
    data=edited.to_csv(index=False).encode("utf-8-sig"),
    file_name="stroke_열설명표.csv",
    mime="text/csv",
)
missing_bmi = int(df["bmi"].isna().sum())
st.info(f"💡 bmi의 빈 값: **{missing_bmi:,}개**. 다음 탐색 페이지에서 빈 값이 있는 집단의 뇌졸중 비율도 비교합니다.")

st.markdown('<div class="section-kicker">03 / RAW DATA</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">원본 데이터 앞 5줄</div>', unsafe_allow_html=True)
st.caption("행과 열의 원래 순서 및 값을 유지합니다. 표가 넓으면 좌우로 스크롤하세요.")
st.dataframe(df.head(5), hide_index=True, use_container_width=True, height=226)

st.markdown('<div class="section-kicker">04 / SOURCE NOTE</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">데이터 출처 직접 기록</div>', unsafe_allow_html=True)
st.caption("교재에 나온 출처 내용을 아래 빈칸에 입력하세요.")
note = st.text_area(
    "데이터 출처 기록란",
    key="data_source_note",
    placeholder="교재를 참고해 데이터 출처를 적어 보세요.",
    height=115,
    label_visibility="collapsed",
)
st.download_button(
    "⬇️ 출처 메모 TXT 저장",
    data=note.encode("utf-8-sig"),
    file_name="stroke_데이터출처.txt",
    mime="text/plain",
    disabled=not note.strip(),
)
st.caption("입력 내용은 현재 브라우저 세션에 보관됩니다. 중요한 내용은 다운로드해 두세요.")

st.divider()
left, right = st.columns([2, 1])
with left:
    st.markdown("**다음 단계: 나이·혈당·고혈압·심장병·흡연 상태 탐색하기**")
    st.caption("탐색 페이지에서 필터를 바꾸고 그래프 위에 마우스를 올려 상세 정보를 확인하세요.")
with right:
    st.page_link("pages/1_탐색.py", label="🔎 탐색 페이지로 이동 →", use_container_width=True)
