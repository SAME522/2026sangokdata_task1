
import pandas as pd
import streamlit as st

# =========================================================
# NEURO LAB | 메인 화면
# =========================================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실 | NEURO LAB",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/main/data/stroke.csv"
)

REQUIRED_COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
]

DEFAULT_SOURCE = """출처: Kaggle Stroke Prediction Dataset
게시자: fedesoriano
원본 주소: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
데이터 설명: 건강 정보와 뇌졸중 경험 여부를 담은 교육용 데이터
참고: 데이터 수집 경위와 대표성은 별도 검증이 필요합니다.
"""

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 85% 0%, #18395c 0%, transparent 35%),
        radial-gradient(circle at 0% 75%, #0b353d 0%, transparent 32%),
        #080f1f;
    color: #e9f1ff;
}
[data-testid="stSidebar"] {
    background: #0d172b;
    border-right: 1px solid #273d59;
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1320px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
h1, h2, h3 {
    color: #f2f7ff !important;
}
.hero {
    padding: 48px;
    border-radius: 26px;
    background: linear-gradient(120deg, #122448, #173a58, #096c70);
    border: 1px solid #37627c;
    box-shadow: 0 20px 55px #00000035;
    margin-bottom: 28px;
}
.hero h1 {
    font-size: clamp(32px, 4vw, 52px);
    line-height: 1.18;
    font-weight: 850;
    margin: 15px 0;
}
.hero p {
    color: #d2e7f2;
    font-size: 16px;
    line-height: 1.9;
}
.eyebrow {
    color: #74f2df;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}
.pill {
    display: inline-block;
    padding: 8px 14px;
    margin: 12px 8px 0 0;
    border-radius: 100px;
    border: 1px solid #76e6da55;
    background: #ffffff14;
    color: #d0fff8;
    font-size: 12px;
}
.section {
    margin-top: 42px;
    margin-bottom: 18px;
}
.section small {
    color: #5cead8;
    font-weight: 800;
    letter-spacing: 2px;
}
.section h2 {
    margin: 7px 0;
    font-size: 27px;
    font-weight: 800;
}
.section p {
    color: #afc2d8;
    font-size: 14px;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #152640, #101b30);
    border: 1px solid #304764;
    border-radius: 20px;
    padding: 22px;
    min-height: 132px;
}
[data-testid="stMetricValue"] {
    color: #60eadb;
    font-weight: 850;
}
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
    border: 1px solid #304764;
    border-radius: 14px;
    overflow: hidden;
}
.stButton button,
.stDownloadButton button {
    border-radius: 12px;
}
.footer {
    margin-top: 48px;
    padding: 24px;
    text-align: center;
    color: #91a8c1;
    border-top: 1px solid #293d58;
    font-size: 12px;
}
@media (max-width: 700px) {
    .hero {padding: 27px 23px;}
    .block-container {padding-top: 1rem;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("필수 열 누락: " + ", ".join(missing))
    return df


def section(number, title, description):
    st.markdown(
        f"""
        <div class="section">
            <small>SECTION {number} / NEURO LAB</small>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def value_type(series):
    values = series.dropna()

    if values.empty:
        return "값 없음"

    if pd.api.types.is_numeric_dtype(series) and values.nunique() > 2:
        return f"숫자 {values.min():g} ~ {values.max():g}"

    kinds = sorted(values.astype(str).unique())
    return f"{len(kinds)}가지 · " + " · ".join(kinds)


# =========================================================
# 사이드바
# st.page_link 사용 안 함!
# Streamlit 기본 페이지 메뉴 사용
# =========================================================

with st.sidebar:
    st.markdown("## 🧠 NEURO LAB")
    st.caption("STROKE DATA SCIENCE STUDIO")
    st.divider()
    st.markdown("### 📍 현재 화면")
    st.success("01 · 데이터 소개")
    st.caption(
        "다른 화면으로 이동하려면 위쪽의 "
        "Streamlit 기본 페이지 메뉴에서 '탐색'을 선택하세요."
    )
    st.divider()
    st.markdown("### 🎯 학습 순서")
    st.write("① 데이터 구조 이해")
    st.write("② 분포 시각화")
    st.write("③ 집단별 차이 비교")
    st.write("④ 결측치와 흡연 상태 분석")


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">● NEURO LAB / DATA INTELLIGENCE</div>
    <h1>🧠 뇌졸중 예측<br>실습실</h1>
    <p>
        5,110명의 건강 기록으로 시작하는 데이터 사이언스.<br>
        데이터를 이해하고, 시각화하고, 숨겨진 패턴을 발견하세요.
    </p>
    <span class="pill">01 / DATA OVERVIEW</span>
    <span class="pill">INTERACTIVE LEARNING</span>
    <span class="pill">STREAMLIT + PYTHON</span>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 데이터 로딩
# =========================================================

try:
    with st.spinner("건강 데이터를 불러오는 중..."):
        df = load_data()

except Exception as exc:
    st.error("데이터를 불러오지 못했습니다.")

    with st.expander("오류 상세"):
        st.code(str(exc))

    if st.button("🔄 데이터 다시 불러오기"):
        load_data.clear()
        st.rerun()

    st.stop()


# =========================================================
# SECTION 01 — KPI
# =========================================================

section(
    "01",
    "데이터 인사이트",
    "데이터의 규모와 뇌졸중 발생 현황을 확인합니다.",
)

total = len(df)
stroke_count = int(df["stroke"].eq(1).sum())
stroke_rate = stroke_count / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)

c1.metric("👥 전체 사람 수", f"{total:,}명")
c2.metric("🗂 열 개수", f"{df.shape[1]}개")
c3.metric("🧠 뇌졸중 경험자", f"{stroke_count:,}명")
c4.metric("📈 뇌졸중 비율", f"{stroke_rate:.2f}%")

st.info(
    "💡 뇌졸중 비율 = "
    "(stroke가 1인 사람 수 ÷ 전체 사람 수) × 100"
)


# =========================================================
# SECTION 02 — 데이터 사전
# =========================================================

section(
    "02",
    "인터랙티브 데이터 사전",
    "교재를 보고 '우리말 뜻' 칸을 직접 작성하세요.",
)

if "meanings" not in st.session_state:
    st.session_state.meanings = {
        column: "" for column in df.columns
    }

dictionary = pd.DataFrame([
    {
        "열 이름": column,
        "우리말 뜻": st.session_state.meanings.get(column, ""),
        "값의 종류": value_type(df[column]),
        "빈 값 개수": int(df[column].isna().sum()),
    }
    for column in df.columns
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
        "열 이름": st.column_config.TextColumn(
            "열 이름", width="medium"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "✏️ 우리말 뜻", width="large"
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류", width="large"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수", format="%d"
        ),
    },
)

for _, row in edited.iterrows():
    value = row["우리말 뜻"]
    st.session_state.meanings[row["열 이름"]] = (
        "" if pd.isna(value) else str(value)
    )

completed = sum(
    bool(str(value).strip())
    for value in st.session_state.meanings.values()
)

st.progress(
    completed / len(df.columns),
    text=f"📝 데이터 사전 작성 진행률: {completed}/{len(df.columns)}",
)

if completed == len(df.columns):
    st.success("🏆 모든 열의 우리말 뜻을 작성했습니다!")

st.download_button(
    "⬇️ 데이터 사전 CSV 다운로드",
    data=edited.to_csv(index=False).encode("utf-8-sig"),
    file_name="neuro_data_dictionary.csv",
    mime="text/csv",
)


# =========================================================
# SECTION 03 — 원본 데이터
# =========================================================

section(
    "03",
    "원본 데이터 미리보기",
    "원본 CSV의 첫 5줄을 그대로 확인합니다.",
)

st.dataframe(
    df.head(5),
    hide_index=True,
    use_container_width=True,
)

bmi_missing = int(df["bmi"].isna().sum())

st.warning(
    f"🧩 BMI 열에는 {bmi_missing:,}개의 빈 값이 있습니다. "
    "원본 결측치는 변경하지 않았습니다."
)

with st.expander("🔍 전체 열의 결측치 개수 확인"):
    missing_table = pd.DataFrame({
        "열 이름": df.columns,
        "빈 값 개수": df.isna().sum().values,
    })

    st.dataframe(
        missing_table,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# SECTION 04 — 출처
# =========================================================

section(
    "04",
    "데이터 출처와 신뢰성",
    "교재의 출처 정보를 확인하고 직접 수정할 수 있습니다.",
)

source_text = st.text_area(
    "📚 데이터 출처",
    value=DEFAULT_SOURCE,
    height=180,
    key="source_note",
)

st.download_button(
    "⬇️ 출처 메모 다운로드",
    data=source_text.encode("utf-8-sig"),
    file_name="neuro_source.txt",
    mime="text/plain",
)

with st.expander("🔐 데이터 윤리와 주의사항"):
    st.write(
        "이 데이터는 교육용 분석 자료입니다. "
        "데이터의 수집 과정과 대표성이 충분히 확인되지 않았으므로 "
        "분석 결과를 실제 의료 진단이나 치료 판단에 사용하면 안 됩니다."
    )


# =========================================================
# 마무리
# =========================================================

st.divider()

st.markdown("### 🚀 다음 단계: 데이터 탐색")
st.write(
    "왼쪽 기본 페이지 메뉴에서 **탐색**을 선택하면 "
    "히스토그램, 상자그림, 건강 상태별 비율을 확인할 수 있습니다."
)

st.markdown(
    '<div class="footer">'
    'NEURO LAB · EDUCATIONAL DATA SCIENCE STUDIO'
    '</div>',
    unsafe_allow_html=True,
)
