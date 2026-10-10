
import pandas as pd
import streamlit as st

# ============================================================
# NEURO LAB — 데이터 소개
# ============================================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실 | NEURO LAB",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

SOURCE_TEXT = """출처: 캐글 Stroke Prediction Dataset (게시자 fedesoriano, 2021-01-26 공개)
원본 주소: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
라이선스: Data files © Original Authors
게시자가 적은 문구: (Confidential Source) - Use only for educational purposes
이 수업에서는 교육 목적으로만 사용합니다."""

CSS = """
<style>
:root {
    --cyan: #40E0D0;
    --violet: #A78BFA;
    --muted: #94A3B8;
}
.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, rgba(44,77,145,.17), transparent 35%),
        radial-gradient(ellipse at 0% 65%, rgba(16,185,170,.07), transparent 35%),
        #080F1F;
    color: #E8F0FF;
}
[data-testid="stSidebar"] {
    background: #0D172B;
    border-right: 1px solid #233450;
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1320px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}
h1, h2, h3 {
    color: #F2F7FF !important;
    letter-spacing: -.035em;
}
p, label {
    line-height: 1.7;
}
.hero {
    position: relative;
    overflow: hidden;
    padding: 48px 50px;
    border-radius: 26px;
    background: linear-gradient(115deg, #122448, #152D50 60%, #075C65);
    border: 1px solid #345A77;
    box-shadow: 0 18px 55px #00000040;
    margin-bottom: 30px;
}
.hero:after {
    content: "◉";
    position: absolute;
    right: 6%;
    top: -75px;
    font-size: 330px;
    line-height: 1;
    color: rgba(80,240,230,.055);
    pointer-events: none;
}
.eyebrow {
    color: #65F2DE;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}
.hero h1 {
    font-size: clamp(32px, 4vw, 52px);
    line-height: 1.2;
    margin: 18px 0;
    font-weight: 850;
}
.hero p {
    color: #C5D8E9;
    font-size: 16px;
    max-width: 760px;
}
.pill {
    display: inline-block;
    border: 1px solid #70DAD35A;
    background: #5CDED018;
    padding: 7px 14px;
    border-radius: 99px;
    font-size: 12px;
    margin: 14px 8px 0 0;
    color: #B9FFF1;
}
.section {
    margin-top: 44px;
    margin-bottom: 18px;
}
.section .number {
    color: #5CEAD8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}
.section h2 {
    font-size: 27px;
    margin: 7px 0;
}
.section p {
    color: #A9BAD0;
    font-size: 14px;
    margin: 0;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #14243D, #101B30);
    border: 1px solid #2A405E;
    border-radius: 20px;
    padding: 23px;
    min-height: 140px;
    box-shadow: 0 8px 25px #00000018;
}
[data-testid="stMetricLabel"] {
    color: #A7BDD2;
}
[data-testid="stMetricValue"] {
    color: #5CEAD8;
    font-size: clamp(25px, 2.5vw, 38px);
    font-weight: 850;
}
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
    border: 1px solid #2A405E;
    border-radius: 15px;
    overflow: hidden;
}
.stButton button,
.stDownloadButton button,
[data-testid="stPageLink"] a {
    border-radius: 12px;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #16B8AE, #8B5CF6);
}
.footer {
    margin-top: 50px;
    padding: 25px;
    border-top: 1px solid #27364C;
    text-align: center;
    color: #8EA3BB;
    font-size: 12px;
    letter-spacing: 1px;
}
@media(max-width: 700px) {
    .block-container {padding-top: 1rem;}
    .hero {padding: 28px 24px;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "id", "gender", "age", "hypertension", "heart_disease",
        "ever_married", "work_type", "Residence_type",
        "avg_glucose_level", "bmi", "smoking_status", "stroke",
    ]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError("필수 열 누락: " + ", ".join(missing))
    return data


def section(number, title, description):
    st.markdown(
        f"""
        <div class="section">
            <div class="number">SECTION {number} / NEURO LAB</div>
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


with st.sidebar:
    st.markdown("## 🧠 NEURO LAB")
    st.caption("STROKE DATA SCIENCE STUDIO")
    st.divider()
    st.page_link("main.py", label="데이터 소개", icon="🏠")
    st.page_link("pages/1_탐색.py", label="데이터 탐색", icon="🔎")
    st.divider()
    st.markdown("**LEARNING PATH**")
    st.caption("01 · 데이터 구조 이해")
    st.caption("02 · 데이터 시각화")
    st.caption("03 · 패턴 발견")
    st.caption("04 · 모델링 준비")
    st.divider()
    st.caption("교육용 데이터 분석 프로젝트")


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">● NEURO LAB / DATA INTELLIGENCE</div>
        <h1>🧠 뇌졸중 예측<br>실습실</h1>
        <p>
            건강 기록 속 숨겨진 패턴을 발견하는 데이터 사이언스 스튜디오.
            데이터의 구조를 이해하고, 시각화하고, 분석하는 여정을 시작하세요.
        </p>
        <span class="pill">01 / DATA OVERVIEW</span>
        <span class="pill">INTERACTIVE LEARNING</span>
        <span class="pill">PYTHON + STREAMLIT</span>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("데이터를 불러오는 중입니다..."):
        df = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다.")
    with st.expander("오류 내용"):
        st.code(str(exc))
    if st.button("🔄 다시 불러오기"):
        load_data.clear()
        st.rerun()
    st.stop()

total = len(df)
stroke_count = int(df["stroke"].eq(1).sum())
stroke_rate = stroke_count / total * 100 if total else 0

section(
    "01",
    "데이터 인사이트",
    "데이터의 규모와 뇌졸중 발생 현황을 숫자로 확인합니다.",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 전체 사람 수", f"{total:,}명")
c2.metric("🗂 전체 열 개수", f"{df.shape[1]}개")
c3.metric("🧠 뇌졸중 경험자", f"{stroke_count:,}명")
c4.metric("📈 뇌졸중 비율", f"{stroke_rate:.2f}%")

st.info(
    "💡 뇌졸중 비율은 stroke가 1인 사람 수를 "
    "전체 사람 수로 나눈 뒤 100을 곱해 계산합니다."
)

section(
    "02",
    "인터랙티브 데이터 사전",
    "교재를 참고하여 우리말 뜻을 직접 작성하고 학습 결과를 저장하세요.",
)

if "meanings" not in st.session_state:
    st.session_state.meanings = {col: "" for col in df.columns}

dictionary = pd.DataFrame([
    {
        "열 이름": col,
        "우리말 뜻": st.session_state.meanings.get(col, ""),
        "값의 종류": value_type(df[col]),
        "빈 값 개수": int(df[col].isna().sum()),
    }
    for col in df.columns
])

edited = st.data_editor(
    dictionary,
    key="dictionary_editor",
    hide_index=True,
    width="stretch",
    height=460,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", width="medium"),
        "우리말 뜻": st.column_config.TextColumn("✏️ 우리말 뜻", width="large"),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
)

for _, row in edited.iterrows():
    value = row["우리말 뜻"]
    st.session_state.meanings[row["열 이름"]] = (
        "" if pd.isna(value) else str(value)
    )

completed = sum(
    bool(str(v).strip()) for v in st.session_state.meanings.values()
)
progress = completed / len(df.columns)

st.progress(
    progress,
    text=f"데이터 사전 작성 진행률 · {completed}/{len(df.columns)}",
)

if completed == len(df.columns):
    st.success("🎉 모든 열의 우리말 뜻을 작성했습니다!")

st.download_button(
    "⬇️ 작성한 데이터 사전 다운로드",
    data=edited.to_csv(index=False).encode("utf-8-sig"),
    file_name="neuro_data_dictionary.csv",
    mime="text/csv",
)

section(
    "03",
    "원본 데이터 미리보기",
    "처음 다섯 줄을 가공하지 않고 원본 순서 그대로 보여 줍니다.",
)

st.dataframe(
    df.head(5),
    hide_index=True,
    width="stretch",
)

bmi_missing = int(df["bmi"].isna().sum())
st.warning(
    f"🧩 BMI 열에는 빈 값이 {bmi_missing:,}개 있습니다. "
    "빈 값을 임의로 채우지 않고 그대로 유지했습니다."
)

section(
    "04",
    "데이터 출처와 신뢰성",
    "출처 정보를 확인하고 데이터 사용 시 주의사항을 살펴봅니다.",
)

source_text = st.text_area(
    "📚 출처 기록",
    value=SOURCE_TEXT,
    height=190,
    key="source_note",
)

st.download_button(
    "⬇️ 출처 기록 다운로드",
    data=source_text.encode("utf-8-sig"),
    file_name="neuro_source.txt",
    mime="text/plain",
)

with st.expander("🔐 데이터 윤리와 주의사항"):
    st.write(
        "원본 데이터의 수집 과정과 대표성이 충분히 확인되지 않았습니다. "
        "개인 식별 정보가 없다는 사실만으로 재식별 가능성이 "
        "완전히 사라지는 것은 아닙니다. "
        "분석 결과를 실제 의료 진단이나 치료 판단에 사용하지 마세요."
    )

st.divider()
st.markdown("### 🚀 다음 실습으로 이동")
st.write("나이와 혈당의 분포, 집단별 차이, 결측치와 흡연 상태를 분석합니다.")

st.page_link(
    "pages/1_탐색.py",
    label="🔎 데이터 탐색 스튜디오 입장 →",
    use_container_width=True,
)

st.markdown(
    '<div class="footer">NEURO LAB © EDUCATIONAL DATA SCIENCE STUDIO</div>',
    unsafe_allow_html=True,
)
