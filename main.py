
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.markdown("""
<style>
.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
.hero {
    padding: 45px;
    border-radius: 24px;
    background: linear-gradient(120deg, #111c3e, #184c70, #087e83);
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 43px;
    font-weight: 850;
    margin-bottom: 12px;
}
.hero p {
    color: #d9f5f4;
    font-size: 17px;
    line-height: 1.8;
}
.tag {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 30px;
    background: #ffffff25;
    margin: 8px 8px 0 0;
    font-size: 13px;
}
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid #64748b40;
    border-radius: 18px;
    padding: 20px;
}
[data-testid="stMetricValue"] {
    font-weight: 800;
}
.section {
    margin-top: 38px;
    margin-bottom: 15px;
}
.section h2 {
    font-size: 26px;
    font-weight: 800;
}
div.stButton button {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


with st.sidebar:
    st.title("🧠 실습실 메뉴")
    st.page_link("main.py", label="🏠 데이터 소개")
    st.page_link("pages/1_탐색.py", label="🔎 데이터 탐색")
    st.divider()
    st.info("데이터를 이해하고 다양한 그래프로 탐색해 보세요.")


st.markdown("""
<div class="hero">
    <div style="color:#80eee2;font-weight:800;letter-spacing:2px;">
        STROKE DATA LAB
    </div>
    <h1>🧠 뇌졸중 예측 실습실</h1>
    <p>
        건강 데이터로 배우는 데이터 분석과 인공지능.<br>
        실제 데이터의 구조를 이해하고 다양한 시각화로 특징을 발견해 보세요.
    </p>
    <span class="tag">📊 데이터 분석</span>
    <span class="tag">🔎 탐색적 데이터 분석</span>
    <span class="tag">🤖 머신러닝 실습</span>
</div>
""", unsafe_allow_html=True)

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    if st.button("🔄 다시 불러오기"):
        load_data.clear()
        st.rerun()
    st.stop()

total = len(df)
stroke_count = int((df["stroke"] == 1).sum())
stroke_rate = stroke_count / total * 100 if total else 0

st.markdown("## 📊 데이터 한눈에 보기")

c1, c2, c3, c4 = st.columns(4)

c1.metric("👥 전체 사람 수", f"{total:,}명")
c2.metric("🗂️ 전체 열 개수", f"{len(df.columns)}개")
c3.metric("🧠 뇌졸중 경험자", f"{stroke_count:,}명")
c4.metric("📈 뇌졸중 비율", f"{stroke_rate:.2f}%")

st.info(
    "뇌졸중 비율 = 뇌졸중을 경험한 사람 수 ÷ 전체 사람 수 × 100"
)

st.markdown("---")
st.markdown("## 📚 데이터 사전 만들기")
st.caption("교재를 참고하여 '우리말 뜻' 칸을 직접 작성하세요.")


def get_type(series, name):
    if name == "id":
        return "식별자"
    if name in ["stroke", "hypertension", "heart_disease"]:
        return "이진형 (0 / 1)"
    if pd.api.types.is_numeric_dtype(series):
        return "수치형"
    return "범주형"


if "meanings" not in st.session_state:
    st.session_state.meanings = {
        col: "" for col in df.columns
    }

rows = []

for col in df.columns:
    rows.append({
        "열 이름": col,
        "우리말 뜻": st.session_state.meanings.get(col, ""),
        "값의 종류": get_type(df[col], col),
        "빈 값 개수": int(df[col].isna().sum()),
    })

dictionary = pd.DataFrame(rows)

edited = st.data_editor(
    dictionary,
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
            "값의 종류", width="medium"
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
    bool(value.strip())
    for value in st.session_state.meanings.values()
)

st.progress(
    completed / len(df.columns),
    text=f"📝 작성 진행률: {completed}/{len(df.columns)}개",
)

st.download_button(
    "⬇️ 데이터 사전 CSV 다운로드",
    data=edited.to_csv(index=False).encode("utf-8-sig"),
    file_name="stroke_dictionary.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("## 🔍 원본 데이터 미리보기")
st.caption("CSV 파일의 처음 5줄을 원본 순서대로 보여 줍니다.")

st.dataframe(
    df.head(5),
    hide_index=True,
    use_container_width=True,
)

missing_bmi = int(df["bmi"].isna().sum())

st.warning(
    f"⚠️ BMI 열에는 총 {missing_bmi:,}개의 빈 값이 있습니다."
)

st.markdown("---")
st.markdown("## 📖 데이터 출처 기록")
st.caption("교재에 나온 데이터 출처를 직접 작성하세요.")

source_text = st.text_area(
    "데이터 출처",
    placeholder="여기에 교재의 데이터 출처를 입력하세요.",
    height=120,
    key="source_text",
)

st.download_button(
    "⬇️ 출처 메모 다운로드",
    data=source_text.encode("utf-8-sig"),
    file_name="stroke_source.txt",
    mime="text/plain",
    disabled=not source_text.strip(),
)

st.markdown("---")
st.markdown("### 🚀 다음 단계")
st.write("이제 데이터를 다양한 그래프로 탐색해 보세요.")

st.page_link(
    "pages/1_탐색.py",
    label="🔎 데이터 탐색 페이지로 이동 →",
    use_container_width=True,
)
