
from pathlib import Path
import re

import pandas as pd
import streamlit as st


# =========================================================
# 01. 기본 설정
# =========================================================

st.set_page_config(
    page_title="NEURO LAB | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "Pages"  # 대문자 P 유지

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/main/data/stroke.csv"
)

SOURCE = """출처: Kaggle Stroke Prediction Dataset
게시자: fedesoriano
원본 주소: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
원본 설명: (Confidential Source) - Use only for educational purposes
용도: 교육용 데이터 분석 실습"""


# =========================================================
# 02. 공통 디자인
# 메인과 탐색 페이지에 모두 적용됩니다.
# =========================================================

st.markdown("""
<style>
:root {
    --paper: #F8F8F5;
    --ink: #242923;
    --muted: #697068;
    --blue: #3559D5;
    --line: #DFE2DA;
}

.stApp {
    background: var(--paper);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: var(--paper);
}

[data-testid="stSidebar"] {
    background: #F0F1ED;
    border-right: 1px solid var(--line);
}

.block-container {
    max-width: 1220px;
    padding-top: 2.25rem;
    padding-bottom: 4.5rem;
}

h1, h2, h3 {
    color: var(--ink) !important;
    letter-spacing: -.04em;
}

[data-testid="stMarkdownContainer"] p {
    line-height: 1.75;
}

.masthead {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid #D5DAD1;
    padding-bottom: 18px;
    margin-bottom: 46px;
}

.brand {
    font-size: 15px;
    font-weight: 850;
    letter-spacing: -.03em;
    color: var(--ink);
}

.issue {
    font-size: 11px;
    letter-spacing: .14em;
    color: #737A70;
    font-weight: 700;
}

.kicker {
    color: var(--blue);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .17em;
    margin-bottom: 17px;
}

.headline {
    font-size: clamp(39px, 6vw, 76px);
    font-weight: 850;
    line-height: 1.14;
    letter-spacing: -.075em;
    color: var(--ink);
}

.headline em {
    color: var(--blue);
    font-style: normal;
}

.lead {
    max-width: 670px;
    font-size: 16px;
    line-height: 1.9;
    color: #60695F;
    margin-top: 22px;
    margin-bottom: 40px;
}

.section {
    border-top: 1px solid #D5DAD1;
    padding-top: 21px;
    margin-top: 68px;
    display: flex;
    gap: 19px;
    align-items: baseline;
}

.section b {
    font-size: 12px;
    color: var(--blue);
}

.section span {
    font-size: clamp(23px, 3vw, 32px);
    font-weight: 800;
    letter-spacing: -.055em;
    color: var(--ink);
}

.section-desc {
    font-size: 14px;
    color: #6A7469;
    margin: 9px 0 23px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DFE3DB;
    border-radius: 6px;
    padding: 21px;
    min-height: 129px;
}

[data-testid="stMetricLabel"] {
    color: #667065;
}

[data-testid="stMetricValue"] {
    color: var(--ink);
    font-size: clamp(25px, 2.8vw, 38px);
    font-weight: 800;
    letter-spacing: -.05em;
}

[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
    border: 1px solid #E1E4DE;
    border-radius: 6px;
    overflow: hidden;
}

.stButton button,
.stDownloadButton button {
    border-radius: 6px;
    font-weight: 700;
}

[data-testid="stProgressBar"] > div > div {
    background: var(--blue);
}

.info-note {
    background: #EDF0E9;
    border-left: 3px solid var(--blue);
    padding: 17px 20px;
    color: #445045;
    font-size: 14px;
    line-height: 1.8;
    margin: 20px 0;
}

.footer {
    margin-top: 76px;
    border-top: 1px solid #D5DAD1;
    padding-top: 20px;
    font-size: 11px;
    letter-spacing: .13em;
    color: #7C8479;
}

@media(max-width:700px) {
    .block-container {
        padding-top: 1rem;
    }

    .headline {
        font-size: 40px;
    }

    .masthead {
        margin-bottom: 34px;
    }

    .section {
        margin-top: 44px;
    }

    .issue {
        font-size: 10px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 03. 데이터 읽기
# =========================================================

@st.cache_data(ttl=3600)
def load_data():

    data = pd.read_csv(
        DATA_URL,
        encoding="utf-8",
    )

    required = [
        "id",
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: "
            + ", ".join(missing)
        )

    return data


def section(number, title, description):

    st.markdown(
        f"""
        <div class="section">
            <b>{number}</b>
            <span>{title}</span>
        </div>

        <div class="section-desc">
            {description}
        </div>
        """,
        unsafe_allow_html=True,
    )


def value_summary(series):

    observed = series.dropna()

    if observed.empty:
        return "값 없음"

    if (
        pd.api.types.is_numeric_dtype(series)
        and observed.nunique() > 2
    ):
        return (
            f"숫자 {observed.min():g} "
            f"~ {observed.max():g}"
        )

    values = sorted(
        observed.astype(str).unique()
    )

    return (
        f"{len(values)}가지 · "
        + " / ".join(values)
    )


# =========================================================
# 04. 첫 화면 전체 내용
# =========================================================

def show_home():

    # 상단 타이틀
    st.markdown(
        """
        <div class="masthead">
            <span class="brand">NEURO / LAB</span>
            <span class="issue">DATA STUDY · 01</span>
        </div>

        <div class="kicker">
            01 / THE DATASET
        </div>

        <div class="headline">
            건강 데이터를<br>
            <em>읽는 새로운 방식.</em>
        </div>

        <div class="lead">
            숫자에서 시작해 질문을 발견하는 데이터 탐구.
            변수의 뜻을 이해하고, 원본 데이터를 살펴보고,
            결과를 해석할 준비를 합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 데이터 불러오기
    try:
        df = load_data()

    except Exception as exc:

        st.error(
            "데이터를 불러오지 못했습니다."
        )

        with st.expander("자세한 오류 보기"):
            st.code(str(exc))

        if st.button("데이터 다시 불러오기"):
            load_data.clear()
            st.rerun()

        st.stop()

    # -----------------------------------------------------
    # 01. 데이터 한눈에 보기
    # -----------------------------------------------------

    section(
        "01",
        "데이터 한눈에 보기",
        "전체 규모와 뇌졸중 경험 비율을 확인합니다.",
    )

    count = len(df)

    cases = int(
        df["stroke"].eq(1).sum()
    )

    cols = st.columns(4)

    cols[0].metric(
        "전체 관측 인원",
        f"{count:,}명",
    )

    cols[1].metric(
        "데이터 변수",
        f"{df.shape[1]}개",
    )

    cols[2].metric(
        "뇌졸중 경험자",
        f"{cases:,}명",
    )

    cols[3].metric(
        "뇌졸중 경험 비율",
        (
            f"{cases / count * 100:.2f}%"
            if count
            else "—"
        ),
    )

    st.markdown(
        """
        <div class="info-note">
            <b>읽는 법</b> ·
            stroke가 1이면 뇌졸중 경험 있음,
            0이면 경험 없음을 뜻합니다.
            경험 비율은 경험자 수를 전체 인원으로
            나누어 계산합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # 02. 데이터 사전
    # -----------------------------------------------------

    section(
        "02",
        "변수를 이해하는 시간",
        "우리말 뜻 칸을 직접 채우고 CSV로 내려받을 수 있습니다.",
    )

    if "neuro_meanings" not in st.session_state:

        st.session_state.neuro_meanings = {
            name: ""
            for name in df.columns
        }

    dictionary = pd.DataFrame([
        {
            "열 이름": name,
            "우리말 뜻": (
                st.session_state.neuro_meanings
                .get(name, "")
            ),
            "값의 종류": value_summary(
                df[name]
            ),
            "빈 값 개수": int(
                df[name].isna().sum()
            ),
        }
        for name in df.columns
    ])

    edited = st.data_editor(
        dictionary,
        key="neuro_dictionary_editor",
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        height=445,
        disabled=[
            "열 이름",
            "값의 종류",
            "빈 값 개수",
        ],
        column_config={
            "열 이름": st.column_config.TextColumn(
                width="medium",
            ),
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻 · 직접 입력",
                width="large",
            ),
            "값의 종류": st.column_config.TextColumn(
                width="large",
            ),
            "빈 값 개수": st.column_config.NumberColumn(
                format="%d",
            ),
        },
    )

    for _, row in edited.iterrows():

        value = row["우리말 뜻"]

        st.session_state.neuro_meanings[
            row["열 이름"]
        ] = (
            ""
            if pd.isna(value)
            else str(value)
        )

    completed = sum(
        bool(str(value).strip())
        for value in (
            st.session_state.neuro_meanings.values()
        )
    )

    st.progress(
        completed / len(df.columns),
        text=(
            f"작성 진행률 · "
            f"{completed} / {len(df.columns)}"
        ),
    )

    st.download_button(
        "데이터 사전 저장 ↗",
        data=edited.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="neuro_dictionary.csv",
        mime="text/csv",
    )

    # -----------------------------------------------------
    # 03. 원본 데이터와 결측치
    # -----------------------------------------------------

    section(
        "03",
        "원본 데이터 살펴보기",
        "원본 데이터의 첫 다섯 줄과 결측치를 살펴봅니다.",
    )

    st.dataframe(
        df.head(5),
        hide_index=True,
        use_container_width=True,
    )

    bmi_missing = int(
        df["bmi"].isna().sum()
    )

    st.markdown(
        f"""
        <div class="info-note">
            <b>관찰 포인트</b> ·
            BMI 열의 빈 값은
            <b>{bmi_missing:,}개</b>입니다.
            결측치도 중요한 분석 단서입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(
        "모든 열의 빈 값 개수 보기"
    ):

        # 기존 reset_index(names=...) 오류 해결
        # pandas 버전에 영향을 덜 받는 방식
        missing_table = pd.DataFrame({
            "열 이름": df.columns,
            "빈 값 개수": (
                df.isna().sum().to_numpy()
            ),
        })

        st.dataframe(
            missing_table,
            hide_index=True,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # 04. 데이터 출처
    # -----------------------------------------------------

    section(
        "04",
        "데이터의 출처",
        "출처를 기록하고 교육용 데이터의 한계를 이해합니다.",
    )

    source = st.text_area(
        "출처 기록",
        value=SOURCE,
        key="neuro_source",
        height=165,
    )

    st.download_button(
        "출처 메모 저장 ↗",
        data=source.encode("utf-8-sig"),
        file_name="neuro_source.txt",
        mime="text/plain",
    )

    st.markdown(
        """
        <div class="info-note">
            <b>데이터 윤리</b> ·
            데이터 수집 과정과 대표성은 충분히
            확인되지 않았습니다.
            이 자료의 분석 결과를 실제 의료 진단이나
            치료 판단에 사용하지 마세요.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # 05. 다음 단계
    # -----------------------------------------------------

    section(
        "05",
        "이제 패턴을 찾아볼 차례",
        "왼쪽 페이지 메뉴의 '탐색'을 선택하면 "
        "분포와 집단별 차이를 확인할 수 있습니다.",
    )

    st.markdown(
        """
        <div class="footer">
            NEURO / LAB ·
            A STUDY IN DATA ·
            EDUCATIONAL USE ONLY
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 05. 대문자 Pages 폴더 자동 탐색
# =========================================================

def page_title(path):

    # 예: 1_탐색.py → 탐색
    # 예: 2_모델링.py → 모델링

    no_prefix = re.sub(
        r"^\d+[_\- ]*",
        "",
        path.stem,
    )

    return no_prefix.replace("_", " ")


# =========================================================
# 06. 페이지 자동 등록
# =========================================================

home = st.Page(
    show_home,
    title="데이터 소개",
    icon="🏠",
    default=True,
)

page_items = [home]

if PAGES_DIR.is_dir():

    # Pages 폴더의 모든 .py 파일 검색
    for page_path in sorted(
        PAGES_DIR.glob("*.py")
    ):

        # _로 시작하는 파일은 제외
        if page_path.name.startswith("_"):
            continue

        # main.py 기준 상대 경로
        relative_path = (
            page_path
            .relative_to(BASE_DIR)
            .as_posix()
        )

        page_items.append(
            st.Page(
                relative_path,
                title=page_title(page_path),
                icon="🔎",
            )
        )


# =========================================================
# 07. 왼쪽 사이드바 메뉴 생성
# =========================================================

nav = st.navigation(
    page_items,
    position="sidebar",
    expanded=True,
)

with st.sidebar:
    st.caption(
        "NEURO / LAB · DATA STUDY"
    )

# 선택한 페이지 실행
nav.run()
