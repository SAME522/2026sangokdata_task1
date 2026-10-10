
from pathlib import Path
import re

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NEURO LAB | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

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
# 01. 모든 페이지에서 사용할 디자인
# =========================================================

st.markdown("""
<style>
:root {
    --ink: #202943;
    --muted: #65718A;
    --line: #DBE2F1;
    --blue: #386BFA;
}

.stApp {
    background: #FBFCFF;
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: #FBFCFF;
}

[data-testid="stSidebar"] {
    background: #F0F4FF;
    border-right: 1px solid #DBE2F1;
}

.block-container {
    max-width: 1260px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    color: #202943 !important;
    letter-spacing: -.045em;
}

.masthead {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid #DEE5F2;
    padding-bottom: 17px;
    margin-bottom: 45px;
}

.brand {
    font-weight: 900;
    letter-spacing: -.04em;
    color: #202943;
    font-size: 16px;
}

.issue {
    font-size: 11px;
    letter-spacing: .12em;
    font-weight: 800;
    color: #66738A;
}

.kicker {
    color: #386BFA;
    font-weight: 850;
    letter-spacing: .17em;
    font-size: 12px;
}

.headline {
    font-size: clamp(41px, 6vw, 76px);
    font-weight: 900;
    line-height: 1.14;
    letter-spacing: -.075em;
    color: #202943;
    margin: 17px 0;
}

.headline em {
    color: #386BFA;
    font-style: normal;
}

.lead {
    font-size: 16px;
    line-height: 1.9;
    color: #68748A;
    max-width: 720px;
    margin-bottom: 36px;
}

.section {
    display: flex;
    align-items: baseline;
    gap: 19px;
    border-top: 1px solid #DEE5F2;
    padding-top: 22px;
    margin-top: 65px;
}

.section b {
    font-size: 12px;
    color: #386BFA;
}

.section span {
    font-size: clamp(24px, 3vw, 33px);
    font-weight: 850;
    letter-spacing: -.05em;
}

.section-desc {
    font-size: 14px;
    color: #6A768A;
    margin: 8px 0 23px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DFE5F1;
    border-radius: 12px;
    padding: 22px;
    min-height: 130px;
    box-shadow: 0 5px 24px #344B7810;
}

[data-testid="stMetricLabel"] {
    color: #65718A;
}

[data-testid="stMetricValue"] {
    font-weight: 850;
    color: #202943;
    letter-spacing: -.05em;
}

[data-testid="stDataFrame"],
[data-testid="stDataEditor"],
[data-testid="stPlotlyChart"] {
    border: 1px solid #DFE5F1;
    border-radius: 12px;
    overflow: hidden;
}

.stButton button,
.stDownloadButton button {
    border-radius: 9px;
    font-weight: 700;
}

.note {
    background: #EEF3FF;
    border-left: 4px solid #386BFA;
    padding: 18px 20px;
    color: #40516B;
    line-height: 1.8;
    border-radius: 0 10px 10px 0;
    margin: 18px 0;
}

.footer {
    border-top: 1px solid #DEE5F2;
    padding-top: 20px;
    margin-top: 73px;
    color: #8190A5;
    font-size: 11px;
    letter-spacing: .1em;
}

/* 다섯 가지 전대 컬러 */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 5px;
    z-index: 999999;
    background: linear-gradient(
        90deg,
        #F34E67 0% 20%,
        #386BFA 20% 40%,
        #19B58A 40% 60%,
        #FFC83D 60% 80%,
        #E968C6 80% 100%
    );
}

@media(max-width: 700px) {
    .block-container {
        padding-top: 1.1rem;
    }

    .headline {
        font-size: 40px;
    }

    .masthead {
        margin-bottom: 30px;
    }

    .section {
        margin-top: 45px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 02. 데이터 읽기
# =========================================================

@st.cache_data(ttl=3600)
def load_data():

    df = pd.read_csv(
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
        name for name in required
        if name not in df.columns
    ]

    if missing:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: "
            + ", ".join(missing)
        )

    return df


def section(num, heading, description):

    st.markdown(
        f"""
        <div class="section">
            <b>{num}</b>
            <span>{heading}</span>
        </div>
        <div class="section-desc">
            {description}
        </div>
        """,
        unsafe_allow_html=True,
    )


def describe_values(series):

    s = series.dropna()

    if s.empty:
        return "값 없음"

    if (
        pd.api.types.is_numeric_dtype(series)
        and s.nunique() > 2
    ):
        return f"숫자 {s.min():g} ~ {s.max():g}"

    values = sorted(
        s.astype(str).unique()
    )

    return (
        f"{len(values)}가지 · "
        + " / ".join(values)
    )


# =========================================================
# 03. 메인 페이지
# =========================================================

def show_home():

    st.markdown("""
    <div class="masthead">
        <span class="brand">NEURO / LAB</span>
        <span class="issue">DATA STUDY · VOL. 01</span>
    </div>

    <div class="kicker">01 / THE DATASET</div>

    <div class="headline">
        건강 데이터를<br>
        <em>읽는 새로운 방식.</em>
    </div>

    <div class="lead">
        숫자에서 시작해 질문을 발견하는 데이터 탐구.
        변수의 뜻을 이해하고, 원본 데이터를 살펴보고,
        결과를 해석할 준비를 합니다.
    </div>
    """, unsafe_allow_html=True)

    try:
        df = load_data()

    except Exception as exc:

        st.error("데이터를 불러오지 못했습니다.")

        with st.expander("오류 상세"):
            st.code(str(exc))

        if st.button("다시 불러오기"):
            load_data.clear()
            st.rerun()

        st.stop()

    # -----------------------------------------------------
    # 데이터 통계
    # -----------------------------------------------------

    section(
        "01",
        "데이터 한눈에 보기",
        "규모와 뇌졸중 경험 비율을 확인합니다.",
    )

    total = len(df)

    positive = int(
        df["stroke"].eq(1).sum()
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "전체 관측 인원",
        f"{total:,}명",
    )

    b.metric(
        "데이터 변수",
        f"{df.shape[1]}개",
    )

    c.metric(
        "뇌졸중 경험자",
        f"{positive:,}명",
    )

    d.metric(
        "뇌졸중 경험 비율",
        (
            f"{positive / total * 100:.2f}%"
            if total
            else "—"
        ),
    )

    st.markdown("""
    <div class="note">
        <b>읽는 법</b> ·
        stroke=1은 뇌졸중 경험 있음(양성),
        stroke=0은 경험 없음(음성)입니다.
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # 데이터 사전
    # -----------------------------------------------------

    section(
        "02",
        "변수를 이해하는 시간",
        "우리말 뜻을 직접 작성하고 CSV로 저장할 수 있습니다.",
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
            "값의 종류": describe_values(
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
        height=445,
        num_rows="fixed",
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

        val = row["우리말 뜻"]

        st.session_state.neuro_meanings[
            row["열 이름"]
        ] = (
            ""
            if pd.isna(val)
            else str(val)
        )

    done = sum(
        bool(str(v).strip())
        for v in (
            st.session_state.neuro_meanings.values()
        )
    )

    st.progress(
        done / len(df.columns),
        text=(
            f"작성 진행률 · "
            f"{done} / {len(df.columns)}"
        ),
    )

    st.download_button(
        "데이터 사전 저장 ↗",
        edited.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="neuro_dictionary.csv",
        mime="text/csv",
    )

    # -----------------------------------------------------
    # 원본 데이터와 결측치
    # -----------------------------------------------------

    section(
        "03",
        "원본 데이터 살펴보기",
        "앞 다섯 줄과 열별 결측치를 확인합니다.",
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
        <div class="note">
            <b>관찰 포인트</b> ·
            BMI 열의 빈 값은
            <b>{bmi_missing:,}개</b>입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(
        "모든 열의 빈 값 개수 보기"
    ):

        # 이전 pandas TypeError 방지
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
    # 출처
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
        source.encode("utf-8-sig"),
        file_name="neuro_source.txt",
        mime="text/plain",
    )

    st.markdown("""
    <div class="note">
        <b>데이터 윤리</b> ·
        수집 과정과 대표성이 충분히 확인되지 않았습니다.
        실제 의료 진단에 사용하지 마세요.
    </div>
    """, unsafe_allow_html=True)

    section(
        "05",
        "분류 모델까지 이어지는 탐구",
        "왼쪽 메뉴에서 탐색과 분류 모델 실습을 선택해 보세요.",
    )

    st.markdown("""
    <div class="footer">
        NEURO / LAB · A STUDY IN DATA · EDUCATIONAL USE ONLY
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 04. 페이지 이름 변환
# =========================================================

def clean_title(path):
    """
    1_탐색.py -> 탐색
    2_분류_모델.py -> 분류 모델
    """

    title = re.sub(
        r"^\d+[_\- ]*",
        "",
        path.stem,
    )

    return title.replace("_", " ")


# =========================================================
# 05. 페이지 자동 등록
# =========================================================

home = st.Page(
    show_home,
    title="데이터 소개",
    icon="🏠",
    default=True,
)

page_list = [home]

# 대문자 Pages 우선, 소문자 pages도 지원
for folder_name in ("Pages", "pages"):

    folder = ROOT / folder_name

    if folder.is_dir():

        for path in sorted(
            folder.glob("*.py")
        ):

            if (
                path.is_file()
                and not path.name.startswith("_")
            ):

                page_list.append(
                    st.Page(
                        path.relative_to(
                            ROOT
                        ).as_posix(),
                        title=clean_title(path),
                        icon=(
                            "🤖"
                            if "분류" in path.stem
                            else "🔎"
                        ),
                    )
                )

        break


# =========================================================
# 06. 왼쪽 사이드바 자동 페이지 메뉴
# =========================================================

nav = st.navigation(
    page_list,
    position="sidebar",
    expanded=True,
)

with st.sidebar:
    st.markdown("**NEURO / LAB**")
    st.caption(
        "DATA EXPLORATION · MODEL ARENA"
    )

nav.run()
