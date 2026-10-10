
from pathlib import Path
import html

import pandas as pd
import streamlit as st


# ============================================================
# NEURO FORCE
# 뇌졸중 예측 실습실
#
# 기능:
# 1. 데이터 소개 화면
# 2. Pages 폴더 자동 페이지 등록
# 3. 전대물 테마 공통 디자인
# 4. 데이터 사전 편집
# 5. 결측치 확인
# 6. 데이터 출처 기록
#
# 페이지 설정은 이 파일에서만 관리합니다.
# ============================================================


# ============================================================
# 01. 페이지 기본 설정
# ============================================================

st.set_page_config(
    page_title="NEURO FORCE | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/main/data/stroke.csv"
)

SOURCE_TEXT = """출처: Kaggle Stroke Prediction Dataset
게시자: fedesoriano
원본 주소: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
원본 설명: (Confidential Source) - Use only for educational purposes
용도: 교육용 데이터 분석 실습"""


# ============================================================
# 02. 색상 설정
# ============================================================

RED = "#F34E67"
BLUE = "#386BFA"
GREEN = "#19B58A"
YELLOW = "#FFC83D"
PINK = "#E968C6"

INK = "#202943"
MUTED = "#66748D"


# ============================================================
# 03. 공통 디자인
#
# st.markdown(unsafe_allow_html=True) 대신
# st.html()을 사용합니다.
# ============================================================

st.html("""
<style>

/* ---------------------------------------------------------
   GLOBAL
--------------------------------------------------------- */

:root {
    --red: #F34E67;
    --blue: #386BFA;
    --green: #19B58A;
    --yellow: #FFC83D;
    --pink: #E968C6;

    --ink: #202943;
    --muted: #66748D;
    --line: #DFE6F3;
}

.stApp {
    background: #FAFCFF;
    color: var(--ink);
}

/* 화면 최상단의 5색 바 */

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

[data-testid="stHeader"] {
    background: #FAFCFF;
}

[data-testid="stSidebar"] {
    background: #F0F4FF;
    border-right: 1px solid #DBE4F4;
}

.block-container {
    max-width: 1320px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

h1, h2, h3 {
    color: #202943 !important;
    letter-spacing: -.045em;
}

[data-testid="stMarkdownContainer"] p {
    line-height: 1.75;
}


/* ---------------------------------------------------------
   STREAMLIT COMPONENTS
--------------------------------------------------------- */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DFE6F4;

    border-radius: 15px;
    padding: 22px;

    box-shadow: 0 8px 24px #344B7810;
}

[data-testid="stMetricLabel"] {
    color: #66748D;
}

[data-testid="stMetricValue"] {
    color: #202943;
    font-weight: 850;
    letter-spacing: -.05em;
}

[data-testid="stDataFrame"],
[data-testid="stDataEditor"],
[data-testid="stPlotlyChart"] {
    border: 1px solid #DFE6F4;
    border-radius: 13px;
    overflow: hidden;
}

.stButton button,
.stDownloadButton button {
    border-radius: 10px;
    font-weight: 700;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(
        90deg,
        #386BFA,
        #E968C6
    );
}


/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

.nf-hero {
    position: relative;
    overflow: hidden;

    background: linear-gradient(
        112deg,
        #EEF4FF 0%,
        #F0FBF9 48%,
        #FFF0F9 100%
    );

    border: 1px solid #DCE5FF;
    border-radius: 24px;

    padding: 44px;
    margin: 12px 0 30px;
}

.nf-hero::after {
    content: "✦";

    position: absolute;
    right: 30px;
    top: -55px;

    font-size: 240px;
    color: #E968C61A;

    pointer-events: none;
}

.nf-hero-kicker {
    position: relative;
    z-index: 1;

    color: #386BFA;

    font-size: 12px;
    font-weight: 900;
    letter-spacing: .16em;
}

.nf-hero-title {
    position: relative;
    z-index: 1;

    font-size: clamp(35px, 5.4vw, 66px);
    font-weight: 900;

    letter-spacing: -.075em;
    line-height: 1.16;

    margin: 17px 0;

    color: #202943;
}

.nf-gradient-text {
    background: linear-gradient(
        95deg,
        #F34E67,
        #E968C6,
        #386BFA
    );

    background-clip: text;
    -webkit-background-clip: text;

    color: transparent;
    -webkit-text-fill-color: transparent;
}

.nf-hero-lead {
    position: relative;
    z-index: 1;

    max-width: 700px;

    color: #52617C;
    font-size: 15px;
    line-height: 1.9;
}


/* ---------------------------------------------------------
   FIVE COLOR STRIP
--------------------------------------------------------- */

.nf-color-strip {
    display: flex;
    gap: 6px;

    max-width: 280px;
    margin-top: 26px;
}

.nf-color-strip i {
    display: block;
    flex: 1;

    height: 8px;
    border-radius: 99px;
}


/* ---------------------------------------------------------
   SECTION
--------------------------------------------------------- */

.nf-section {
    display: flex;
    align-items: baseline;
    gap: 17px;

    border-top: 1px solid #DCE4F1;

    padding-top: 22px;
    margin-top: 52px;
    margin-bottom: 8px;
}

.nf-section-num {
    color: #386BFA;

    font-size: 12px;
    font-weight: 900;
}

.nf-section-title {
    color: #202943;

    font-size: clamp(23px, 3vw, 32px);
    font-weight: 850;

    letter-spacing: -.05em;
}

.nf-section-desc {
    color: #68758C;

    font-size: 14px;
    line-height: 1.8;

    margin: 0 0 23px;
}


/* ---------------------------------------------------------
   STAT CARDS
--------------------------------------------------------- */

.nf-stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
}

.nf-stat {
    background: #FFFFFF;

    border: 1px solid #E0E7F3;
    border-top: 5px solid var(--accent);

    border-radius: 17px;

    padding: 22px 20px;

    box-shadow: 0 9px 22px #273F6A0D;
}

.nf-stat-label {
    color: #5F6C82;

    font-size: 13px;
    font-weight: 650;
}

.nf-stat-value {
    color: #202943;

    font-size: clamp(25px, 3vw, 40px);
    font-weight: 900;

    line-height: 1.3;
    letter-spacing: -.055em;

    margin-top: 12px;

    font-variant-numeric: tabular-nums;
}

.nf-stat-foot {
    color: var(--accent);

    font-size: 11px;
    font-weight: 850;

    letter-spacing: .08em;

    margin-top: 9px;
}


/* ---------------------------------------------------------
   INFORMATION NOTE
--------------------------------------------------------- */

.nf-note {
    background: #F0F4FF;

    border: 1px solid #E0E8FF;
    border-left: 4px solid #386BFA;

    border-radius: 0 12px 12px 0;

    padding: 18px 20px;
    margin: 20px 0;

    color: #44536E;

    font-size: 14px;
    line-height: 1.85;
}


/* ---------------------------------------------------------
   MISSION CARD
--------------------------------------------------------- */

.nf-mission {
    background: linear-gradient(
        115deg,
        #FFF7DF,
        #FFF0F8
    );

    border: 1px solid #F3DFB6;
    border-radius: 16px;

    padding: 22px 24px;
    margin: 20px 0;
}

.nf-mission-label {
    color: #B77920;

    font-size: 11px;
    font-weight: 900;

    letter-spacing: .14em;
}

.nf-mission-title {
    color: #3D3A47;

    font-size: 19px;
    font-weight: 850;

    margin: 8px 0;
}

.nf-mission-text {
    color: #655C65;

    font-size: 13px;
    line-height: 1.75;
}


/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.nf-footer {
    margin-top: 70px;
    padding-top: 20px;

    border-top: 1px solid #DFE5F1;

    color: #8792A6;

    font-size: 11px;
    letter-spacing: .1em;
}


/* ---------------------------------------------------------
   RESPONSIVE
--------------------------------------------------------- */

@media(max-width: 860px) {
    .nf-stat-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media(max-width: 520px) {
    .nf-hero {
        padding: 29px 23px;
    }

    .nf-hero-title {
        font-size: 37px;
    }

    .nf-stat-grid {
        gap: 9px;
    }

    .nf-stat {
        padding: 16px 12px;
    }

    .nf-stat-value {
        font-size: 25px;
    }

    .nf-section {
        margin-top: 36px;
    }
}

</style>
""")


# ============================================================
# 04. 공통 디자인 함수
# ============================================================

def color_strip():
    return (
        '<div class="nf-color-strip">'
        '<i style="background:#F34E67"></i>'
        '<i style="background:#386BFA"></i>'
        '<i style="background:#19B58A"></i>'
        '<i style="background:#FFC83D"></i>'
        '<i style="background:#E968C6"></i>'
        '</div>'
    )


def section(number, title, description):
    st.html(
        '<div class="nf-section">'
        f'<span class="nf-section-num">{html.escape(number)}</span>'
        f'<span class="nf-section-title">{html.escape(title)}</span>'
        '</div>'
        '<div class="nf-section-desc">'
        f'{html.escape(description)}'
        '</div>'
    )


def info_note(content):
    st.html(
        '<div class="nf-note">'
        + content
        + '</div>'
    )


def mission_card(title, description):
    st.html(
        '<div class="nf-mission">'
        '<div class="nf-mission-label">'
        'MISSION ACTIVE / YELLOW'
        '</div>'
        '<div class="nf-mission-title">'
        f'{html.escape(title)}'
        '</div>'
        '<div class="nf-mission-text">'
        f'{html.escape(description)}'
        '</div>'
        '</div>'
    )


# ============================================================
# 05. 데이터 읽기
# ============================================================

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
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: "
            + ", ".join(missing)
        )

    return df


def describe_values(series):

    values = series.dropna()

    if values.empty:
        return "값 없음"

    if (
        pd.api.types.is_numeric_dtype(series)
        and values.nunique() > 2
    ):
        return (
            f"숫자 {values.min():g} "
            f"~ {values.max():g}"
        )

    kinds = sorted(
        values.astype(str).unique()
    )

    return (
        f"{len(kinds)}가지 · "
        + " / ".join(kinds)
    )


# ============================================================
# 06. 데이터 소개 화면
# ============================================================

def show_home():

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    st.html(
        '<div class="nf-hero">'
        '<div class="nf-hero-kicker">'
        '01 / NEURO FORCE · DATA ORIGIN'
        '</div>'
        '<div class="nf-hero-title">'
        '데이터 해독 작전,<br>'
        '<span class="nf-gradient-text">'
        '첫 번째 미션!'
        '</span>'
        '</div>'
        '<div class="nf-hero-lead">'
        '건강 기록 속에 숨겨진 의미를 찾아봅니다. '
        '열의 뜻을 이해하고, 원본 데이터를 살피고, '
        '다음 단계의 분석을 준비하세요.'
        '</div>'
        + color_strip()
        + '</div>'
    )

    # --------------------------------------------------------
    # 데이터 불러오기
    # --------------------------------------------------------

    try:
        df = load_data()

    except Exception as exc:

        st.error(
            "데이터를 불러오지 못했습니다."
        )

        with st.expander("오류 상세"):
            st.code(str(exc))

        if st.button("데이터 다시 불러오기"):
            load_data.clear()
            st.rerun()

        st.stop()

    # --------------------------------------------------------
    # SECTION 01 — DATA OVERVIEW
    # --------------------------------------------------------

    section(
        "01",
        "데이터의 기본 정보",
        "분석에 사용할 데이터의 규모를 확인합니다.",
    )

    total = len(df)

    stroke_count = int(
        df["stroke"].eq(1).sum()
    )

    stroke_rate = (
        stroke_count / total * 100
        if total
        else 0
    )

    st.html(
        '<div class="nf-stat-grid">'

        f'<div class="nf-stat" style="--accent:{RED}">'
        '<div class="nf-stat-label">전체 관측 인원</div>'
        f'<div class="nf-stat-value">{total:,}</div>'
        '<div class="nf-stat-foot">RED / RECORDS</div>'
        '</div>'

        f'<div class="nf-stat" style="--accent:{BLUE}">'
        '<div class="nf-stat-label">데이터 변수</div>'
        f'<div class="nf-stat-value">{df.shape[1]}</div>'
        '<div class="nf-stat-foot">BLUE / FEATURES</div>'
        '</div>'

        f'<div class="nf-stat" style="--accent:{GREEN}">'
        '<div class="nf-stat-label">뇌졸중 경험자</div>'
        f'<div class="nf-stat-value">{stroke_count:,}</div>'
        '<div class="nf-stat-foot">GREEN / POSITIVE</div>'
        '</div>'

        f'<div class="nf-stat" style="--accent:{PINK}">'
        '<div class="nf-stat-label">뇌졸중 경험 비율</div>'
        f'<div class="nf-stat-value">{stroke_rate:.2f}%</div>'
        '<div class="nf-stat-foot">PINK / RATE</div>'
        '</div>'

        '</div>'
    )

    info_note(
        "<b>정답 규칙</b> · "
        "stroke가 1이면 뇌졸중 경험 있음(양성), "
        "0이면 뇌졸중 경험 없음(음성)입니다."
    )

    # --------------------------------------------------------
    # SECTION 02 — DATA DICTIONARY
    # --------------------------------------------------------

    section(
        "02",
        "데이터 사전 해독 미션",
        "열 이름의 우리말 뜻을 직접 채워 보세요.",
    )

    mission_card(
        f"{len(df.columns)}개 열의 의미를 밝혀라!",
        "교재를 참고해 우리말 뜻을 작성하세요. "
        "완성한 데이터 사전은 CSV로 저장할 수 있습니다.",
    )

    if "nf_meanings" not in st.session_state:

        st.session_state.nf_meanings = {
            column: ""
            for column in df.columns
        }

    dictionary = pd.DataFrame([
        {
            "열 이름": column,
            "우리말 뜻": (
                st.session_state.nf_meanings.get(
                    column,
                    "",
                )
            ),
            "값의 종류": describe_values(
                df[column]
            ),
            "빈 값 개수": int(
                df[column].isna().sum()
            ),
        }

        for column in df.columns
    ])

    edited = st.data_editor(
        dictionary,
        key="nf_dictionary",
        hide_index=True,
        use_container_width=True,
        height=440,
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

        value = row["우리말 뜻"]

        st.session_state.nf_meanings[
            row["열 이름"]
        ] = (
            ""
            if pd.isna(value)
            else str(value)
        )

    completed = sum(
        bool(str(value).strip())
        for value in (
            st.session_state.nf_meanings.values()
        )
    )

    st.progress(
        completed / len(df.columns),
        text=(
            f"사전 해독 진행률 · "
            f"{completed}/{len(df.columns)}"
        ),
    )

    if completed == len(df.columns):
        st.success(
            "미션 완료! 모든 열의 뜻을 작성했습니다."
        )

    st.download_button(
        "데이터 사전 다운로드",
        data=edited.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="neuro_dictionary.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # SECTION 03 — DATA PREVIEW
    # --------------------------------------------------------

    section(
        "03",
        "원본 데이터 스캔",
        "가공하지 않은 원본의 첫 다섯 행을 확인합니다.",
    )

    st.dataframe(
        df.head(5),
        hide_index=True,
        use_container_width=True,
    )

    bmi_missing = int(
        df["bmi"].isna().sum()
    )

    info_note(
        "<b>스캔 결과</b> · "
        f"BMI가 비어 있는 기록은 "
        f"<b>{bmi_missing:,}개</b>입니다. "
        "이 값은 임의로 채우지 않았습니다."
    )

    with st.expander(
        "전체 열의 결측치 보기"
    ):

        # pandas reset_index(names=...) 사용하지 않음
        # 이전 TypeError 방지

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

    # --------------------------------------------------------
    # SECTION 04 — SOURCE
    # --------------------------------------------------------

    section(
        "04",
        "데이터의 출처 조사",
        "데이터의 출처와 한계를 확인합니다.",
    )

    source = st.text_area(
        "출처 기록",
        value=SOURCE_TEXT,
        height=180,
        key="nf_source",
    )

    st.download_button(
        "출처 메모 다운로드",
        data=source.encode("utf-8-sig"),
        file_name="neuro_source.txt",
        mime="text/plain",
    )

    info_note(
        "<b>데이터 윤리</b> · "
        "데이터의 수집 과정과 대표성은 "
        "충분히 확인되지 않았습니다. "
        "분석 결과를 실제 의료 진단이나 "
        "치료 판단에 사용하지 마세요."
    )

    # --------------------------------------------------------
    # SECTION 05 — NEXT MISSION
    # --------------------------------------------------------

    section(
        "05",
        "다음 작전 개시",
        "왼쪽 메뉴에서 다음 실습 페이지를 선택하세요.",
    )

    st.html(
        '<div class="nf-mission">'
        '<div class="nf-mission-label">'
        'NEXT EPISODE'
        '</div>'
        '<div class="nf-mission-title">'
        '02 · 데이터 탐색 작전'
        '</div>'
        '<div class="nf-mission-text">'
        '나이, 혈당, 건강 상태의 차이를 '
        '그래프로 조사합니다. '
        '왼쪽 메뉴에서 탐색을 선택하세요.'
        '</div>'
        '</div>'
    )

    st.html(
        '<div class="nf-footer">'
        'NEURO FORCE · DATA ORIGIN · '
        'EDUCATIONAL USE ONLY'
        '</div>'
    )


# ============================================================
# 07. 페이지 파일 이름 변환
#
# 정규표현식을 사용하지 않습니다.
#
# 1_탐색.py        → 탐색
# 2_분류_모델.py   → 분류 모델
# 3_모델_평가.py   → 모델 평가
# ============================================================

def page_title(path):

    name = path.stem

    # 이름 앞에 있는 숫자 제거
    # 정규표현식 대신 문자열 메서드 사용

    position = 0

    while (
        position < len(name)
        and name[position].isdigit()
    ):
        position += 1

    # 숫자 바로 뒤의 구분자 제거
    if position > 0:
        while (
            position < len(name)
            and name[position] in "_- "
        ):
            position += 1

        name = name[position:]

    # 단어 사이의 밑줄을 공백으로 변경
    name = name.replace("_", " ").strip()

    return name or path.stem


# ============================================================
# 08. Pages 폴더 자동 탐색
#
# 대문자 Pages 우선
# 없으면 소문자 pages도 검색
# ============================================================

def find_pages_folder():

    for folder_name in (
        "Pages",
        "pages",
    ):

        folder = ROOT / folder_name

        if folder.is_dir():
            return folder

    return None


def find_page_files():

    folder = find_pages_folder()

    if folder is None:
        return []

    files = [
        path
        for path in folder.glob("*.py")
        if (
            path.is_file()
            and not path.name.startswith("_")
        )
    ]

    # 파일 이름으로 정렬
    return sorted(
        files,
        key=lambda path: path.name,
    )


# ============================================================
# 09. 페이지 자동 등록
# ============================================================

home = st.Page(
    show_home,
    title="데이터 소개",
    icon="🏠",
    default=True,
)

page_list = [home]

page_files = find_page_files()

for page_file in page_files:

    # 파일 이름에서 메뉴 제목 생성
    title = page_title(page_file)

    # main.py 기준 상대 경로
    relative_path = (
        page_file
        .relative_to(ROOT)
        .as_posix()
    )

    # 페이지 종류에 따라 아이콘 지정
    if "분류" in title:
        icon = "🤖"

    elif "탐색" in title:
        icon = "🔎"

    else:
        icon = "📘"

    page_list.append(
        st.Page(
            relative_path,
            title=title,
            icon=icon,
        )
    )


# ============================================================
# 10. 사이드바 메뉴
# ============================================================

navigation = st.navigation(
    page_list,
    position="sidebar",
    expanded=True,
)

with st.sidebar:

    st.markdown("### NEURO FORCE")

    st.caption(
        "FIVE COLORS · ONE DATA MISSION"
    )

    st.divider()

    st.markdown("**PROJECT STATUS**")

    st.caption(
        f"등록된 실습 페이지: {len(page_files)}개"
    )

    st.caption(
        "데이터 소개 → 탐색 → 분류 모델"
    )

    st.divider()

    st.caption(
        "EDUCATIONAL DATA SCIENCE"
    )


# ============================================================
# 11. 선택한 페이지 실행
# ============================================================

navigation.run()
