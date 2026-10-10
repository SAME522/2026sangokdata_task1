
from pathlib import Path
import re
import html

import pandas as pd
import streamlit as st


# ============================================================
# NEURO FORCE | 메인 진입점 + 공통 디자인
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

RED = "#F34E67"
BLUE = "#386BFA"
GREEN = "#19B58A"
YELLOW = "#FFC83D"
PINK = "#E968C6"


# ============================================================
# 모든 페이지에 적용되는 공통 CSS
# st.html()을 사용해 HTML 코드 노출 문제 방지
# ============================================================

st.html("""
<style>
:root {
    --nf-red: #F34E67;
    --nf-blue: #386BFA;
    --nf-green: #19B58A;
    --nf-yellow: #FFC83D;
    --nf-pink: #E968C6;
    --nf-ink: #202943;
    --nf-muted: #66748D;
    --nf-line: #DDE5F3;
}

.stApp {
    background: #FAFCFF;
    color: var(--nf-ink);
}

.stApp::before {
    content: "";
    position: fixed;
    left: 0;
    top: 0;
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
    border-right: 1px solid #DAE3F4;
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

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DEE6F4;
    border-radius: 16px;
    padding: 22px;
    min-height: 132px;
    box-shadow: 0 8px 24px #344B7810;
}

[data-testid="stMetricLabel"] {
    color: #64718B;
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
    font-weight: 750;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(
        90deg,
        #386BFA,
        #E968C6
    );
}

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
    color: #386BFA;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: .16em;
    position: relative;
    z-index: 1;
}

.nf-hero-title {
    font-size: clamp(35px, 5.4vw, 66px);
    font-weight: 900;
    letter-spacing: -.075em;
    line-height: 1.16;
    margin: 17px 0;
    color: #202943;
    position: relative;
    z-index: 1;
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
    max-width: 700px;
    color: #52617C;
    font-size: 15px;
    line-height: 1.9;
    position: relative;
    z-index: 1;
}

.nf-color-strip {
    display: flex;
    gap: 6px;
    margin-top: 26px;
    max-width: 280px;
}

.nf-color-strip i {
    height: 8px;
    border-radius: 99px;
    flex: 1;
}

.nf-section {
    display: flex;
    gap: 17px;
    align-items: baseline;
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
    font-size: 14px;
    line-height: 1.8;
    color: #68758C;
    margin: 0 0 23px;
}

.nf-stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
}

.nf-stat {
    background: white;
    border: 1px solid #E0E7F3;
    border-radius: 17px;
    padding: 22px 20px;
    border-top: 5px solid var(--accent);
    box-shadow: 0 9px 22px #273F6A0D;
}

.nf-stat-label {
    font-size: 13px;
    color: #5F6C82;
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
    margin-top: 9px;
    font-size: 11px;
    color: var(--accent);
    font-weight: 850;
    letter-spacing: .08em;
}

.nf-note {
    background: #F0F4FF;
    border: 1px solid #E0E8FF;
    border-left: 4px solid #386BFA;
    padding: 18px 20px;
    border-radius: 0 12px 12px 0;
    color: #44536E;
    font-size: 14px;
    line-height: 1.85;
    margin: 20px 0;
}

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

.nf-footer {
    margin-top: 70px;
    border-top: 1px solid #DFE5F1;
    padding-top: 20px;
    color: #8792A6;
    font-size: 11px;
    letter-spacing: .1em;
}

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
        f'<span class="nf-section-num">{number}</span>'
        f'<span class="nf-section-title">{html.escape(title)}</span>'
        '</div>'
        f'<div class="nf-section-desc">{html.escape(description)}</div>'
    )


def stat_card(label, value, accent, foot):
    st.html(
        f'<div class="nf-stat" style="--accent:{accent}">'
        f'<div class="nf-stat-label">{html.escape(label)}</div>'
        f'<div class="nf-stat-value">{html.escape(str(value))}</div>'
        f'<div class="nf-stat-foot">{html.escape(foot)}</div>'
        '</div>'
    )


def note(content):
    st.html(
        '<div class="nf-note">'
        + content
        + '</div>'
    )


@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "id", "gender", "age",
        "hypertension", "heart_disease",
        "Residence_type", "avg_glucose_level",
        "bmi", "smoking_status", "stroke",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("필수 열 누락: " + ", ".join(missing))

    return df


def describe_values(series):
    values = series.dropna()

    if values.empty:
        return "값 없음"

    if (
        pd.api.types.is_numeric_dtype(series)
        and values.nunique() > 2
    ):
        return f"숫자 {values.min():g} ~ {values.max():g}"

    kinds = sorted(values.astype(str).unique())
    return f"{len(kinds)}가지 · " + " / ".join(kinds)


# ============================================================
# 첫 화면
# ============================================================

def show_home():
    st.html(
        '<div class="nf-hero">'
        '<div class="nf-hero-kicker">'
        '01 / NEURO FORCE · DATA ORIGIN'
        '</div>'
        '<div class="nf-hero-title">'
        '데이터 해독 작전,<br>'
        '<span class="nf-gradient-text">첫 번째 미션!</span>'
        '</div>'
        '<div class="nf-hero-lead">'
        '5,110명의 건강 기록 속에 숨겨진 의미를 찾아봅니다. '
        '열의 뜻을 이해하고, 원본 데이터를 살피고, '
        '다음 단계의 분석을 준비하세요.'
        '</div>'
        + color_strip()
        + '</div>'
    )

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

    # --------------------------------------------------------
    # SECTION 01: 숫자 카드
    # --------------------------------------------------------

    section(
        "01",
        "데이터의 기본 정보",
        "프로젝트에 사용되는 데이터의 규모부터 확인합니다.",
    )

    total = len(df)
    stroke_count = int(df["stroke"].eq(1).sum())
    stroke_rate = (
        stroke_count / total * 100 if total else 0
    )

    st.html(
        '<div class="nf-stat-grid">'
        f'<div class="nf-stat" style="--accent:{RED}">'
        '<div class="nf-stat-label">전체 관측 인원</div>'
        f'<div class="nf-stat-value">{total:,}</div>'
        '<div class="nf-stat-foot">RED / RECORDS</div></div>'
        f'<div class="nf-stat" style="--accent:{BLUE}">'
        '<div class="nf-stat-label">데이터 변수</div>'
        f'<div class="nf-stat-value">{df.shape[1]}</div>'
        '<div class="nf-stat-foot">BLUE / FEATURES</div></div>'
        f'<div class="nf-stat" style="--accent:{GREEN}">'
        '<div class="nf-stat-label">뇌졸중 경험자</div>'
        f'<div class="nf-stat-value">{stroke_count:,}</div>'
        '<div class="nf-stat-foot">GREEN / POSITIVE</div></div>'
        f'<div class="nf-stat" style="--accent:{PINK}">'
        '<div class="nf-stat-label">뇌졸중 경험 비율</div>'
        f'<div class="nf-stat-value">{stroke_rate:.2f}%</div>'
        '<div class="nf-stat-foot">PINK / RATE</div></div>'
        '</div>'
    )

    note(
        "<b>정답 규칙</b> · "
        "stroke가 1이면 뇌졸중 경험 있음(양성), "
        "0이면 뇌졸중 경험 없음(음성)입니다."
    )

    # --------------------------------------------------------
    # SECTION 02: 데이터 사전
    # --------------------------------------------------------

    section(
        "02",
        "데이터 사전 해독 미션",
        "열 이름의 우리말 뜻을 직접 채워 보세요.",
    )

    st.html(
        '<div class="nf-mission">'
        '<div class="nf-mission-label">MISSION / YELLOW</div>'
        '<div class="nf-mission-title">12개 열의 의미를 밝혀라!</div>'
        '<div class="nf-mission-text">'
        '교재를 참고해 우리말 뜻을 작성하세요. '
        '완성한 데이터 사전은 CSV로 내려받을 수 있습니다.'
        '</div></div>'
    )

    if "nf_meanings" not in st.session_state:
        st.session_state.nf_meanings = {
            name: "" for name in df.columns
        }

    # 최초 표 생성. 이후에는 편집 위젯 자체가 값을 유지합니다.
    dictionary = pd.DataFrame([
        {
            "열 이름": name,
            "우리말 뜻": st.session_state.nf_meanings.get(name, ""),
            "값의 종류": describe_values(df[name]),
            "빈 값 개수": int(df[name].isna().sum()),
        }
        for name in df.columns
    ])

    edited = st.data_editor(
        dictionary,
        hide_index=True,
        use_container_width=True,
        height=440,
        num_rows="fixed",
        key="nf_dictionary",
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn(
                width="medium"
            ),
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻 · 직접 입력",
                width="large",
            ),
            "값의 종류": st.column_config.TextColumn(
                width="large"
            ),
            "빈 값 개수": st.column_config.NumberColumn(
                format="%d"
            ),
        },
    )

    for _, row in edited.iterrows():
        value = row["우리말 뜻"]
        st.session_state.nf_meanings[row["열 이름"]] = (
            "" if pd.isna(value) else str(value)
        )

    completed = sum(
        bool(str(v).strip())
        for v in st.session_state.nf_meanings.values()
    )

    st.progress(
        completed / len(df.columns),
        text=f"사전 해독 진행률 · {completed}/{len(df.columns)}",
    )

    if completed == len(df.columns):
        st.success("미션 완료! 모든 열의 뜻을 작성했습니다.")

    st.download_button(
        "데이터 사전 다운로드",
        data=edited.to_csv(index=False).encode("utf-8-sig"),
        file_name="neuro_dictionary.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # SECTION 03: 원본 데이터
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

    bmi_missing = int(df["bmi"].isna().sum())

    note(
        "<b>스캔 결과</b> · "
        f"BMI가 비어 있는 기록은 <b>{bmi_missing:,}개</b>입니다. "
        "이 값은 임의로 채우지 않았습니다."
    )

    with st.expander("전체 열의 결측치 보기"):
        missing_table = pd.DataFrame({
            "열 이름": df.columns,
            "빈 값 개수": df.isna().sum().to_numpy(),
        })

        st.dataframe(
            missing_table,
            hide_index=True,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # SECTION 04: 출처
    # --------------------------------------------------------

    section(
        "04",
        "데이터의 출처 조사",
        "데이터가 어디서 왔는지 기록하고 신뢰성의 한계를 이해합니다.",
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

    note(
        "<b>데이터 윤리</b> · "
        "이 데이터의 수집 과정과 대표성은 충분히 확인되지 않았습니다. "
        "분석 결과를 실제 의료 진단이나 치료 판단에 사용하지 마세요."
    )

    # --------------------------------------------------------
    # SECTION 05: 다음 단계
    # --------------------------------------------------------

    section(
        "05",
        "다음 작전 개시",
        "왼쪽 메뉴에서 탐색 또는 분류 모델로 이동하세요.",
    )

    st.html(
        '<div class="nf-mission">'
        '<div class="nf-mission-label">NEXT EPISODE</div>'
        '<div class="nf-mission-title">02 · 데이터 탐색 작전</div>'
        '<div class="nf-mission-text">'
        '나이, 혈당, 건강 상태의 차이를 그래프로 조사합니다. '
        '왼쪽 페이지 메뉴에서 탐색을 선택하세요.'
        '</div></div>'
    )

    st.html(
        '<div class="nf-footer">'
        'NEURO FORCE · DATA ORIGIN · EDUCATIONAL USE ONLY'
        '</div>'
    )


# ============================================================
# 자동 페이지 탐색
# ============================================================

def page_title(path):
    name = re.sub(r"^\\d+[_\\- ]*", "", path.stem)
    return name.replace("_", " ")


home = st.Page(
    show_home,
    title="데이터 소개",
    icon="🏠",
    default=True,
)

pages = [home]

# 사용자가 지정한 대문자 Pages를 우선합니다.
page_folder = None

for folder_name in ("Pages", "pages"):
    candidate = ROOT / folder_name
    if candidate.is_dir():
        page_folder = candidate
        break

if page_folder is not None:
    for path in sorted(page_folder.glob("*.py")):
        if not path.is_file() or path.name.startswith("_"):
            continue

        name = page_title(path)

        pages.append(
            st.Page(
                path.relative_to(ROOT).as_posix(),
                title=name,
                icon=(
                    "🤖" if "분류" in name
                    else "🔎"
                ),
            )
        )

navigation = st.navigation(
    pages,
    position="sidebar",
    expanded=True,
)

with st.sidebar:
    st.markdown("### NEURO FORCE")
    st.caption("FIVE COLORS · ONE DATA MISSION")
    st.divider()

navigation.run()
