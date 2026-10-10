
import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# NEURO FORCE | 데이터 탐색 작전
# main.py에서 페이지 설정과 공통 CSS를 관리합니다.
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/main/data/stroke.csv"
)

RED = "#F34E67"
BLUE = "#386BFA"
GREEN = "#19B58A"
YELLOW = "#FFC83D"
PINK = "#E968C6"

GROUP_ORDER = ["경험 없음", "경험 있음"]

GROUP_COLORS = {
    "경험 없음": BLUE,
    "경험 있음": RED,
}


# ============================================================
# 화면 구성
# ============================================================

def section(number, title, description, color):
    st.html(
        '<div class="nf-section">'
        f'<span class="nf-section-num" style="color:{color}">'
        f'{html.escape(number)}</span>'
        f'<span class="nf-section-title">'
        f'{html.escape(title)}</span>'
        '</div>'
        f'<div class="nf-section-desc">'
        f'{html.escape(description)}</div>'
    )


def mission(title, description, color=YELLOW):
    st.html(
        '<div class="nf-mission">'
        f'<div class="nf-mission-label" style="color:{color}">'
        'MISSION ACTIVE</div>'
        f'<div class="nf-mission-title">{html.escape(title)}</div>'
        f'<div class="nf-mission-text">'
        f'{html.escape(description)}</div>'
        '</div>'
    )


def note(content):
    st.html(
        '<div class="nf-note">'
        + content
        + '</div>'
    )


def chart_style(fig, y_title="사람 수"):
    fig.update_layout(
        template="plotly_white",
        height=385,
        margin=dict(l=28, r=25, t=30, b=37),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Arial, sans-serif",
            color="#35435B",
            size=12,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#DDE4F0",
    )

    fig.update_yaxes(
        title=y_title,
        gridcolor="#EDF1F8",
        zeroline=False,
    )

    return fig


# ============================================================
# 데이터 읽기
# ============================================================

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "id", "gender", "age",
        "Residence_type", "avg_glucose_level",
        "hypertension", "heart_disease",
        "stroke", "bmi", "smoking_status",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "필수 열 누락: " + ", ".join(missing)
        )

    return df


def percentage(cases, people):
    return cases / people * 100 if people else float("nan")


def risk_table(data, column, label):
    result = (
        data.groupby(column)["stroke"]
        .agg(
            사람_수="size",
            경험자_수=lambda s: int(s.eq(1).sum()),
        )
        .reindex([0, 1], fill_value=0)
        .reset_index()
    )

    result["뇌졸중 비율(%)"] = (
        result["경험자_수"]
        .div(result["사람_수"].replace(0, float("nan")))
        * 100
    )

    result["구분"] = result[column].map({
        0: f"{label} 없음",
        1: f"{label} 있음",
    })

    return result[
        ["구분", "사람_수", "경험자_수", "뇌졸중 비율(%)"]
    ]


def risk_chart(data, color):
    shown = data[data["사람_수"] > 0]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=shown["구분"],
            y=shown["뇌졸중 비율(%)"],
            marker_color=[
                BLUE if "없음" in x else color
                for x in shown["구분"]
            ],
            text=[
                f"{v:.2f}%"
                for v in shown["뇌졸중 비율(%)"]
            ],
            textposition="outside",
            customdata=shown[
                ["사람_수", "경험자_수"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "뇌졸중 경험 비율: %{y:.2f}%<br>"
                "전체: %{customdata[0]}명<br>"
                "경험자: %{customdata[1]}명"
                "<extra></extra>"
            ),
        )
    )

    chart_style(fig, "뇌졸중 경험 비율 (%)")

    if not shown.empty:
        maximum = float(shown["뇌졸중 비율(%)"].max())
        fig.update_yaxes(
            range=[0, max(5, maximum * 1.35)]
        )

    return fig


# ============================================================
# 데이터와 필터
# ============================================================

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


with st.sidebar:
    st.divider()
    st.markdown("**SCAN CONTROL / 분석 조건**")
    st.caption("변경한 조건은 모든 그래프에 적용됩니다.")

    if st.button("필터 초기화", use_container_width=True):
        for key in [
            "nf_gender", "nf_residence",
            "nf_age", "nf_bins", "nf_mean",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    gender = st.selectbox(
        "성별",
        ["전체"] + sorted(
            df["gender"].dropna().unique().tolist()
        ),
        key="nf_gender",
    )

    residence = st.selectbox(
        "거주 지역",
        ["전체"] + sorted(
            df["Residence_type"].dropna().unique().tolist()
        ),
        key="nf_residence",
    )

    max_age = int(df["age"].max())

    age_range = st.slider(
        "나이 범위",
        0, max_age, (0, max_age),
        key="nf_age",
    )

    bins = st.slider(
        "분포 그래프 구간 수",
        10, 70, 40, 5,
        key="nf_bins",
    )

    show_mean = st.checkbox(
        "평균선 표시",
        value=True,
        key="nf_mean",
    )


filtered = df.copy()

if gender != "전체":
    filtered = filtered[filtered["gender"] == gender]

if residence != "전체":
    filtered = filtered[
        filtered["Residence_type"] == residence
    ]

filtered = filtered[
    filtered["age"].between(*age_range)
].copy()


# ============================================================
# HERO
# ============================================================

st.html(
    '<div class="nf-hero">'
    '<div class="nf-hero-kicker">'
    '02 / NEURO FORCE · SCAN MODE'
    '</div>'
    '<div class="nf-hero-title">'
    '데이터 스캔 작전,<br>'
    '<span class="nf-gradient-text">패턴을 찾아라!</span>'
    '</div>'
    '<div class="nf-hero-lead">'
    '나이와 혈당의 분포를 조사하고, '
    '두 집단의 차이를 비교하고, '
    '건강 상태와 결측치를 분석합니다. '
    '사이드바의 필터를 바꿔 분석 결과를 확인하세요.'
    '</div>'
    '<div class="nf-color-strip">'
    '<i style="background:#F34E67"></i>'
    '<i style="background:#386BFA"></i>'
    '<i style="background:#19B58A"></i>'
    '<i style="background:#FFC83D"></i>'
    '<i style="background:#E968C6"></i>'
    '</div></div>'
)

if filtered.empty:
    st.warning("현재 필터에 해당하는 사람이 없습니다.")
    st.info("왼쪽 사이드바에서 필터를 초기화해 주세요.")
    st.stop()


# ============================================================
# 통계 카드
# ============================================================

filtered["뇌졸중"] = filtered["stroke"].map({
    0: "경험 없음",
    1: "경험 있음",
})

total = len(filtered)
cases = int(filtered["stroke"].eq(1).sum())
rate = percentage(cases, total)
bmi_missing = int(filtered["bmi"].isna().sum())

st.html(
    '<div class="nf-stat-grid">'
    f'<div class="nf-stat" style="--accent:{RED}">'
    '<div class="nf-stat-label">현재 분석 대상</div>'
    f'<div class="nf-stat-value">{total:,}</div>'
    '<div class="nf-stat-foot">RED / SELECTED</div></div>'
    f'<div class="nf-stat" style="--accent:{BLUE}">'
    '<div class="nf-stat-label">뇌졸중 경험자</div>'
    f'<div class="nf-stat-value">{cases:,}</div>'
    '<div class="nf-stat-foot">BLUE / POSITIVE</div></div>'
    f'<div class="nf-stat" style="--accent:{GREEN}">'
    '<div class="nf-stat-label">경험 비율</div>'
    f'<div class="nf-stat-value">{rate:.2f}%</div>'
    '<div class="nf-stat-foot">GREEN / RATE</div></div>'
    f'<div class="nf-stat" style="--accent:{PINK}">'
    '<div class="nf-stat-label">BMI 빈 값</div>'
    f'<div class="nf-stat-value">{bmi_missing:,}</div>'
    '<div class="nf-stat-foot">PINK / MISSING</div></div>'
    '</div>'
)

st.caption(
    f"원본 {len(df):,}명 중 {total:,}명을 선택했습니다. "
    "아래 결과는 모두 선택된 사람 기준입니다."
)

with st.sidebar:
    st.download_button(
        "선택 데이터 다운로드",
        data=filtered.drop(columns=["뇌졸중"]).to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="neuro_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# 01. 히스토그램
# ============================================================

section(
    "01",
    "RED & BLUE / 분포 스캔",
    "나이와 평균 혈당의 분포를 조사합니다.",
    RED,
)

left, right = st.columns(2, gap="large")

for target, col, title, color in [
    (left, "age", "나이 분포", RED),
    (right, "avg_glucose_level", "평균 혈당 분포", BLUE),
]:
    with target:
        st.markdown(f"#### {title}")

        fig = px.histogram(
            filtered,
            x=col,
            nbins=bins,
            color_discrete_sequence=[color],
            labels={col: title},
        )

        fig.update_traces(
            marker_line_color="white",
            marker_line_width=0.5,
            opacity=0.9,
        )

        if show_mean and filtered[col].notna().any():
            fig.add_vline(
                x=filtered[col].mean(),
                line_dash="dash",
                line_color=GREEN,
                line_width=2,
            )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True,
        )

        st.caption(
            f"평균 {filtered[col].mean():.2f} · "
            f"중앙값 {filtered[col].median():.2f}"
        )


# ============================================================
# 02. 집단별 상자그림
# ============================================================

section(
    "02",
    "BLUE & PINK / 두 집단 비교",
    "뇌졸중 경험 여부에 따른 나이와 혈당 차이를 확인합니다.",
    BLUE,
)

left, right = st.columns(2, gap="large")

for target, col, title in [
    (left, "age", "나이 비교"),
    (right, "avg_glucose_level", "평균 혈당 비교"),
]:
    with target:
        st.markdown(f"#### {title}")

        fig = px.box(
            filtered,
            x="뇌졸중",
            y=col,
            color="뇌졸중",
            category_orders={"뇌졸중": GROUP_ORDER},
            color_discrete_map=GROUP_COLORS,
            labels={col: title},
        )

        fig.update_traces(
            line_width=2,
        )

        st.plotly_chart(
            chart_style(fig, title),
            use_container_width=True,
        )


comparison = (
    filtered.groupby("뇌졸중")
    .agg(
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
        고혈압_비율=("hypertension", "mean"),
        심장병_비율=("heart_disease", "mean"),
        사람_수=("id", "size"),
    )
    .reindex(GROUP_ORDER)
    .reset_index()
)

comparison["고혈압_비율"] *= 100
comparison["심장병_비율"] *= 100
comparison["사람_수"] = (
    comparison["사람_수"].fillna(0).astype(int)
)

st.markdown("#### 집단별 상세 비교표")

st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True,
    column_config={
        "평균_나이": st.column_config.NumberColumn(
            "평균 나이", format="%.2f"
        ),
        "평균_혈당": st.column_config.NumberColumn(
            "평균 혈당", format="%.2f"
        ),
        "고혈압_비율": st.column_config.NumberColumn(
            "고혈압 비율", format="%.1f%%"
        ),
        "심장병_비율": st.column_config.NumberColumn(
            "심장병 비율", format="%.1f%%"
        ),
    },
)

if filtered["stroke"].nunique() < 2:
    st.info(
        "현재 선택된 데이터에는 한 집단만 존재합니다. "
        "두 집단을 비교하려면 필터를 넓혀 주세요."
    )

st.download_button(
    "집단 비교표 다운로드",
    data=comparison.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="neuro_group_comparison.csv",
    mime="text/csv",
)


# ============================================================
# 03. 고혈압 / 심장병
# ============================================================

section(
    "03",
    "GREEN / 건강 상태 분석",
    "고혈압과 심장병 여부에 따른 뇌졸중 경험 비율을 비교합니다.",
    GREEN,
)

hypertension = risk_table(
    filtered, "hypertension", "고혈압"
)

heart = risk_table(
    filtered, "heart_disease", "심장병"
)

left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### 고혈압 여부")
    st.plotly_chart(
        risk_chart(hypertension, RED),
        use_container_width=True,
    )

with right:
    st.markdown("#### 심장병 여부")
    st.plotly_chart(
        risk_chart(heart, PINK),
        use_container_width=True,
    )

with st.expander("계산에 사용된 사람 수 확인"):
    a, b = st.columns(2)
    with a:
        st.markdown("**고혈압 집단**")
        st.dataframe(
            hypertension,
            hide_index=True,
            use_container_width=True,
        )
    with b:
        st.markdown("**심장병 집단**")
        st.dataframe(
            heart,
            hide_index=True,
            use_container_width=True,
        )

note(
    "<b>해석 규칙</b> · "
    "두 집단의 비율이 다르더라도 "
    "그 차이만으로 원인과 결과를 단정할 수 없습니다."
)


# ============================================================
# 04. BMI 결측치
# ============================================================

section(
    "04",
    "YELLOW / 결측치 수색",
    "BMI가 비어 있는 사람과 전체의 뇌졸중 경험 비율을 비교합니다.",
    "#C69213",
)

missing = filtered[filtered["bmi"].isna()]
missing_cases = int(missing["stroke"].eq(1).sum())

bmi_table = pd.DataFrame([
    {
        "구분": "BMI가 비어 있는 사람",
        "사람 수": len(missing),
        "뇌졸중 경험자": missing_cases,
        "뇌졸중 비율(%)": percentage(
            missing_cases, len(missing)
        ),
    },
    {
        "구분": "전체",
        "사람 수": total,
        "뇌졸중 경험자": cases,
        "뇌졸중 비율(%)": rate,
    },
])

st.dataframe(
    bmi_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "뇌졸중 비율(%)": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
    },
)

if missing.empty:
    st.info("현재 선택한 데이터에는 BMI 결측치가 없습니다.")
else:
    st.caption(
        f"선택한 사람 중 BMI 결측치 비중: "
        f"{len(missing) / total * 100:.2f}%"
    )

mission(
    "빈 값도 하나의 단서!",
    "BMI가 없는 사람을 무조건 삭제하거나 "
    "임의의 숫자로 채우면 분석 결과가 달라질 수 있습니다.",
)


# ============================================================
# 05. 흡연 상태
# ============================================================

section(
    "05",
    "PINK / 흡연 상태 조사",
    "흡연 상태의 종류별 인원과 비중을 살펴봅니다.",
    PINK,
)

smoking = (
    filtered["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

smoking["비율(%)"] = (
    smoking["사람 수"] / total * 100
)

left, right = st.columns([1.2, 1], gap="large")

with left:
    st.dataframe(
        smoking,
        hide_index=True,
        use_container_width=True,
        column_config={
            "비율(%)": st.column_config.ProgressColumn(
                "비율 (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )

    st.download_button(
        "흡연 상태 집계 다운로드",
        data=smoking.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="neuro_smoking.csv",
        mime="text/csv",
    )

with right:
    note(
        "<b>Unknown은 빈 값일까?</b><br>"
        "아닙니다. Unknown은 흡연 상태를 모른다는 "
        "뜻으로 적힌 문자 범주입니다. "
        "BMI의 NaN과는 구분해야 합니다."
    )


# ============================================================
# 06. 탐색 작전 체크리스트
# ============================================================

section(
    "06",
    "FINAL MISSION / 탐색 완료",
    "관찰 결과를 교재에 기록했다면 체크하세요.",
    BLUE,
)

tasks = [
    "나이와 평균 혈당 분포 확인",
    "뇌졸중 경험별 나이 비교",
    "뇌졸중 경험별 혈당 비교",
    "고혈압별 뇌졸중 비율 확인",
    "심장병별 뇌졸중 비율 확인",
    "BMI 결측치 집단 비교",
    "흡연 상태 집계 확인",
]

done = 0

for i, task in enumerate(tasks):
    if st.checkbox(task, key=f"nf_explore_task_{i}"):
        done += 1

st.progress(
    done / len(tasks),
    text=f"탐색 작전 완료율 · {done}/{len(tasks)}",
)

if done == len(tasks):
    st.success(
        "탐색 작전 완료! 이제 분류 모델 페이지로 이동할 수 있습니다."
    )

st.html(
    '<div class="nf-footer">'
    'NEURO FORCE · SCAN MODE · EDUCATIONAL USE ONLY'
    '</div>'
)
