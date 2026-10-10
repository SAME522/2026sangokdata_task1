
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# NEURO LAB — 데이터 탐색
# 공통 스타일과 페이지 메뉴는 main.py에서 관리
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/main/data/stroke.csv"
)

BLUE = "#3559D5"
ORANGE = "#D77C51"
GREEN = "#478779"


# =========================================================
# 데이터 읽기
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
        "Residence_type",
        "age",
        "avg_glucose_level",
        "stroke",
        "hypertension",
        "heart_disease",
        "bmi",
        "smoking_status",
    ]

    missing = [
        col
        for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: "
            + ", ".join(missing)
        )

    return data


# =========================================================
# 공통 분석 함수
# =========================================================

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


def style_chart(fig, ylabel="사람 수"):

    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(
            l=28,
            r=25,
            t=25,
            b=30,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(
            size=12,
            color="#4B534B",
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#DDE1D9",
    )

    fig.update_yaxes(
        title=ylabel,
        gridcolor="#ECEEE9",
        zeroline=False,
    )

    return fig


def pct(cases, people):

    if people == 0:
        return float("nan")

    return cases / people * 100


def risk_table(data, column, label):

    grouped = (
        data.groupby(column)["stroke"]
        .agg(
            사람_수="size",
            경험자_수=lambda s: int(
                s.eq(1).sum()
            ),
        )
        .reindex(
            [0, 1],
            fill_value=0,
        )
        .reset_index()
    )

    grouped["뇌졸중 비율(%)"] = (
        grouped["경험자_수"]
        .div(
            grouped["사람_수"].replace(
                0,
                float("nan"),
            )
        )
        * 100
    )

    grouped["구분"] = grouped[column].map({
        0: label + " 없음",
        1: label + " 있음",
    })

    return grouped[
        [
            "구분",
            "사람_수",
            "경험자_수",
            "뇌졸중 비율(%)",
        ]
    ]


def risk_chart(table):

    valid = table.loc[
        table["사람_수"] > 0
    ]

    fig = go.Figure(
        go.Bar(
            x=valid["구분"],
            y=valid["뇌졸중 비율(%)"],
            marker_color=[
                BLUE if "없음" in name else ORANGE
                for name in valid["구분"]
            ],
            text=[
                f"{v:.2f}%"
                for v in valid["뇌졸중 비율(%)"]
            ],
            textposition="outside",
            customdata=valid[
                ["사람_수", "경험자_수"]
            ].to_numpy(),
            hovertemplate=(
                "%{x}<br>"
                "뇌졸중 비율 %{y:.2f}%<br>"
                "전체 %{customdata[0]}명<br>"
                "경험자 %{customdata[1]}명"
                "<extra></extra>"
            ),
        )
    )

    style_chart(
        fig,
        "뇌졸중 비율 (%)",
    )

    if not valid.empty:

        maximum = float(
            valid["뇌졸중 비율(%)"].max()
        )

        fig.update_yaxes(
            range=[
                0,
                max(5, maximum * 1.3),
            ]
        )

    return fig


# =========================================================
# 데이터 불러오기
# =========================================================

try:
    df = load_data()

except Exception as exc:

    st.error(
        "데이터를 불러오지 못했습니다."
    )

    with st.expander("오류 상세"):
        st.code(str(exc))

    if st.button("다시 시도"):
        load_data.clear()
        st.rerun()

    st.stop()


# =========================================================
# 사이드바 필터
# =========================================================

with st.sidebar:

    st.divider()

    st.markdown("**분석 조건**")

    st.caption(
        "조건을 바꾸면 모든 그래프에 적용됩니다."
    )

    if st.button(
        "필터 초기화",
        use_container_width=True,
    ):

        for key in [
            "filter_gender",
            "filter_residence",
            "filter_age",
            "filter_bins",
            "filter_mean",
        ]:
            st.session_state.pop(
                key,
                None,
            )

        st.rerun()

    gender = st.selectbox(
        "성별",
        ["전체"]
        + sorted(
            df["gender"]
            .dropna()
            .unique()
            .tolist()
        ),
        key="filter_gender",
    )

    residence = st.selectbox(
        "거주 지역",
        ["전체"]
        + sorted(
            df["Residence_type"]
            .dropna()
            .unique()
            .tolist()
        ),
        key="filter_residence",
    )

    max_age = int(
        df["age"].max()
    )

    age_range = st.slider(
        "나이",
        0,
        max_age,
        (0, max_age),
        key="filter_age",
    )

    bins = st.slider(
        "히스토그램 구간 수",
        10,
        70,
        40,
        5,
        key="filter_bins",
    )

    show_mean = st.checkbox(
        "평균선 표시",
        value=True,
        key="filter_mean",
    )


# =========================================================
# 필터 적용
# =========================================================

filtered = df.copy()

if gender != "전체":
    filtered = filtered[
        filtered["gender"] == gender
    ]

if residence != "전체":
    filtered = filtered[
        filtered["Residence_type"] == residence
    ]

filtered = filtered[
    filtered["age"].between(
        *age_range
    )
].copy()


# =========================================================
# 상단 헤더
# =========================================================

st.markdown(
    """
    <div class="masthead">
        <span class="brand">NEURO / LAB</span>
        <span class="issue">DATA STUDY · 02</span>
    </div>

    <div class="kicker">
        02 / EXPLORE THE EVIDENCE
    </div>

    <div class="headline">
        데이터에서<br>
        <em>패턴을 발견하다.</em>
    </div>

    <div class="lead">
        나이와 혈당은 어떻게 분포할까요?
        건강 상태별 차이는 무엇일까요?
        필터를 조정하며 데이터를 직접 탐색해 보세요.
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered.empty:

    st.warning(
        "현재 조건에 해당하는 데이터가 없습니다. "
        "왼쪽 필터를 초기화해 주세요."
    )

    st.stop()


# =========================================================
# 핵심 통계
# =========================================================

filtered["뇌졸중"] = filtered["stroke"].map({
    0: "경험 없음",
    1: "경험 있음",
})

total = len(filtered)

cases = int(
    filtered["stroke"].eq(1).sum()
)

missing_bmi = int(
    filtered["bmi"].isna().sum()
)

cards = st.columns(4)

cards[0].metric(
    "분석 대상",
    f"{total:,}명",
)

cards[1].metric(
    "뇌졸중 경험자",
    f"{cases:,}명",
)

cards[2].metric(
    "뇌졸중 경험 비율",
    f"{pct(cases, total):.2f}%",
)

cards[3].metric(
    "BMI 빈 값",
    f"{missing_bmi:,}개",
)

st.caption(
    f"전체 {len(df):,}명 중 "
    f"{total:,}명을 선택했습니다. "
    "아래 통계는 모두 필터 적용 후의 값입니다."
)

with st.sidebar:

    st.download_button(
        "선택 데이터 저장 ↗",
        data=filtered.drop(
            columns=["뇌졸중"]
        ).to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="neuro_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# 01. 분포 분석
# =========================================================

section(
    "01",
    "분포를 읽다",
    "나이와 평균 혈당의 분포를 비교합니다. "
    "점선은 평균값입니다.",
)

left, right = st.columns(
    2,
    gap="large",
)

charts = [
    (
        left,
        "age",
        "나이",
        BLUE,
    ),
    (
        right,
        "avg_glucose_level",
        "평균 혈당",
        GREEN,
    ),
]

for target, col, title, color in charts:

    with target:

        st.markdown(
            "#### " + title
        )

        fig = px.histogram(
            filtered,
            x=col,
            nbins=bins,
            color_discrete_sequence=[
                color
            ],
            labels={
                col: title
            },
        )

        fig.update_traces(
            marker_line_color="white",
            marker_line_width=0.5,
        )

        if show_mean:

            fig.add_vline(
                x=filtered[col].mean(),
                line_dash="dash",
                line_color=ORANGE,
                line_width=2,
            )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

        st.caption(
            f"평균 {filtered[col].mean():.2f} · "
            f"중앙값 {filtered[col].median():.2f}"
        )


# =========================================================
# 02. 뇌졸중 경험별 비교
# =========================================================

section(
    "02",
    "두 집단을 비교하다",
    "뇌졸중을 경험한 사람과 경험하지 않은 "
    "사람을 비교합니다.",
)

order = [
    "경험 없음",
    "경험 있음",
]

colors = {
    "경험 없음": BLUE,
    "경험 있음": ORANGE,
}

left, right = st.columns(
    2,
    gap="large",
)

comparison_charts = [
    (
        left,
        "age",
        "나이 비교",
    ),
    (
        right,
        "avg_glucose_level",
        "평균 혈당 비교",
    ),
]

for target, col, title in comparison_charts:

    with target:

        st.markdown(
            "#### " + title
        )

        fig = px.box(
            filtered,
            x="뇌졸중",
            y=col,
            color="뇌졸중",
            category_orders={
                "뇌졸중": order,
            },
            color_discrete_map=colors,
            labels={
                col: title,
            },
        )

        st.plotly_chart(
            style_chart(fig, title),
            use_container_width=True,
        )


# 집단별 평균표
comparison = (
    filtered.groupby("뇌졸중")
    .agg(
        평균_나이=("age", "mean"),
        평균_혈당=(
            "avg_glucose_level",
            "mean",
        ),
        고혈압_비율=(
            "hypertension",
            "mean",
        ),
        심장병_비율=(
            "heart_disease",
            "mean",
        ),
        사람_수=("id", "size"),
    )
    .reindex(order)
    .reset_index()
)

comparison["고혈압_비율"] *= 100
comparison["심장병_비율"] *= 100

comparison["사람_수"] = (
    comparison["사람_수"]
    .fillna(0)
    .astype(int)
)

st.markdown(
    "#### 수치로 확인하기"
)

st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True,
    column_config={
        "평균_나이": st.column_config.NumberColumn(
            "평균 나이",
            format="%.2f",
        ),
        "평균_혈당": st.column_config.NumberColumn(
            "평균 혈당",
            format="%.2f",
        ),
        "고혈압_비율": st.column_config.NumberColumn(
            "고혈압 비율",
            format="%.1f%%",
        ),
        "심장병_비율": st.column_config.NumberColumn(
            "심장병 비율",
            format="%.1f%%",
        ),
    },
)

st.download_button(
    "집단 비교 결과 저장 ↗",
    data=comparison.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="neuro_comparison.csv",
    mime="text/csv",
)


# =========================================================
# 03. 고혈압 / 심장병
# =========================================================

section(
    "03",
    "건강 상태와 뇌졸중",
    "고혈압과 심장병 유무별 뇌졸중 "
    "경험 비율을 비교합니다.",
)

hypertension = risk_table(
    filtered,
    "hypertension",
    "고혈압",
)

heart = risk_table(
    filtered,
    "heart_disease",
    "심장병",
)

left, right = st.columns(
    2,
    gap="large",
)

with left:

    st.markdown(
        "#### 고혈압 여부"
    )

    st.plotly_chart(
        risk_chart(hypertension),
        use_container_width=True,
    )

with right:

    st.markdown(
        "#### 심장병 여부"
    )

    st.plotly_chart(
        risk_chart(heart),
        use_container_width=True,
    )

with st.expander(
    "비율 계산에 사용한 인원 확인"
):

    a, b = st.columns(2)

    with a:

        st.dataframe(
            hypertension,
            hide_index=True,
            use_container_width=True,
        )

    with b:

        st.dataframe(
            heart,
            hide_index=True,
            use_container_width=True,
        )

st.markdown(
    """
    <div class="info-note">
        <b>해석할 때 주의</b> ·
        집단 간 비율 차이가 있다는 사실만으로
        인과관계를 단정할 수 없습니다.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 04. BMI 결측치
# =========================================================

section(
    "04",
    "비어 있는 값도 정보다",
    "BMI가 비어 있는 사람과 전체의 "
    "뇌졸중 경험 비율을 비교합니다.",
)

missing = filtered[
    filtered["bmi"].isna()
]

missing_cases = int(
    missing["stroke"].eq(1).sum()
)

bmi_table = pd.DataFrame([
    {
        "구분": "BMI가 비어 있는 사람",
        "사람 수": len(missing),
        "뇌졸중 경험자": missing_cases,
        "뇌졸중 비율(%)": pct(
            missing_cases,
            len(missing),
        ),
    },
    {
        "구분": "전체",
        "사람 수": total,
        "뇌졸중 경험자": cases,
        "뇌졸중 비율(%)": pct(
            cases,
            total,
        ),
    },
])

st.dataframe(
    bmi_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "뇌졸중 비율(%)": st.column_config.NumberColumn(
            format="%.2f%%",
        ),
    },
)

st.caption(
    "선택된 데이터에 BMI 빈 값이 없다면 "
    "해당 집단의 비율은 계산할 수 없습니다."
)


# =========================================================
# 05. 흡연 상태
# =========================================================

section(
    "05",
    "흡연 상태를 분류하다",
    "흡연 상태별 사람 수와 비율을 확인합니다.",
)

smoking = (
    filtered["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

smoking["비율(%)"] = (
    smoking["사람 수"]
    / total
    * 100
)

left, right = st.columns(
    [1.2, 1],
    gap="large",
)

with left:

    st.dataframe(
        smoking,
        hide_index=True,
        use_container_width=True,
        column_config={
            "비율(%)": st.column_config.NumberColumn(
                format="%.1f%%",
            ),
        },
    )

    st.download_button(
        "흡연 상태 집계 저장 ↗",
        data=smoking.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="neuro_smoking.csv",
        mime="text/csv",
    )

with right:

    st.markdown(
        """
        <div class="info-note">
            <b>Unknown</b>은
            '상태를 모름'을 뜻하는 문자 범주입니다.
            BMI의 빈 값(NaN)과 다릅니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 06. 실습 체크리스트
# =========================================================

section(
    "06",
    "관찰을 마무리하다",
    "교재에 결과를 기록했다면 체크하세요.",
)

tasks = [
    "나이와 혈당 분포 확인",
    "뇌졸중 경험별 나이와 혈당 비교",
    "집단 평균 기록",
    "고혈압별 비율 확인",
    "심장병별 비율 확인",
    "BMI 결측치 비교",
    "흡연 상태 확인",
]

done = 0

for idx, task in enumerate(tasks):

    checked = st.checkbox(
        task,
        key=f"neuro_check_{idx}",
    )

    if checked:
        done += 1

st.progress(
    done / len(tasks),
    text=(
        f"완료한 활동 · "
        f"{done} / {len(tasks)}"
    ),
)

if done == len(tasks):
    st.success(
        "모든 탐색 활동을 완료했습니다!"
    )

st.markdown(
    """
    <div class="footer">
        NEURO / LAB ·
        EXPLORE THE EVIDENCE ·
        EDUCATIONAL USE ONLY
    </div>
    """,
    unsafe_allow_html=True,
)
