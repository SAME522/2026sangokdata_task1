
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

BLUE = "#5B8DEF"
TEAL = "#12B8AE"
PURPLE = "#9270E9"
CORAL = "#F08077"

st.markdown("""
<style>
.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
.hero {
    padding: 43px;
    border-radius: 24px;
    background: linear-gradient(120deg, #111a37, #253e69, #126e79);
    color: white;
    margin-bottom: 28px;
}
.hero h1 {
    color: white;
    font-size: 42px;
    font-weight: 850;
}
.hero p {
    color: #d9eeee;
    font-size: 16px;
    line-height: 1.8;
}
.tag {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 30px;
    background: #ffffff25;
    margin: 10px 8px 0 0;
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
.section-title {
    margin-top: 45px;
    font-size: 26px;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def chart_style(fig, y_title="사람 수"):
    fig.update_layout(
        template="plotly_white",
        height=370,
        margin=dict(l=20, r=20, t=35, b=30),
        font=dict(size=13),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        hoverlabel=dict(bgcolor="white"),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        title=y_title,
        gridcolor="#E9EDF5",
        zeroline=False,
    )

    return fig


def calculate_rate(cases, people):
    if people == 0:
        return None
    return cases / people * 100


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러올 수 없습니다.")
    st.code(str(e))
    if st.button("🔄 다시 불러오기"):
        load_data.clear()
        st.rerun()
    st.stop()


# =========================
# 사이드바
# =========================

with st.sidebar:
    st.title("🧠 실습실 메뉴")

    st.page_link("main.py", label="🏠 데이터 소개")
    st.page_link("pages/1_탐색.py", label="🔎 데이터 탐색")

    st.divider()

    st.header("🎛️ 데이터 필터")
    st.caption("조건을 바꾸면 모든 그래프와 표가 변경됩니다.")

    if st.button("↩️ 필터 초기화", use_container_width=True):
        for key in [
            "filter_gender",
            "filter_residence",
            "filter_age",
            "filter_bins",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    gender_options = [
        "전체",
        *sorted(df["gender"].dropna().unique().tolist()),
    ]

    residence_options = [
        "전체",
        *sorted(df["Residence_type"].dropna().unique().tolist()),
    ]

    gender = st.selectbox(
        "성별",
        gender_options,
        key="filter_gender",
    )

    residence = st.selectbox(
        "거주 지역",
        residence_options,
        key="filter_residence",
    )

    max_age = int(df["age"].max())

    age_range = st.slider(
        "나이 범위",
        min_value=0,
        max_value=max_age,
        value=(0, max_age),
        key="filter_age",
    )

    bins = st.slider(
        "히스토그램 구간 수",
        min_value=10,
        max_value=60,
        value=25,
        step=5,
        key="filter_bins",
    )


# =========================
# 필터 적용
# =========================

filtered = df.copy()

if gender != "전체":
    filtered = filtered[filtered["gender"] == gender]

if residence != "전체":
    filtered = filtered[
        filtered["Residence_type"] == residence
    ]

filtered = filtered[
    filtered["age"].between(age_range[0], age_range[1])
].copy()


# =========================
# 상단 디자인
# =========================

st.markdown("""
<div class="hero">
    <div style="color:#82EDE0;font-weight:800;letter-spacing:2px;">
        STROKE DATA LAB / EXPLORATION
    </div>
    <h1>🔎 데이터 탐색 스튜디오</h1>
    <p>
        나이와 혈당의 분포부터 건강 상태별 뇌졸중 비율까지.<br>
        데이터를 직접 탐색하고 숨겨진 특징을 발견해 보세요.
    </p>
    <span class="tag">📊 인터랙티브 차트</span>
    <span class="tag">🎛️ 실시간 필터</span>
    <span class="tag">📥 데이터 다운로드</span>
</div>
""", unsafe_allow_html=True)

if filtered.empty:
    st.warning("현재 필터에 해당하는 데이터가 없습니다.")
    st.info("왼쪽에서 필터를 초기화하거나 범위를 넓혀 주세요.")
    st.stop()

filtered["stroke_group"] = filtered["stroke"].map({
    0: "겪지 않음",
    1: "겪음",
})

total = len(filtered)
stroke_count = int((filtered["stroke"] == 1).sum())
stroke_rate = calculate_rate(stroke_count, total)
missing_bmi = int(filtered["bmi"].isna().sum())

c1, c2, c3, c4 = st.columns(4)

c1.metric("👥 선택된 사람", f"{total:,}명")
c2.metric("🧠 뇌졸중 경험자", f"{stroke_count:,}명")
c3.metric("📈 뇌졸중 비율", f"{stroke_rate:.2f}%")
c4.metric("🧩 BMI 빈 값", f"{missing_bmi:,}개")

st.caption(
    f"전체 {len(df):,}명 중 현재 {total:,}명을 분석하고 있습니다."
)

with st.sidebar:
    st.divider()

    download_df = filtered.drop(columns=["stroke_group"])

    st.download_button(
        "⬇️ 필터링 데이터 다운로드",
        data=download_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="stroke_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================
# 1. 히스토그램
# =========================

st.markdown(
    '<div class="section-title">01 📊 나이와 평균 혈당의 분포</div>',
    unsafe_allow_html=True,
)

st.caption("각 구간에 몇 명이 포함되어 있는지 확인하세요. 점선은 평균값입니다.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("나이 분포")

    fig_age = px.histogram(
        filtered,
        x="age",
        nbins=bins,
        color_discrete_sequence=[TEAL],
        labels={"age": "나이 (세)"},
    )

    fig_age.add_vline(
        x=filtered["age"].mean(),
        line_dash="dash",
        line_color=CORAL,
        line_width=2,
    )

    fig_age = chart_style(fig_age)

    st.plotly_chart(
        fig_age,
        use_container_width=True,
    )

    st.caption(
        f"평균 나이: {filtered['age'].mean():.2f}세"
    )

with col2:
    st.subheader("평균 혈당 분포")

    fig_glucose = px.histogram(
        filtered,
        x="avg_glucose_level",
        nbins=bins,
        color_discrete_sequence=[PURPLE],
        labels={"avg_glucose_level": "평균 혈당"},
    )

    fig_glucose.add_vline(
        x=filtered["avg_glucose_level"].mean(),
        line_dash="dash",
        line_color=CORAL,
        line_width=2,
    )

    fig_glucose = chart_style(fig_glucose)

    st.plotly_chart(
        fig_glucose,
        use_container_width=True,
    )

    st.caption(
        f"평균 혈당: {filtered['avg_glucose_level'].mean():.2f}"
    )


# =========================
# 2. 상자그림
# =========================

st.markdown(
    '<div class="section-title">02 📦 뇌졸중 경험별 비교</div>',
    unsafe_allow_html=True,
)

st.caption(
    "상자그림의 가운데 선은 중앙값입니다. 평균값은 아래 표에서 확인하세요."
)

group_order = ["겪지 않음", "겪음"]

group_colors = {
    "겪지 않음": BLUE,
    "겪음": CORAL,
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("뇌졸중 경험별 나이")

    fig = px.box(
        filtered,
        x="stroke_group",
        y="age",
        color="stroke_group",
        category_orders={"stroke_group": group_order},
        color_discrete_map=group_colors,
        labels={
            "stroke_group": "뇌졸중 경험",
            "age": "나이 (세)",
        },
    )

    st.plotly_chart(
        chart_style(fig, "나이 (세)"),
        use_container_width=True,
    )

with col2:
    st.subheader("뇌졸중 경험별 평균 혈당")

    fig = px.box(
        filtered,
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        category_orders={"stroke_group": group_order},
        color_discrete_map=group_colors,
        labels={
            "stroke_group": "뇌졸중 경험",
            "avg_glucose_level": "평균 혈당",
        },
    )

    st.plotly_chart(
        chart_style(fig, "평균 혈당"),
        use_container_width=True,
    )

st.subheader("📋 두 집단의 평균값")

mean_table = (
    filtered.groupby("stroke_group")
    .agg(
        사람_수=("stroke", "size"),
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
    )
    .reindex(group_order)
    .reset_index()
)

mean_table.columns = [
    "뇌졸중 경험",
    "사람 수",
    "평균 나이",
    "평균 혈당",
]

mean_table["사람 수"] = (
    mean_table["사람 수"].fillna(0).astype(int)
)

st.dataframe(
    mean_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "평균 나이": st.column_config.NumberColumn(format="%.2f"),
        "평균 혈당": st.column_config.NumberColumn(format="%.2f"),
    },
)


# =========================
# 3. 고혈압 / 심장병 비율
# =========================

def make_risk_chart(data, column):
    result = (
        data.groupby(column)["stroke"]
        .agg(
            사람_수="size",
            뇌졸중_수=lambda x: int((x == 1).sum()),
        )
        .reindex([0, 1], fill_value=0)
        .reset_index()
    )

    result["뇌졸중 비율"] = (
        result["뇌졸중_수"]
        / result["사람_수"].replace(0, float("nan"))
        * 100
    )

    present = result[result["사람_수"] > 0].copy()

    present["구분"] = present[column].map({
        0: "없음 (0)",
        1: "있음 (1)",
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=present["구분"],
        y=present["뇌졸중 비율"],
        marker_color=[
            BLUE if value == 0 else CORAL
            for value in present[column]
        ],
        text=[
            f"{value:.2f}%"
            for value in present["뇌졸중 비율"]
        ],
        textposition="outside",
        customdata=present[
            ["사람_수", "뇌졸중_수"]
        ].to_numpy(),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "뇌졸중 비율: %{y:.2f}%<br>"
            "전체 사람 수: %{customdata[0]}명<br>"
            "뇌졸중 경험자: %{customdata[1]}명"
            "<extra></extra>"
        ),
    ))

    fig = chart_style(fig, "뇌졸중 비율 (%)")

    fig.update_yaxes(
        range=[
            0,
            max(
                5,
                float(present["뇌졸중 비율"].max()) * 1.3,
            ),
        ]
    )

    return fig, result


st.markdown(
    '<div class="section-title">03 ❤️ 건강 상태별 뇌졸중 비율</div>',
    unsafe_allow_html=True,
)

st.caption(
    "각 집단에서 뇌졸중을 겪은 사람의 비율을 계산합니다."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("고혈압 여부")

    fig_ht, table_ht = make_risk_chart(
        filtered,
        "hypertension",
    )

    st.plotly_chart(
        fig_ht,
        use_container_width=True,
    )

with col2:
    st.subheader("심장병 여부")

    fig_hd, table_hd = make_risk_chart(
        filtered,
        "heart_disease",
    )

    st.plotly_chart(
        fig_hd,
        use_container_width=True,
    )

with st.expander("📋 상세 계산표 보기"):
    c1, c2 = st.columns(2)

    with c1:
        st.write("고혈압")
        st.dataframe(
            table_ht,
            hide_index=True,
            use_container_width=True,
        )

    with c2:
        st.write("심장병")
        st.dataframe(
            table_hd,
            hide_index=True,
            use_container_width=True,
        )


# =========================
# 4. BMI 결측치 비교
# =========================

st.markdown(
    '<div class="section-title">04 🧩 BMI 빈 값 분석</div>',
    unsafe_allow_html=True,
)

st.caption(
    "BMI가 비어 있는 사람들의 뇌졸중 비율과 전체 비율을 비교합니다."
)

bmi_missing = filtered[
    filtered["bmi"].isna()
]

missing_total = len(bmi_missing)

missing_stroke = int(
    (bmi_missing["stroke"] == 1).sum()
)

bmi_table = pd.DataFrame([
    {
        "비교 집단": "BMI가 비어 있는 사람",
        "사람 수": missing_total,
        "뇌졸중 경험자": missing_stroke,
        "뇌졸중 비율 (%)": calculate_rate(
            missing_stroke,
            missing_total,
        ),
    },
    {
        "비교 집단": "전체 사람",
        "사람 수": total,
        "뇌졸중 경험자": stroke_count,
        "뇌졸중 비율 (%)": stroke_rate,
    },
])

st.dataframe(
    bmi_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "뇌졸중 비율 (%)": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
    },
)

if missing_total == 0:
    st.info("현재 선택한 데이터에는 BMI 빈 값이 없습니다.")

st.info(
    "BMI가 비어 있다는 사실만으로 뇌졸중 발생의 원인을 "
    "판단할 수는 없습니다."
)


# =========================
# 5. 흡연 상태
# =========================

st.markdown(
    '<div class="section-title">05 🚬 흡연 상태별 사람 수</div>',
    unsafe_allow_html=True,
)

st.caption(
    "흡연 상태의 각 범주별 사람 수를 집계합니다."
)

smoking_table = (
    filtered["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

smoking_table["비율 (%)"] = (
    smoking_table["사람 수"] / total * 100
)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.dataframe(
        smoking_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "비율 (%)": st.column_config.ProgressColumn(
                "비율 (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

    st.download_button(
        "⬇️ 흡연 상태 집계 다운로드",
        data=smoking_table.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="stroke_smoking.csv",
        mime="text/csv",
    )

with col2:
    st.info(
        "💡 Unknown은 흡연 상태가 알려지지 않았다는 뜻입니다. "
        "실제 빈 값(NaN)과는 다릅니다."
    )


# =========================
# 하단
# =========================

st.markdown("---")

st.success("🎉 데이터 탐색을 완료했습니다!")

st.caption(
    "이 앱은 교육용 데이터 분석 실습이며 의료 진단 도구가 아닙니다."
)

st.page_link(
    "main.py",
    label="🏠 데이터 소개로 돌아가기",
    use_container_width=True,
)
