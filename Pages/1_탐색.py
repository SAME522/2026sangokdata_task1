
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="데이터 탐색 | NEURO LAB",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

BLUE = "#3559D5"
ORANGE = "#D77C51"
GREEN = "#478779"
INK = "#242923"
PAPER = "#F8F8F5"

CSS = """
<style>
.stApp {
    background: #F8F8F5;
    color: #242923;
}
[data-testid="stHeader"] {background: #F8F8F5;}
[data-testid="stSidebar"] {
    background: #F0F1ED;
    border-right: 1px solid #DFE2DA;
}
[data-testid="stSidebar"] * {color: #303630;}
.block-container {
    max-width: 1280px;
    padding-top: 2.3rem;
    padding-bottom: 5rem;
}
h1,h2,h3,p {color: #242923;}
h1,h2,h3 {letter-spacing: -.045em;}
.topline {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #D7DBD3;
    padding-bottom: 18px;
    margin-bottom: 52px;
}
.brand {font-size: 15px; font-weight: 800;}
.edition {
    color: #777D75;
    font-size: 11px;
    letter-spacing: 1.4px;
}
.kicker {
    color: #3559D5;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}
.display {
    font-size: clamp(40px, 5.5vw, 76px);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -.075em;
    margin: 19px 0;
}
.display span {color: #3559D5;}
.lead {
    color: #6D756C;
    font-size: 16px;
    line-height: 1.9;
    max-width: 680px;
    margin-bottom: 42px;
}
.section {
    display: flex;
    gap: 18px;
    align-items: baseline;
    border-top: 1px solid #D7DBD3;
    padding-top: 21px;
    margin-top: 76px;
    margin-bottom: 12px;
}
.section-no {
    font-size: 12px;
    font-weight: 800;
    color: #3559D5;
}
.section-name {
    font-size: clamp(24px, 3vw, 34px);
    font-weight: 800;
    letter-spacing: -.05em;
}
.section-desc {
    color: #6D756C;
    font-size: 14px;
    margin-bottom: 25px;
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E1E4DC;
    border-radius: 5px;
    padding: 22px;
    min-height: 135px;
}
[data-testid="stMetricValue"] {
    color: #242923;
    font-weight: 800;
    letter-spacing: -.05em;
}
[data-testid="stPlotlyChart"] {
    border: 1px solid #E0E3DB;
    border-radius: 5px;
    overflow: hidden;
}
[data-testid="stDataFrame"] {
    border: 1px solid #E0E3DB;
    border-radius: 5px;
    overflow: hidden;
}
.stButton button,
.stDownloadButton button {
    border-radius: 5px;
    border-color: #C8CFC5;
    font-weight: 700;
}
.stButton button:hover,
.stDownloadButton button:hover {
    color: #3559D5;
    border-color: #3559D5;
}
[data-testid="stProgressBar"] > div > div {
    background: #3559D5;
}
.note {
    background: #EDF0E9;
    border-left: 3px solid #3559D5;
    padding: 17px 20px;
    color: #4E594F;
    font-size: 13px;
    line-height: 1.8;
    margin-top: 16px;
}
.footer {
    margin-top: 75px;
    border-top: 1px solid #D7DBD3;
    padding-top: 20px;
    color: #8B9189;
    font-size: 11px;
    letter-spacing: 1px;
}
@media(max-width:700px) {
    .block-container {padding-top: 1rem;}
    .topline {margin-bottom: 32px;}
    .display {font-size: 40px;}
    .section {margin-top: 48px;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "id", "gender", "Residence_type", "age",
        "avg_glucose_level", "stroke", "hypertension",
        "heart_disease", "bmi", "smoking_status",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"필수 열 누락: {', '.join(missing)}")
    return data


def section(number, title, description):
    st.markdown(
        f"""
        <div class="section">
            <span class="section-no">{number}</span>
            <span class="section-name">{title}</span>
        </div>
        <div class="section-desc">{description}</div>
        """,
        unsafe_allow_html=True,
    )


def chart_style(fig, y_title="사람 수"):
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=28, r=25, t=30, b=32),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#4B534B",
        ),
        showlegend=False,
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
        title=y_title,
        gridcolor="#ECEEE9",
        zeroline=False,
    )
    return fig


def percentage(cases, people):
    return cases / people * 100 if people else float("nan")


def risk_table(data, column, label):
    table = (
        data.groupby(column)["stroke"]
        .agg(
            사람_수="size",
            경험자_수=lambda s: int(s.eq(1).sum()),
        )
        .reindex([0, 1], fill_value=0)
        .reset_index()
    )
    table["뇌졸중 비율(%)"] = (
        table["경험자_수"]
        .div(table["사람_수"].replace(0, float("nan")))
        * 100
    )
    table["구분"] = table[column].map({
        0: f"{label} 없음",
        1: f"{label} 있음",
    })
    return table[["구분", "사람_수", "경험자_수", "뇌졸중 비율(%)"]]


def risk_chart(table):
    valid = table[table["사람_수"] > 0]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=valid["구분"],
        y=valid["뇌졸중 비율(%)"],
        marker_color=[
            BLUE if "없음" in x else ORANGE
            for x in valid["구분"]
        ],
        text=[f"{v:.2f}%" for v in valid["뇌졸중 비율(%)"]],
        textposition="outside",
        customdata=valid[["사람_수", "경험자_수"]].to_numpy(),
        hovertemplate=(
            "%{x}<br>"
            "뇌졸중 비율: %{y:.2f}%<br>"
            "전체 인원: %{customdata[0]}명<br>"
            "경험자: %{customdata[1]}명"
            "<extra></extra>"
        ),
    ))
    chart_style(fig, "뇌졸중 비율 (%)")
    if not valid.empty:
        maximum = float(valid["뇌졸중 비율(%)"].max())
        fig.update_yaxes(range=[0, max(5, maximum * 1.3)])
    return fig


try:
    df = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다.")
    with st.expander("오류 상세"):
        st.code(str(exc))
    if st.button("다시 시도"):
        load_data.clear()
        st.rerun()
    st.stop()


# =========================================================
# 필터
# =========================================================

with st.sidebar:
    st.markdown("### NEURO / LAB")
    st.caption("02 / 데이터 탐색")
    st.divider()
    st.markdown("#### 분석 조건")
    st.caption("조건을 바꾸면 모든 그래프가 업데이트됩니다.")

    if st.button("필터 초기화", use_container_width=True):
        for key in [
            "explore_gender", "explore_residence",
            "explore_age", "explore_bins",
            "explore_mean",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    gender = st.selectbox(
        "성별",
        ["전체"] + sorted(df["gender"].dropna().unique().tolist()),
        key="explore_gender",
    )

    residence = st.selectbox(
        "거주 지역",
        ["전체"] + sorted(
            df["Residence_type"].dropna().unique().tolist()
        ),
        key="explore_residence",
    )

    max_age = int(df["age"].max())

    age_range = st.slider(
        "나이",
        min_value=0,
        max_value=max_age,
        value=(0, max_age),
        key="explore_age",
    )

    bins = st.slider(
        "분포 그래프 구간 수",
        10, 70, 40, 5,
        key="explore_bins",
    )

    show_mean = st.checkbox(
        "평균선 표시",
        value=True,
        key="explore_mean",
    )

filtered = df.copy()

if gender != "전체":
    filtered = filtered[filtered["gender"] == gender]

if residence != "전체":
    filtered = filtered[filtered["Residence_type"] == residence]

filtered = filtered[
    filtered["age"].between(age_range[0], age_range[1])
].copy()


# =========================================================
# 헤더
# =========================================================

st.markdown("""
<div class="topline">
    <span class="brand">NEURO / LAB</span>
    <span class="edition">DATA STUDY — VOL. 02</span>
</div>
<div class="kicker">02 / EXPLORE THE EVIDENCE</div>
<div class="display">
    데이터에서<br>
    <span>패턴을 발견하다.</span>
</div>
<div class="lead">
    사람들의 나이와 혈당은 어떻게 분포할까요?
    뇌졸중 경험에 따라 차이가 있을까요?
    그래프와 숫자를 함께 읽으며 답을 찾아봅니다.
</div>
""", unsafe_allow_html=True)

if filtered.empty:
    st.warning("현재 필터에 해당하는 데이터가 없습니다.")
    st.info("사이드바에서 필터를 초기화해 주세요.")
    st.stop()

filtered["뇌졸중"] = filtered["stroke"].map({
    0: "경험 없음",
    1: "경험 있음",
})

total = len(filtered)
cases = int(filtered["stroke"].eq(1).sum())
rate = percentage(cases, total)
bmi_missing = int(filtered["bmi"].isna().sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("분석 대상", f"{total:,}명")
c2.metric("뇌졸중 경험자", f"{cases:,}명")
c3.metric("뇌졸중 비율", f"{rate:.2f}%")
c4.metric("BMI 빈 값", f"{bmi_missing:,}개")

st.caption(
    f"원본 {len(df):,}명 중 {total:,}명을 선택했습니다. "
    "아래 모든 분석은 선택한 사람을 기준으로 합니다."
)

with st.sidebar:
    st.divider()
    st.download_button(
        "선택 데이터 저장 ↗",
        data=filtered.drop(columns=["뇌졸중"]).to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="neuro_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# 01 분포
# =========================================================

section(
    "01",
    "분포를 읽다",
    "나이와 평균 혈당의 분포를 확인합니다. 점선은 평균값입니다.",
)

left, right = st.columns(2, gap="large")

for target, column, title, color in [
    (left, "age", "나이", BLUE),
    (right, "avg_glucose_level", "평균 혈당", GREEN),
]:
    with target:
        st.markdown(f"#### {title}")
        fig = px.histogram(
            filtered,
            x=column,
            nbins=bins,
            color_discrete_sequence=[color],
            labels={column: title},
        )
        fig.update_traces(
            marker_line_color="#FFFFFF",
            marker_line_width=.5,
        )
        if show_mean:
            fig.add_vline(
                x=filtered[column].mean(),
                line_dash="dash",
                line_color=ORANGE,
                line_width=2,
            )
        st.plotly_chart(
            chart_style(fig),
            use_container_width=True,
        )
        st.caption(
            f"평균 {filtered[column].mean():.2f}  ·  "
            f"중앙값 {filtered[column].median():.2f}"
        )


# =========================================================
# 02 집단 비교
# =========================================================

section(
    "02",
    "두 집단을 비교하다",
    "뇌졸중을 경험한 집단과 경험하지 않은 집단을 비교합니다.",
)

order = ["경험 없음", "경험 있음"]
colors = {"경험 없음": BLUE, "경험 있음": ORANGE}

left, right = st.columns(2, gap="large")

for target, column, title in [
    (left, "age", "나이 비교"),
    (right, "avg_glucose_level", "평균 혈당 비교"),
]:
    with target:
        st.markdown(f"#### {title}")
        fig = px.box(
            filtered,
            x="뇌졸중",
            y=column,
            color="뇌졸중",
            category_orders={"뇌졸중": order},
            color_discrete_map=colors,
            labels={column: title},
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
    .reindex(order)
    .reset_index()
)

comparison["고혈압_비율"] *= 100
comparison["심장병_비율"] *= 100
comparison["사람_수"] = comparison["사람_수"].fillna(0).astype(int)

st.markdown("#### 숫자로 다시 확인하기")
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

st.download_button(
    "집단 비교 결과 저장 ↗",
    data=comparison.to_csv(index=False).encode("utf-8-sig"),
    file_name="neuro_comparison.csv",
    mime="text/csv",
)


# =========================================================
# 03 고혈압 / 심장병
# =========================================================

section(
    "03",
    "건강 상태와 뇌졸중",
    "고혈압과 심장병 여부에 따른 뇌졸중 경험 비율을 비교합니다.",
)

hypertension = risk_table(filtered, "hypertension", "고혈압")
heart = risk_table(filtered, "heart_disease", "심장병")

left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### 고혈압 여부")
    st.plotly_chart(
        risk_chart(hypertension),
        use_container_width=True,
    )

with right:
    st.markdown("#### 심장병 여부")
    st.plotly_chart(
        risk_chart(heart),
        use_container_width=True,
    )

with st.expander("비율을 계산한 인원 확인"):
    c1, c2 = st.columns(2)
    with c1:
        st.write("고혈압")
        st.dataframe(
            hypertension,
            hide_index=True,
            use_container_width=True,
        )
    with c2:
        st.write("심장병")
        st.dataframe(
            heart,
            hide_index=True,
            use_container_width=True,
        )

st.markdown("""
<div class="note">
    <b>해석할 때 주의하세요.</b><br>
    집단별 뇌졸중 경험 비율이 다르다고 해서
    그 질환이 직접적인 원인이라는 뜻은 아닙니다.
    나이 등 다른 요인의 영향을 함께 고려해야 합니다.
</div>
""", unsafe_allow_html=True)


# =========================================================
# 04 BMI 결측치
# =========================================================

section(
    "04",
    "비어 있는 값도 정보다",
    "BMI가 비어 있는 사람들과 전체의 뇌졸중 비율을 비교합니다.",
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
    st.caption("현재 선택한 데이터에는 BMI 결측치가 없습니다.")
else:
    st.caption(
        f"현재 분석 대상 중 BMI 결측치 비중: "
        f"{len(missing) / total * 100:.2f}%"
    )


# =========================================================
# 05 흡연 상태
# =========================================================

section(
    "05",
    "흡연 상태를 분류하다",
    "기록된 흡연 상태의 종류와 인원을 확인합니다.",
)

smoking = (
    filtered["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
smoking["비율(%)"] = smoking["사람 수"] / total * 100

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.dataframe(
        smoking,
        hide_index=True,
        use_container_width=True,
        column_config={
            "비율(%)": st.column_config.NumberColumn(
                "비율 (%)",
                format="%.1f%%",
            ),
        },
    )
    st.download_button(
        "흡연 상태 집계 저장 ↗",
        data=smoking.to_csv(index=False).encode("utf-8-sig"),
        file_name="neuro_smoking.csv",
        mime="text/csv",
    )

with right:
    st.markdown("""
    <div class="note">
        <b>Unknown은 무엇일까요?</b><br>
        흡연 상태를 알 수 없다는 뜻으로 기록된
        하나의 문자열 범주입니다.<br>
        BMI 열의 실제 빈 값(NaN)과는 다릅니다.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 06 실습 체크리스트
# =========================================================

section(
    "06",
    "관찰을 마무리하다",
    "교재에 결과를 기록했다면 하나씩 체크하세요.",
)

tasks = [
    "나이와 혈당의 분포 확인",
    "뇌졸중 경험별 나이 비교",
    "뇌졸중 경험별 평균 혈당 비교",
    "고혈압 여부별 비율 확인",
    "심장병 여부별 비율 확인",
    "BMI 결측치 집단 분석",
    "흡연 상태 집계 확인",
]

done = 0

for i, task in enumerate(tasks):
    if st.checkbox(task, key=f"study_task_{i}"):
        done += 1

st.progress(
    done / len(tasks),
    text=f"완료한 활동  {done} / {len(tasks)}",
)

if done == len(tasks):
    st.success("모든 탐색 활동을 완료했습니다.")

st.markdown("""
<div class="footer">
    NEURO / LAB &nbsp;&nbsp;·&nbsp;&nbsp;
    EXPLORE THE EVIDENCE &nbsp;&nbsp;·&nbsp;&nbsp;
    EDUCATIONAL USE ONLY
</div>
""", unsafe_allow_html=True)
