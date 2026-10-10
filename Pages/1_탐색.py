
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# NEURO LAB — 데이터 탐색 스튜디오
# ============================================================

st.set_page_config(
    page_title="데이터 탐색 | NEURO LAB",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

CYAN = "#40E0D0"
BLUE = "#648FFF"
VIOLET = "#A78BFA"
CORAL = "#FF7C91"
TEXT = "#243553"

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 90% 0%, #233E702B, transparent 38%),
        #080F1F;
    color: #E8F0FF;
}
[data-testid="stSidebar"] {
    background: #0D172B;
    border-right: 1px solid #253653;
}
[data-testid="stHeader"] {background: transparent;}
.block-container {
    max-width: 1380px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}
h1,h2,h3 {color: #F1F7FF !important;}
.hero {
    background: linear-gradient(115deg, #122448, #27375C 60%, #155A69);
    border: 1px solid #345B7B;
    padding: 45px;
    border-radius: 25px;
    margin-bottom: 28px;
    box-shadow: 0 18px 45px #00000035;
}
.hero h1 {
    font-size: clamp(32px, 4vw, 48px);
    font-weight: 850;
    margin: 14px 0;
}
.hero p {
    color: #C9DAEA;
    font-size: 16px;
    line-height: 1.8;
}
.eyebrow {
    color: #62F2DF;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}
.pill {
    display: inline-block;
    border: 1px solid #65DAD755;
    background: #59DACE18;
    padding: 7px 13px;
    border-radius: 99px;
    margin: 12px 7px 0 0;
    font-size: 12px;
    color: #BCFFF4;
}
.section {
    margin: 45px 0 18px;
}
.section small {
    color: #5CEAD8;
    font-weight: 800;
    letter-spacing: 2px;
}
.section h2 {
    font-size: 27px;
    margin: 8px 0;
}
.section p {
    color: #A9BAD0;
    font-size: 14px;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #14243D, #101B30);
    border: 1px solid #2A405E;
    border-radius: 19px;
    padding: 22px;
    min-height: 135px;
}
[data-testid="stMetricValue"] {
    color: #5CEAD8;
    font-weight: 850;
}
[data-testid="stPlotlyChart"] {
    border: 1px solid #2A405E;
    border-radius: 18px;
    overflow: hidden;
}
[data-testid="stDataFrame"] {
    border: 1px solid #2A405E;
    border-radius: 14px;
    overflow: hidden;
}
.stButton button, .stDownloadButton button {
    border-radius: 12px;
}
@media(max-width:700px) {
    .block-container {padding-top: 1rem;}
    .hero {padding: 28px 24px;}
}
</style>
""", unsafe_allow_html=True)


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
            <small>ANALYSIS {number} / NEURO LAB</small>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, y_title="사람 수"):
    fig.update_layout(
        template="plotly_white",
        height=375,
        margin=dict(l=28, r=28, t=35, b=32),
        font=dict(family="Arial, sans-serif", size=13, color=TEXT),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        title=y_title,
        showgrid=True,
        gridcolor="#E9EDF5",
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
            뇌졸중_수=lambda x: int(x.eq(1).sum()),
        )
        .reindex([0, 1], fill_value=0)
        .reset_index()
    )
    table["뇌졸중 비율(%)"] = (
        table["뇌졸중_수"]
        .div(table["사람_수"].replace(0, float("nan")))
        * 100
    )
    table["구분"] = table[column].map({
        0: f"{label} 없음",
        1: f"{label} 있음",
    })
    return table[["구분", "사람_수", "뇌졸중_수", "뇌졸중 비율(%)"]]


def risk_chart(table):
    valid = table[table["사람_수"] > 0]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=valid["구분"],
        y=valid["뇌졸중 비율(%)"],
        marker_color=[BLUE, CORAL][:len(valid)],
        text=[f"{v:.2f}%" for v in valid["뇌졸중 비율(%)"]],
        textposition="outside",
        customdata=valid[["사람_수", "뇌졸중_수"]].to_numpy(),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "뇌졸중 비율: %{y:.2f}%<br>"
            "사람 수: %{customdata[0]:,}명<br>"
            "뇌졸중 경험자: %{customdata[1]:,}명"
            "<extra></extra>"
        ),
    ))
    style_chart(fig, "뇌졸중 비율 (%)")
    maximum = valid["뇌졸중 비율(%)"].max()
    fig.update_yaxes(range=[0, max(5, float(maximum) * 1.3)])
    return fig


try:
    with st.spinner("분석 데이터를 준비하는 중..."):
        df = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다.")
    with st.expander("오류 내용"):
        st.code(str(exc))
    if st.button("🔄 다시 시도"):
        load_data.clear()
        st.rerun()
    st.stop()

# ============================================================
# 사이드바
# ============================================================

with st.sidebar:
    st.markdown("## 🧠 NEURO LAB")
    st.caption("EXPLORATION CONTROL CENTER")
    st.divider()

    st.page_link("main.py", label="데이터 소개", icon="🏠")
    st.page_link("pages/1_탐색.py", label="데이터 탐색", icon="🔎")
    st.divider()

    st.markdown("### 🎛️ ANALYSIS FILTERS")
    st.caption("모든 그래프와 표에 동일하게 적용됩니다.")

    if st.button("↩️ 필터 초기화", use_container_width=True):
        for key in ["f_gender", "f_residence", "f_age", "f_bins", "f_mean"]:
            st.session_state.pop(key, None)
        st.rerun()

    gender = st.selectbox(
        "성별",
        ["전체"] + sorted(df["gender"].dropna().unique().tolist()),
        key="f_gender",
    )

    residence = st.selectbox(
        "거주 지역",
        ["전체"] + sorted(df["Residence_type"].dropna().unique().tolist()),
        key="f_residence",
    )

    age_max = int(df["age"].max())
    age_range = st.slider(
        "나이 범위",
        0, age_max, (0, age_max),
        key="f_age",
    )

    bins = st.slider(
        "히스토그램 구간 수",
        10, 70, 40, 5,
        key="f_bins",
    )

    show_mean = st.toggle(
        "히스토그램 평균선 표시",
        value=True,
        key="f_mean",
    )

filtered = df.copy()

if gender != "전체":
    filtered = filtered[filtered["gender"] == gender]

if residence != "전체":
    filtered = filtered[filtered["Residence_type"] == residence]

filtered = filtered[
    filtered["age"].between(age_range[0], age_range[1])
].copy()

st.markdown("""
<div class="hero">
    <div class="eyebrow">● NEURO LAB / EXPLORATION ENGINE</div>
    <h1>🔎 데이터 탐색 스튜디오</h1>
    <p>
        숫자를 넘어 패턴을 발견하는 공간.<br>
        나이, 혈당, 건강 상태, 결측치와 흡연 상태를
        인터랙티브 그래프로 탐색하세요.
    </p>
    <span class="pill">LIVE FILTERS</span>
    <span class="pill">PLOTLY ANALYTICS</span>
    <span class="pill">DATA EXPLORATION</span>
</div>
""", unsafe_allow_html=True)

if filtered.empty:
    st.warning("현재 필터에 해당하는 데이터가 없습니다.")
    st.info("사이드바에서 조건을 변경하거나 필터를 초기화하세요.")
    st.stop()

filtered["뇌졸중"] = filtered["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음",
})

total = len(filtered)
stroke_count = int(filtered["stroke"].eq(1).sum())
stroke_rate = percentage(stroke_count, total)
missing_bmi = int(filtered["bmi"].isna().sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 분석 대상", f"{total:,}명")
c2.metric("🧠 뇌졸중 경험자", f"{stroke_count:,}명")
c3.metric("📈 발생 비율", f"{stroke_rate:.2f}%")
c4.metric("🧩 BMI 빈 값", f"{missing_bmi:,}개")

st.caption(
    f"원본 {len(df):,}명 중 {total:,}명을 분석하고 있습니다. "
    "모든 결과는 현재 필터 기준입니다."
)

with st.sidebar:
    st.divider()
    st.download_button(
        "⬇️ 선택 데이터 다운로드",
        data=filtered.drop(columns=["뇌졸중"]).to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="neuro_filtered_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# 01 분포
# ============================================================

section(
    "01",
    "나이와 평균 혈당의 분포",
    "각 구간에 몇 명이 있는지 확인하세요. 평균선은 사이드바에서 켜고 끌 수 있습니다.",
)

left, right = st.columns(2, gap="large")

for target, column, title, color in [
    (left, "age", "나이 분포", CYAN),
    (right, "avg_glucose_level", "평균 혈당 분포", VIOLET),
]:
    with target:
        st.markdown(f"#### {title}")
        fig = px.histogram(
            filtered,
            x=column,
            nbins=bins,
            color_discrete_sequence=[color],
            labels={column: title, "count": "사람 수"},
        )
        fig.update_traces(
            marker_line_color="white",
            marker_line_width=.6,
            opacity=.88,
        )
        if show_mean and filtered[column].notna().any():
            fig.add_vline(
                x=filtered[column].mean(),
                line_dash="dash",
                line_color=CORAL,
                line_width=2.5,
            )
        style_chart(fig)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"평균 {filtered[column].mean():.2f} · "
            f"중앙값 {filtered[column].median():.2f}"
        )

# ============================================================
# 02 상자그림
# ============================================================

section(
    "02",
    "뇌졸중 경험별 집단 비교",
    "상자그림으로 중앙값과 분포를 비교하고, 아래 표에서 평균값을 확인합니다.",
)

group_order = ["뇌졸중 없음", "뇌졸중 있음"]
group_colors = {
    "뇌졸중 없음": BLUE,
    "뇌졸중 있음": CORAL,
}

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
            category_orders={"뇌졸중": group_order},
            color_discrete_map=group_colors,
            labels={column: title},
        )
        style_chart(fig, title)
        st.plotly_chart(fig, width="stretch")

group_table = (
    filtered.groupby("뇌졸중")
    .agg(
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
        고혈압_비율=("hypertension", "mean"),
        심장병_비율=("heart_disease", "mean"),
        사람_수=("id", "size"),
    )
    .reindex(group_order)
    .reset_index()
)

group_table["고혈압_비율"] *= 100
group_table["심장병_비율"] *= 100
group_table["사람_수"] = group_table["사람_수"].fillna(0).astype(int)

st.markdown("#### 📋 집단별 상세 비교표")
st.dataframe(
    group_table,
    hide_index=True,
    width="stretch",
    column_config={
        "평균_나이": st.column_config.NumberColumn("평균 나이", format="%.2f"),
        "평균_혈당": st.column_config.NumberColumn("평균 혈당", format="%.2f"),
        "고혈압_비율": st.column_config.NumberColumn("고혈압 비율", format="%.1f%%"),
        "심장병_비율": st.column_config.NumberColumn("심장병 비율", format="%.1f%%"),
        "사람_수": st.column_config.NumberColumn("사람 수", format="%d명"),
    },
)

st.download_button(
    "⬇️ 집단 비교표 다운로드",
    group_table.to_csv(index=False).encode("utf-8-sig"),
    file_name="neuro_group_comparison.csv",
    mime="text/csv",
)

# ============================================================
# 03 고혈압과 심장병
# ============================================================

section(
    "03",
    "건강 상태별 뇌졸중 비율",
    "고혈압과 심장병이 있는 집단과 없는 집단의 뇌졸중 발생 비율을 비교합니다.",
)

hypertension_table = risk_table(filtered, "hypertension", "고혈압")
heart_table = risk_table(filtered, "heart_disease", "심장병")

left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### ❤️ 고혈압 여부")
    st.plotly_chart(
        risk_chart(hypertension_table),
        width="stretch",
    )

with right:
    st.markdown("#### 💜 심장병 여부")
    st.plotly_chart(
        risk_chart(heart_table),
        width="stretch",
    )

with st.expander("🔬 비율 계산에 사용한 사람 수 확인"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**고혈압 집단**")
        st.dataframe(hypertension_table, hide_index=True, width="stretch")
    with c2:
        st.markdown("**심장병 집단**")
        st.dataframe(heart_table, hide_index=True, width="stretch")

st.info(
    "📌 집단별 발생 비율이 다르더라도, "
    "그 차이만으로 고혈압이나 심장병이 뇌졸중의 원인이라고 단정할 수 없습니다."
)

# ============================================================
# 04 BMI 결측치
# ============================================================

section(
    "04",
    "BMI 결측치 분석",
    "BMI가 비어 있는 사람들의 뇌졸중 비율을 전체와 비교합니다.",
)

missing = filtered[filtered["bmi"].isna()]
missing_cases = int(missing["stroke"].eq(1).sum())

bmi_table = pd.DataFrame([
    {
        "비교 집단": "BMI가 비어 있는 사람",
        "사람 수": len(missing),
        "뇌졸중 경험자": missing_cases,
        "뇌졸중 비율(%)": percentage(missing_cases, len(missing)),
    },
    {
        "비교 집단": "전체",
        "사람 수": total,
        "뇌졸중 경험자": stroke_count,
        "뇌졸중 비율(%)": stroke_rate,
    },
])

st.dataframe(
    bmi_table,
    hide_index=True,
    width="stretch",
    column_config={
        "뇌졸중 비율(%)": st.column_config.NumberColumn(format="%.2f%%"),
    },
)

if len(missing) == 0:
    st.info("현재 필터에는 BMI가 비어 있는 사람이 없습니다.")
else:
    st.caption(
        f"전체 선택 인원 중 BMI가 비어 있는 사람의 비중: "
        f"{len(missing) / total * 100:.2f}%"
    )

st.warning(
    "⚠️ 결측치는 함부로 삭제하거나 임의의 숫자로 채우지 않습니다. "
    "모델을 만들 때 처리 방법을 따로 결정해야 합니다."
)

# ============================================================
# 05 흡연 상태
# ============================================================

section(
    "05",
    "흡연 상태별 사람 수",
    "흡연 상태의 범주별 인원과 전체에서 차지하는 비율을 확인합니다.",
)

smoking = (
    filtered["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
smoking["비율(%)"] = smoking["사람 수"] / total * 100

left, right = st.columns([1.2, 1], gap="large")

with left:
    st.dataframe(
        smoking,
        hide_index=True,
        width="stretch",
        column_config={
            "비율(%)": st.column_config.ProgressColumn(
                "비율(%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )
    st.download_button(
        "⬇️ 흡연 상태 집계 다운로드",
        smoking.to_csv(index=False).encode("utf-8-sig"),
        file_name="neuro_smoking_summary.csv",
        mime="text/csv",
    )

with right:
    st.info(
        "💡 `Unknown`은 흡연 상태를 모른다는 뜻으로 기록된 "
        "문자열 범주입니다. BMI의 실제 빈 값(NaN)과 구분해야 합니다."
    )

# ============================================================
# 06 실습 체크리스트
# ============================================================

section(
    "06",
    "실습 완료 체크리스트",
    "그래프를 확인하고 교재에 결과를 기록했다면 항목을 체크하세요.",
)

tasks = [
    "나이와 평균 혈당의 분포 확인",
    "뇌졸중 경험별 나이와 혈당 비교",
    "두 집단의 평균값 교재에 기록",
    "고혈압 여부별 뇌졸중 비율 확인",
    "심장병 여부별 뇌졸중 비율 확인",
    "BMI 결측치 집단과 전체 비율 비교",
    "흡연 상태별 사람 수 확인",
]

if "exploration_tasks" not in st.session_state:
    st.session_state.exploration_tasks = [False] * len(tasks)

for i, task in enumerate(tasks):
    st.session_state.exploration_tasks[i] = st.checkbox(
        task,
        value=st.session_state.exploration_tasks[i],
        key=f"task_{i}",
    )

done = sum(st.session_state.exploration_tasks)
st.progress(
    done / len(tasks),
    text=f"실습 완료율 · {done}/{len(tasks)}",
)

if done == len(tasks):
    st.success("🏆 EXPLORATION COMPLETE! 모든 탐색 실습을 완료했습니다.")

st.divider()
st.caption(
    "NEURO LAB · EDUCATIONAL DATA SCIENCE STUDIO | "
    "이 분석은 교육용이며 실제 의료 진단에 사용해서는 안 됩니다."
)
st.page_link(
    "main.py",
    label="← 데이터 소개로 돌아가기",
    use_container_width=True,
)
