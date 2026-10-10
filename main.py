
"""뇌졸중 예측 실습실 | 02 데이터 탐색 (Plotly 전용)."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type", "avg_glucose_level",
    "bmi", "smoking_status", "stroke",
]

COLORS = {
    "teal": "#16b8af",
    "blue": "#5b8cf7",
    "purple": "#9270e9",
    "coral": "#f08077",
    "ink": "#243553",
}
GROUPS = ["겪지 않음 (0)", "겪음 (1)"]
GROUP_COLOR = {GROUPS[0]: COLORS["blue"], GROUPS[1]: COLORS["coral"]}

st.set_page_config(
    page_title=f"탐색 | {APP_TITLE}",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1320px; padding-top: 1.8rem; padding-bottom: 4rem;}
    [data-testid="stHeader"] {background: transparent;}
    .hero {
        padding: 2.4rem 2.6rem; margin-bottom: 1.5rem; border-radius: 25px;
        background: radial-gradient(circle at 88% 20%, #3869aa 0, transparent 38%),
                    linear-gradient(116deg, #111a37 0%, #21365e 58%, #146b78 100%);
        box-shadow: 0 18px 36px rgba(19, 42, 79, .16); color: #fff;
    }
    .hero h1 {color: #fff; font-size: clamp(1.95rem, 3.4vw, 2.85rem);
              letter-spacing: -.055em; line-height: 1.22; margin: .5rem 0 .7rem;}
    .hero p {color: #dceafa; line-height: 1.8; margin: 0;}
    .eyebrow {color: #81ece1; font-size: .76rem; letter-spacing: .16em; font-weight: 800;}
    .pill {display: inline-block; border: 1px solid #ffffff55; background: #ffffff19;
           border-radius: 99px; font-size: .77rem; color: #fff;
           padding: .35rem .85rem; margin: 1.15rem .4rem 0 0;}
    .section-kicker {color: #12a79a; font-size: .76rem; font-weight: 800;
                     letter-spacing: .14em; margin: 2.2rem 0 .35rem;}
    .section-title {font-size: 1.5rem; font-weight: 800; letter-spacing: -.045em;
                    margin: 0 0 .4rem;}
    .section-desc {opacity: .75; margin: 0 0 1rem; font-size: .94rem;}
    [data-testid="stMetric"] {background: var(--secondary-background-color);
        border: 1px solid rgba(121, 153, 176, .22); border-radius: 17px;
        padding: 1rem 1.12rem; min-height: 125px;}
    [data-testid="stMetricValue"] {font-weight: 800; letter-spacing: -.05em;}
    div[data-testid="stPlotlyChart"] {background: #ffffff;
        border: 1px solid rgba(120, 152, 176, .16); border-radius: 18px;
        padding: .35rem; overflow: hidden;}
    div[data-testid="stDataFrame"] {border-radius: 13px; overflow: hidden;}
    div.stButton > button, div.stDownloadButton > button {border-radius: 11px;}
    @media(max-width: 700px) {
        .block-container {padding-top: .9rem;}
        .hero {padding: 1.65rem 1.4rem; border-radius: 19px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    missing = [col for col in COLUMNS if col not in data.columns]
    if missing:
        raise ValueError("필수 열 누락: " + ", ".join(missing))
    return data


def decorate(fig, y_title="사람 수"):
    """그래프 공통 디자인. Plotly 이외의 그래프 라이브러리는 사용하지 않음."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial, sans-serif", color=COLORS["ink"], size=13),
        margin=dict(l=24, r=24, t=30, b=24),
        height=355,
        showlegend=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_yaxes(title=y_title, gridcolor="rgba(143,158,180,.18)", zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    return fig


def histogram(data, column, label, color, bins):
    fig = px.histogram(
        data, x=column, nbins=bins,
        labels={column: label},
        color_discrete_sequence=[color],
    )
    fig.update_traces(marker_line_color="white", marker_line_width=.7, opacity=.88)
    if data[column].notna().any():
        mean = data[column].mean()
        fig.add_vline(x=mean, line_dash="dash", line_width=2, line_color=COLORS["coral"])
    fig.update_layout(bargap=.035)
    return decorate(fig)


def boxplot(data, column, label):
    fig = px.box(
        data, x="stroke_group", y=column,
        color="stroke_group", points=False,
        category_orders={"stroke_group": GROUPS},
        color_discrete_map=GROUP_COLOR,
        labels={"stroke_group": "뇌졸중 경험", column: label},
    )
    fig.update_traces(line_width=1.6)
    return decorate(fig, label)


def risk_bar(data, field):
    counts = (
        data.groupby(field)["stroke"]
        .agg(people="size", cases=lambda s: int(s.eq(1).sum()))
        .reindex([0, 1], fill_value=0)
        .reset_index()
    )
    counts["ratio"] = counts["cases"].div(counts["people"].where(counts["people"] > 0)) * 100
    present = counts[counts["people"] > 0].copy()
    present["group"] = present[field].map({0: "없음 (0)", 1: "있음 (1)"})
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=present["group"],
        y=present["ratio"],
        marker_color=[COLORS["blue"] if i == 0 else COLORS["coral"] for i in present[field]],
        customdata=present[["people", "cases"]].to_numpy(),
        text=[f"{r:.2f}%" for r in present["ratio"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>뇌졸중 비율: %{y:.2f}%"
            "<br>사람 수: %{customdata[0]:,}명"
            "<br>발생 수: %{customdata[1]:,}명<extra></extra>"
        ),
    ))
    fig = decorate(fig, "뇌졸중 비율 (%)")
    fig.update_xaxes(title="해당 질환 여부", categoryorder="array", categoryarray=["없음 (0)", "있음 (1)"])
    fig.update_yaxes(range=[0, max(6, float(present["ratio"].max()) * 1.27)] if len(present) else [0, 6])
    return fig, counts


def as_percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else None


def section(number, english, title, description):
    st.markdown(f'<div class="section-kicker">{number} / {english}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-desc">{description}</div>', unsafe_allow_html=True)


try:
    with st.spinner("UTF-8 CSV 데이터를 불러오는 중..."):
        source_df = load_data()
except Exception as exc:
    st.error("CSV 데이터를 읽지 못했습니다. 인터넷 연결과 주소를 확인하세요.")
    with st.expander("오류 상세 보기"):
        st.code(str(exc))
    if st.button("🔄 다시 불러오기"):
        load_data.clear()
        st.rerun()
    st.stop()

# 필터는 화면의 모든 수치·그래프·표에 동일하게 적용한다.
min_age = 0
max_age = int(source_df["age"].max())
genders = sorted(source_df["gender"].dropna().astype(str).unique().tolist())
residences = sorted(source_df["Residence_type"].dropna().astype(str).unique().tolist())

with st.sidebar:
    st.markdown("### 🧭 실습실 메뉴")
    st.page_link("main.py", label="01 · 데이터 소개", icon="🏠")
    st.page_link("pages/1_탐색.py", label="02 · 데이터 탐색", icon="🔎")
    st.divider()
    st.markdown("### 🎛️ 실시간 데이터 필터")
    st.caption("필터를 바꾸면 아래의 모든 결과가 함께 바뀝니다.")

    if st.button("↩️ 필터 초기화", use_container_width=True):
        for key in ("explore_gender", "explore_residence", "explore_age", "explore_bins"):
            st.session_state.pop(key, None)
        st.rerun()

    gender = st.selectbox("성별", ["전체", *genders], key="explore_gender")
    residence = st.selectbox("거주 지역", ["전체", *residences], key="explore_residence")
    age_range = st.slider(
        "나이 범위 (세)", min_value=min_age, max_value=max_age,
        value=(min_age, max_age), key="explore_age",
    )
    bins = st.slider("히스토그램 구간 개수", 10, 60, 25, 5, key="explore_bins")
    st.caption("분포 구간 수만 변경하면 히스토그램 모양만 달라집니다.")

filtered = source_df.copy()
if gender != "전체":
    filtered = filtered[filtered["gender"] == gender]
if residence != "전체":
    filtered = filtered[filtered["Residence_type"] == residence]
filtered = filtered[filtered["age"].between(*age_range)].copy()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">STROKE DATA LAB / CHAPTER 02</div>
      <h1>🔎 데이터 탐색 스튜디오</h1>
      <p>데이터를 눈으로 읽는 연습. 분포 → 집단 비교 → 질환별 비율 → 결측치·흡연 상태까지<br>
      필터를 조정하면서 결과가 어떻게 달라지는지 관찰해 보세요.</p>
      <span class="pill">📊 Plotly 인터랙티브 차트</span>
      <span class="pill">🎛️ 실시간 필터</span>
      <span class="pill">⬇️ CSV 내보내기</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("조건에 맞는 사람이 없습니다. 왼쪽 필터를 변경하거나 '필터 초기화'를 눌러 주세요.")
    st.stop()

filtered["stroke_group"] = filtered["stroke"].map({0: GROUPS[0], 1: GROUPS[1]})
n = len(filtered)
cases = int(filtered["stroke"].eq(1).sum())
missing_bmi = int(filtered["bmi"].isna().sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("👥 선택된 사람", f"{n:,}명", help=f"원본 {len(source_df):,}명 중 현재 필터로 선택된 인원")
m2.metric("🧠 뇌졸중 경험", f"{cases:,}명")
m3.metric("📈 뇌졸중 비율", f"{as_percent(cases, n):.2f}%")
m4.metric("🧩 BMI 빈 값", f"{missing_bmi:,}명")

st.caption(f"현재 원본 {len(source_df):,}명 중 **{n:,}명**을 분석하고 있습니다. 아래 표와 그래프는 모두 선택된 사람 기준입니다.")
with st.sidebar:
    st.divider()
    st.download_button(
        "⬇️ 현재 선택 데이터 CSV",
        data=filtered.drop(columns=["stroke_group"]).to_csv(index=False).encode("utf-8-sig"),
        file_name="stroke_탐색_선택데이터.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("한글이 깨지지 않도록 UTF-8 BOM으로 저장됩니다.")

section("01", "DISTRIBUTION", "나이와 평균 혈당의 분포", "막대 위에 마우스를 올리면 구간별 사람 수를 확인할 수 있습니다. 점선은 평균값입니다.")
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**01-A · 나이 분포**")
    st.plotly_chart(histogram(filtered, "age", "나이 (세)", COLORS["teal"], bins), use_container_width=True)
    st.caption(f"평균 나이: **{filtered['age'].mean():.1f}세**")
with c2:
    st.markdown("**01-B · 평균 혈당 분포**")
    st.plotly_chart(histogram(filtered, "avg_glucose_level", "평균 혈당", COLORS["purple"], bins), use_container_width=True)
    st.caption(f"평균 혈당: **{filtered['avg_glucose_level'].mean():.1f}**")

section("02", "GROUP COMPARISON", "뇌졸중 경험별 상자그림", "상자 가운데 선은 중앙값, 상자는 사분위 범위입니다. 평균은 그래프 아래 표에서 따로 확인하세요.")
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**02-A · 뇌졸중 경험별 나이**")
    st.plotly_chart(boxplot(filtered, "age", "나이 (세)"), use_container_width=True)
with c2:
    st.markdown("**02-B · 뇌졸중 경험별 평균 혈당**")
    st.plotly_chart(boxplot(filtered, "avg_glucose_level", "평균 혈당"), use_container_width=True)

means = (
    filtered.groupby("stroke_group")
    .agg(people=("id", "size"), avg_age=("age", "mean"), avg_glucose=("avg_glucose_level", "mean"))
    .reindex(GROUPS)
    .reset_index()
    .rename(columns={
        "stroke_group": "뇌졸중 경험", "people": "사람 수",
        "avg_age": "평균 나이 (세)", "avg_glucose": "평균 혈당",
    })
)
means["사람 수"] = means["사람 수"].fillna(0).astype(int)
st.markdown("**📋 두 집단의 평균값 비교**")
st.dataframe(
    means, hide_index=True, use_container_width=True,
    column_config={
        "사람 수": st.column_config.NumberColumn(format="%d명"),
        "평균 나이 (세)": st.column_config.NumberColumn(format="%.2f"),
        "평균 혈당": st.column_config.NumberColumn(format="%.2f"),
    },
)
if filtered["stroke"].nunique() < 2:
    st.info("현재 필터에는 뇌졸중 경험 집단이 한쪽만 존재합니다. 비교하려면 필터 범위를 넓혀 보세요.")

section("03", "HEALTH CONDITIONS", "고혈압·심장병 여부와 뇌졸중 비율", "각 집단의 뇌졸중 발생 인원 ÷ 해당 집단 전체 인원 × 100으로 계산했습니다.")
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**03-A · 고혈압 여부 (hypertension)**")
    fig_ht, table_ht = risk_bar(filtered, "hypertension")
    st.plotly_chart(fig_ht, use_container_width=True)
with c2:
    st.markdown("**03-B · 심장병 여부 (heart_disease)**")
    fig_hd, table_hd = risk_bar(filtered, "heart_disease")
    st.plotly_chart(fig_hd, use_container_width=True)
with st.expander("🔍 각 막대의 분모·분자 확인하기"):
    a, b = st.columns(2)
    for target, table, field in (
        (a, table_ht, "hypertension"),
        (b, table_hd, "heart_disease"),
    ):
        with target:
            st.markdown(f"**{field}**")
            display = table.rename(columns={field: "질환 여부", "people": "사람 수", "cases": "발생 수", "ratio": "비율 (%)"})
            st.dataframe(display, hide_index=True, use_container_width=True)

section("04", "MISSING VALUES", "BMI 빈 값 집단의 뇌졸중 비율", "BMI가 비어 있는 사람과 전체 선택 인원의 뇌졸중 비율을 비교합니다. 전체에는 BMI가 빈 사람도 포함됩니다.")
bmi_na = filtered[filtered["bmi"].isna()]
na_cases = int(bmi_na["stroke"].eq(1).sum())
bmi_table = pd.DataFrame([
    {"비교 집단": "BMI 빈 값인 사람", "사람 수": len(bmi_na), "뇌졸중 발생 수": na_cases,
     "뇌졸중 비율 (%)": as_percent(na_cases, len(bmi_na))},
    {"비교 집단": "전체 선택 인원", "사람 수": n, "뇌졸중 발생 수": cases,
     "뇌졸중 비율 (%)": as_percent(cases, n)},
])
st.dataframe(
    bmi_table, hide_index=True, use_container_width=True,
    column_config={
        "사람 수": st.column_config.NumberColumn(format="%d명"),
        "뇌졸중 발생 수": st.column_config.NumberColumn(format="%d명"),
        "뇌졸중 비율 (%)": st.column_config.NumberColumn(format="%.2f%%"),
    },
)
if bmi_na.empty:
    st.info("현재 선택한 데이터에는 BMI 빈 값이 없습니다. 첫 번째 집단의 비율을 계산할 수 없어 빈칸으로 표시합니다.")
else:
    st.caption(f"BMI 빈 값 비중: 전체 선택 인원 중 {len(bmi_na) / n * 100:.2f}%")

section("05", "SMOKING STATUS", "흡연 상태별 사람 수", "smoking_status 열의 실제 값을 그대로 세었습니다. 'Unknown'은 CSV에 적힌 범주값입니다.")
smoke = (
    filtered["smoking_status"].value_counts(dropna=False)
    .rename_axis("흡연 상태").reset_index(name="사람 수")
)
smoke["비율 (%)"] = smoke["사람 수"] / n * 100
left, right = st.columns([1.1, .9], gap="large")
with left:
    st.dataframe(
        smoke, hide_index=True, use_container_width=True,
        column_config={
            "사람 수": st.column_config.NumberColumn(format="%d명"),
            "비율 (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        },
    )
    st.download_button(
        "⬇️ 흡연 상태 집계 CSV 저장",
        data=smoke.to_csv(index=False).encode("utf-8-sig"),
        file_name="stroke_흡연상태_집계.csv",
        mime="text/csv",
    )
with right:
    st.info(
        "**📌 읽는 법**\n\n"
        "`Unknown`은 흡연 상태가 알려지지 않았다는 뜻의 **문자열 범주**입니다. "
        "`bmi`의 실제 빈 값(`NaN`)과는 구분해야 합니다.\n\n"
        "또한 집단별 발생 비율 차이만으로 원인과 결과를 단정할 수 없습니다."
    )

st.divider()
st.caption("🧠 뇌졸중 예측 실습실 · 이 페이지는 교육용 데이터 탐색이며 의료 진단을 제공하지 않습니다.")
st.page_link("main.py", label="← 데이터 소개 화면으로 돌아가기", icon="🏠")
