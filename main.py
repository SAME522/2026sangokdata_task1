import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
BASE_YEAR = 1908
LAST_CLASS_YEAR = 2025
MIN_OBSERVATION_DAYS = 300


st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")


@st.cache_data(show_spinner="서울 기온 데이터를 불러오는 중입니다...")
def load_annual_temperature() -> pd.DataFrame:
    daily = pd.read_csv(DATA_URL, encoding="utf-8")
    required = {"날짜", "지점", "평균기온", "최저기온", "최고기온"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(sorted(missing))}")

    daily["날짜"] = pd.to_datetime(daily["날짜"], errors="coerce")
    daily["평균기온"] = pd.to_numeric(daily["평균기온"], errors="coerce")
    daily = daily.dropna(subset=["날짜", "평균기온"]).copy()
    daily["연도"] = daily["날짜"].dt.year
    daily = daily[daily["연도"] <= LAST_CLASS_YEAR]

    # 같은 날짜의 중복 행이 있어도 관측일은 한 번만 셉니다.
    annual = (
        daily.groupby("연도", as_index=False)
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일=("날짜", "nunique"),
        )
        .query("관측일 >= @MIN_OBSERVATION_DAYS")
        .sort_values("연도")
        .reset_index(drop=True)
    )
    if len(annual) < 2:
        raise ValueError("회귀분석에 필요한 유효 연도가 2개 미만입니다.")
    return annual


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온 추세를 단순 선형회귀로 살펴봅니다.")

try:
    annual = load_annual_temperature()
except Exception as exc:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {exc}")
    st.stop()

x = annual["연도"].to_numpy(dtype=float) - BASE_YEAR
y = annual["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
correlation = float(np.corrcoef(x, y)[0, 1])
annual["회귀기온"] = intercept + slope * x

last_year = int(annual["연도"].max())
recent_start_year = last_year - 19
recent = annual[annual["연도"] >= recent_start_year]
if len(recent) < 2:
    st.error("최근 20년 기울기를 계산할 유효 연도가 2개 미만입니다.")
    st.stop()
recent_x = recent["연도"].to_numpy(dtype=float) - BASE_YEAR
recent_y = recent["연평균기온"].to_numpy(dtype=float)
recent_slope, _ = np.polyfit(recent_x, recent_y, 1)

st.subheader("기온 상승 속도 비교")
overall_col, recent_col = st.columns(2)
with overall_col:
    st.metric(
        "전체 기간 기울기",
        f"{slope * 100:+.2f} °C / 100년",
        help="전체 유효 연도의 회귀 직선 기울기를 100년 단위로 환산한 값입니다.",
    )
with recent_col:
    st.metric(
        "최근 20년 기울기",
        f"{recent_slope * 100:+.2f} °C / 100년",
        help=(
            f"{recent_start_year}~{last_year}년 구간 중 기준을 충족한 "
            f"{len(recent)}개 연도로 계산했습니다."
        ),
    )
st.caption(
    f"최근 20년 비교 구간: {recent_start_year}~{last_year}년 "
    f"(유효 연도 {len(recent)}개)"
)

selected_year = st.slider("예측할 연도", 1900, 2100, 2025, 1)
predicted_temperature = intercept + slope * (selected_year - BASE_YEAR)

left, right = st.columns([1, 2])
with left:
    st.metric(
        label=f"{selected_year}년 예상 연평균기온",
        value=f"{predicted_temperature:.2f} °C",
    )
    st.write(
        f"회귀 직선에 사용한 해는 **{len(annual):,}개**이며, "
        f"기간은 **{int(annual['연도'].min())}년부터 "
        f"{int(annual['연도'].max())}년까지**입니다."
    )
    st.write(f"연도와 연평균기온의 피어슨 상관계수: **{correlation:.3f}**")
    st.caption(
        f"회귀식: 기온 = {intercept:.3f} + {slope:.4f} × (연도 − {BASE_YEAR})"
    )

with right:
    scatter = px.scatter(
        annual,
        x="연도",
        y="연평균기온",
        hover_data={"관측일": True, "회귀기온": ":.2f"},
        labels={"연평균기온": "연평균기온 (°C)"},
        title="서울 연평균기온과 선형 추세",
    )
    scatter.add_scatter(
        x=annual["연도"],
        y=annual["회귀기온"],
        mode="lines",
        name="회귀 직선",
        line={"color": "crimson", "width": 3},
    )
    scatter.update_xaxes(tickmode="linear", dtick=10, tickformat="d", title="연도")
    scatter.update_yaxes(title="연평균기온 (°C)")
    scatter.update_layout(hovermode="x unified", legend_title_text="")
    st.plotly_chart(scatter, use_container_width=True)

st.info(
    "이 값은 과거 추세를 직선으로 연장한 통계적 예상치이며, "
    "기후 시나리오를 반영한 장기 기후전망은 아닙니다."
)
