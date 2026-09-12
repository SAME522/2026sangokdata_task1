import pandas as pd
import plotly.express as px
import streamlit as st

# ── 1. 앱 설정 ──────────────────────────────────────────────
TITLE = "영화 데이터 그래프 도감 1 - 시간"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
st.set_page_config(page_title=TITLE, page_icon="🎬", layout="wide")

# ── 2. 데이터 불러오기 및 날짜 변환 ──────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL, dtype=str)
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    required = {"날짜", "순위", "영화코드", "영화명", "일관객", "누적관객", "스크린수", "상영횟수"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필요한 열이 없습니다: {', '.join(sorted(missing))}")
    df["날짜"] = pd.to_datetime(df["날짜"].str.strip(), format="%Y%m%d", errors="raise")
    for column in ["순위", "일관객", "누적관객", "스크린수", "상영횟수"]:
        df[column] = pd.to_numeric(df[column].str.replace(",", "", regex=False), errors="raise")
    df = df.dropna(subset=["날짜", "영화코드", "영화명", "일관객"])
    if df.empty:
        raise ValueError("표시할 데이터가 없습니다.")
    return df.sort_values(["날짜", "순위"])

# 그래프마다 호출하고, 아래 기본 문구 또는 호출 시 전달하는 문장을 수정하세요.
def show_insight(sentence="여기에 이 그래프로 알 수 있는 내용을 한 문장으로 적어 주세요."):
    st.markdown("**이 그래프로 알 수 있는 것**")
    st.caption(sentence)

# ── 3. 첫 그래프: 영화별 일관객 변화 ─────────────────────────
def show_daily_audience(df):
    st.header("1. 영화별 일관객 변화")
    movies = df[["영화코드", "영화명"]].drop_duplicates("영화코드").sort_values("영화명")
    names = movies.set_index("영화코드")["영화명"].to_dict()
    duplicates = set(movies.loc[movies["영화명"].duplicated(keep=False), "영화명"])
    selected = st.selectbox(
        "영화를 선택하세요",
        movies["영화코드"].tolist(),
        format_func=lambda code: f"{names[code]} ({code})" if names[code] in duplicates else names[code],
        key="daily_audience_movie",
    )
    movie = df.loc[df["영화코드"] == selected, ["날짜", "일관객"]].sort_values("날짜")
    # 10위권 밖 날짜는 0명이 아닌 결측값으로 두어 선이 이어지지 않게 합니다.
    dates = pd.date_range(movie["날짜"].min(), movie["날짜"].max(), freq="D")
    movie = movie.set_index("날짜").reindex(dates).rename_axis("날짜").reset_index()
    fig = px.line(movie, x="날짜", y="일관객", markers=True,
                  title=f"{names[selected]} — 날짜별 일관객", labels={"일관객": "일관객 (명)"})
    fig.update_traces(connectgaps=False,
                      hovertemplate="날짜: %{x|%Y-%m-%d}<br>관객수: %{y:,.0f}명<extra></extra>")
    fig.update_layout(xaxis_tickformat="%Y-%m-%d", yaxis_tickformat=",", hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    show_insight()
    st.caption("일별 박스오피스 10위권에 기록된 날짜만 표시하며, 기록이 없는 날은 관객수 0명을 뜻하지 않습니다.")

# ── 4. 화면 구성 ────────────────────────────────────────────
def main():
    st.title(TITLE)
    try:
        with st.spinner("영화 데이터를 불러오는 중입니다..."):
            df = load_data()
    except Exception as exc:
        st.error(f"데이터를 불러오지 못했습니다. 주소와 CSV 형식을 확인해 주세요. ({exc})")
        st.stop()
    st.caption(f"데이터 기간: {df['날짜'].min():%Y-%m-%d} ~ {df['날짜'].max():%Y-%m-%d} · {df['날짜'].nunique()}일")
    with st.container():
        show_daily_audience(df)
    st.divider()
    # ── 5. 다음 그래프 추가 구역 ────────────────────────────
    # 새 그래프 함수를 정의한 뒤 여기에서 호출하세요.
    # with st.container():
    #     st.header("2. 다음 그래프 제목")
    #     st.plotly_chart(..., use_container_width=True)
    #     show_insight("이 그래프의 해석을 한 문장으로 입력하세요.")
    # st.divider()

if __name__ == "__main__":
    main()
