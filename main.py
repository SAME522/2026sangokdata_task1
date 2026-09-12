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
    # 영화코드별로 서로 다른 10위권 진입 날짜를 셉니다.
    top_ten = df.loc[df["순위"].between(1, 10)]
    counts = top_ten.groupby("영화코드")["날짜"].nunique().rename("진입횟수")
    movies = df[["영화코드", "영화명"]].drop_duplicates("영화코드").join(counts, on="영화코드")
    movies = movies.loc[movies["진입횟수"] > 2].copy()
    if movies.empty:
        st.info("10위권에 3일 이상 기록된 영화가 없습니다.")
        return
    sort_order = st.selectbox(
        "영화 목록 정렬", ["10위권 진입 횟수순", "가나다순"], index=0, key="daily_audience_sort_rank_default"
    )
    if sort_order == "10위권 진입 횟수순":
        movies = movies.sort_values(["진입횟수", "영화명", "영화코드"], ascending=[False, True, True])
    else:
        movies = movies.sort_values(["영화명", "영화코드"])
    entry_counts = movies.set_index("영화코드")["진입횟수"].to_dict()
    st.caption("10위권 진입이 3회 이상인 영화만 표시합니다. 하루를 1회로 세며, 진입 횟수가 같으면 가나다순으로 정렬합니다.")
    names = movies.set_index("영화코드")["영화명"].to_dict()
    duplicates = set(movies.loc[movies["영화명"].duplicated(keep=False), "영화명"])
    selected = st.selectbox(
        "영화를 선택하세요",
        movies["영화코드"].tolist(),
        format_func=lambda code: (f"{names[code]} ({code})" if names[code] in duplicates else names[code]) + f" · 10위권 {int(entry_counts[code])}회",
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

# ── 그래프 2: 기간 합계 상위 5편 비교 ────────────────────────
def show_top_five(df):
    st.header("2. 기간 일관객 합계 상위 5편 비교")
    totals = df.groupby("영화코드")["일관객"].sum().sort_values(ascending=False, kind="stable").head(5)
    names = df.drop_duplicates("영화코드").set_index("영화코드")["영화명"]
    labels = {code: f"{names[code]} ({code})" for code in totals.index}
    dates = pd.date_range(df["날짜"].min(), df["날짜"].max(), freq="D")
    selected = df[df["영화코드"].isin(totals.index)]
    index = pd.MultiIndex.from_product([totals.index, dates], names=["영화코드", "날짜"])
    # 기록이 없는 날은 결측값으로 남겨 선을 끊습니다.
    comparison = selected.groupby(["영화코드", "날짜"])["일관객"].sum().reindex(index).reset_index()
    comparison["영화"] = comparison["영화코드"].map(labels)
    fig = px.line(comparison, x="날짜", y="일관객", color="영화", markers=True,
                  category_orders={"영화": list(labels.values())},
                  labels={"일관객": "일관객 (명)"}, custom_data=["영화"])
    fig.update_traces(connectgaps=False,
                      hovertemplate="%{customdata[0]}<br>날짜: %{x|%Y-%m-%d}<br>관객수: %{y:,.0f}명<extra></extra>")
    fig.update_layout(xaxis_tickformat="%Y-%m-%d", yaxis_tickformat=",",
                      legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("범례를 클릭하면 해당 영화를 켜거나 끌 수 있습니다. 선정 기준은 이 기간 10위권 기록의 일관객 합계이며, 기록이 없는 날은 0명으로 처리하지 않습니다.")
    show_insight("기간 일관객 합계 상위 5편의 관객수 변화와 관객이 집중된 시기를 비교할 수 있다.")


# ── 그래프 3: 날짜별 10위권 일관객 합계 ──────────────────────
def show_daily_total(df):
    st.header("3. 날짜별 10위권 일관객 합계")
    daily = df.groupby("날짜", as_index=False)["일관객"].sum().sort_values("날짜")
    top_days = daily.sort_values(["일관객", "날짜"], ascending=[False, True]).head(3)
    fig = px.area(daily, x="날짜", y="일관객", labels={"일관객": "10위권 일관객 합계 (명)"})
    fig.update_traces(hovertemplate="날짜: %{x|%Y-%m-%d}<br>합계: %{y:,.0f}명<extra></extra>")
    fig.add_scatter(x=top_days["날짜"], y=top_days["일관객"], mode="markers",
                    marker=dict(color="#dc2626", size=10), showlegend=False,
                    hovertemplate="날짜: %{x|%Y-%m-%d}<br>합계: %{y:,.0f}명<extra></extra>")
    for rank, (_, row) in enumerate(top_days.iterrows(), start=1):
        fig.add_annotation(x=row["날짜"], y=row["일관객"],
                           text=f"{rank}위 · {row['날짜']:%Y-%m-%d}",
                           showarrow=True, arrowhead=2, ax=(rank - 2) * 100,
                           ay=-40 - (rank - 1) * 35, bgcolor="white", bordercolor="#dc2626")
    fig.update_layout(xaxis_tickformat="%Y-%m-%d", yaxis_tickformat=",",
                      yaxis_range=[0, daily["일관객"].max() * 1.5], margin=dict(t=100))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("전체 영화 시장이 아닌 그날 10위권 영화의 합계입니다. 합계가 같으면 날짜가 빠른 날을 먼저 표시합니다.")
    show_insight("날짜별 10위권 관객 규모의 흐름과 관객 합계가 가장 컸던 세 날짜를 확인할 수 있다.")


# ── 그래프 4: 기간별 관객수 순위 및 TOP 10 ──────────────────
def build_period_ranking(df, start, end):
    selected = df.loc[df["날짜"].between(pd.Timestamp(start), pd.Timestamp(end)) & df["순위"].between(1, 10)]
    ranking = selected.groupby("영화코드", as_index=False).agg(
        영화명=("영화명", "first"), 관객수합계=("일관객", "sum"), 진입일수=("날짜", "nunique")
    )
    ranking = ranking.sort_values(["관객수합계", "영화명", "영화코드"], ascending=[False, True, True]).reset_index(drop=True)
    ranking.insert(0, "순위", ranking["관객수합계"].rank(method="min", ascending=False).astype(int))
    return ranking


def show_period_ranking(df):
    st.header("4. 기간별 관객수 순위 · TOP 10")
    period = st.date_input(
        "관객수 집계 기간",
        value=(df["날짜"].min().date(), df["날짜"].max().date()),
        min_value=df["날짜"].min().date(), max_value=df["날짜"].max().date(),
        key="ranking_period",
    )
    if len(period) != 2:
        st.info("집계 종료일도 선택해 주세요.")
        return
    ranking = build_period_ranking(df, period[0], period[1])
    if ranking.empty:
        st.info("선택한 기간에 기록된 영화가 없습니다.")
        return
    top = ranking.head(10).copy()
    duplicated = top["영화명"].duplicated(keep=False)
    top["영화"] = top["영화명"]
    top.loc[duplicated, "영화"] = top.loc[duplicated, "영화명"] + " (" + top.loc[duplicated, "영화코드"] + ")"
    fig = px.bar(
        top, x="관객수합계", y="영화", orientation="h", text="관객수합계",
        custom_data=["영화명", "진입일수", "순위"],
        labels={"관객수합계": "기간 일관객 합계 (명)"},
        title=f"{period[0]:%Y-%m-%d} ~ {period[1]:%Y-%m-%d} 관객수 TOP 10",
    )
    fig.update_traces(
        texttemplate="%{x:,.0f}", textposition="auto",
        hovertemplate="%{customdata[0]}<br>순위: %{customdata[2]}위<br>기간 관객수 합계: %{x:,.0f}명<br>기간 내 10위권 진입: %{customdata[1]}일<extra></extra>",
    )
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=top["영화"].tolist(), autorange="reversed"),
                      xaxis_tickformat=",", height=550)
    st.plotly_chart(fig, use_container_width=True)
    show_insight("선택한 기간의 일관객 합계가 많은 영화 10편과 각 영화의 10위권 진입 일수를 비교할 수 있다.")
    st.caption("관객수는 선택 기간의 10위권 기록만 합산하며, 진입 일수도 해당 기간에 기록된 날짜 수입니다. 개봉 이후 전체 관객수나 전체 진입 일수를 뜻하지 않습니다. 동률은 공동 순위로 표시하며 TOP 10 경계에서는 영화명·영화코드순으로 선정합니다.")
    with st.expander("전체 영화 관객수 순위표"):
        st.dataframe(
            ranking.rename(columns={"관객수합계": "기간 관객수 합계 (명)", "진입일수": "10위권 진입 일수"}),
            hide_index=True, use_container_width=True,
        )


# ── 그래프 5: 월 × 요일 일관객 합계 ─────────────────────────
def show_month_weekday(df):
    st.header("5. 월 × 요일 일관객 합계")
    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    calendar = df.assign(연월=df["날짜"].dt.strftime("%Y-%m"), 요일=df["날짜"].dt.dayofweek)
    matrix = calendar.groupby(["연월", "요일"])["일관객"].sum().unstack("요일").reindex(columns=range(7)).sort_index()
    matrix.columns = weekdays
    fig = px.imshow(matrix, color_continuous_scale="Blues", aspect="auto",
                    labels={"x": "요일", "y": "연월", "color": "일관객 합계 (명)"})
    fig.update_traces(hovertemplate="연월: %{y}<br>요일: %{x}<br>합계: %{z:,.0f}명<extra></extra>")
    fig.update_layout(xaxis=dict(side="bottom", categoryorder="array", categoryarray=weekdays),
                      coloraxis_colorbar=dict(tickformat=","))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("색이 진할수록 10위권 일관객 합계가 큽니다. 월별 포함 일수와 요일별 날짜 수가 다르므로 하루 평균을 뜻하지 않으며, 기록이 없는 조합은 빈칸으로 표시합니다.")
    show_insight("어느 연월과 요일 조합에 10위권 관객이 많이 모였는지 색의 진하기로 비교할 수 있다.")


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
    for render_graph in (show_top_five, show_daily_total, show_period_ranking, show_month_weekday):
        with st.container():
            render_graph(df)
        st.divider()
    # 이후 그래프는 별도 함수로 정의하고 위 화면 구성에 추가하세요.

if __name__ == "__main__":
    main()
