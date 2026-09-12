import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


TITLE = "영화 데이터 그래프 도감 2 - 분포와 관계"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
REQUIRED_COLUMNS = [
    "movieCd", "movieNm", "openDt", "genre", "nation", "first_scrn",
    "first_show", "first_week_audi", "total_audi", "days_in_top10",
]


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    data = pd.read_csv(
        DATA_URL,
        dtype={"movieCd": "string", "openDt": "string", "genre": "string"},
    )
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(missing)}")
    # 복합 장르도 영화 한 편으로 집계하며, 첫 번째 장르만 사용합니다.
    data["genre"] = (
        data["genre"].str.split("|", regex=False).str[0].str.strip()
        .replace("", pd.NA).fillna("미분류")
    )
    data["openDt"] = pd.to_datetime(
        data["openDt"], format="%Y%m%d", errors="coerce"
    )
    data["nation"] = data["nation"].astype("string").str.strip().replace("", pd.NA).fillna("미분류")
    data["movieNm"] = data["movieNm"].fillna("제목 미상")
    for column in ["total_audi", "first_scrn", "first_week_audi"]:
        values = pd.to_numeric(data[column], errors="coerce")
        data[column] = values.where(values.ge(0) & values.lt(float("inf")))
    return data


def show_insight(key: str) -> None:
    """각 그래프 아래에 재사용할 수 있는 한 문장 설명 구역입니다."""
    st.text_input(
        "이 그래프로 알 수 있는 것",
        placeholder="그래프를 보고 알게 된 내용을 한 문장으로 적어 보세요.",
        key=key,
    )


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon="🎬", layout="wide")
    st.title(TITLE)
    st.caption("박스오피스 10위권에 오른 영화 요약 데이터로 분포와 관계를 살펴봅니다.")
    st.markdown(f"[원본 CSV 보기]({DATA_URL})")

    try:
        data = load_data()
    except Exception as exc:
        st.error("데이터를 불러오지 못했습니다. 데이터 주소와 인터넷 연결을 확인해 주세요.")
        with st.expander("오류 상세"):
            st.text(str(exc))
        st.stop()

    if data.empty:
        st.warning("표시할 영화 데이터가 없습니다.")
        st.stop()

    st.metric("불러온 영화 편수", f"{len(data):,}편")
    if len(data) != 216:
        st.info(
            f"안내된 216편과 현재 CSV의 {len(data):,}편이 다릅니다. "
            "그래프는 현재 CSV 전체를 기준으로 표시합니다."
        )

    st.divider()
    with st.container():
        st.subheader("1. 장르별 영화 편수")
        st.caption("복합 장르는 첫 번째 장르만 사용하며, 장르가 비어 있으면 ‘미분류’로 집계합니다.")
        counts = (
            data["genre"].value_counts().rename_axis("장르")
            .reset_index(name="편수")
        )
        fig = px.pie(
            counts, names="장르", values="편수", hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_traces(
            # 반올림 전 비율이 3% 이상인 조각에만 비율을 표시합니다.
            text=[
                f"{count / counts['편수'].sum():.2%}"
                if count * 100 >= counts["편수"].sum() * 3 else ""
                for count in counts["편수"]
            ],
            textinfo="text",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>편수: %{value:,}편<br>비율: %{percent:.1%}<extra></extra>",
        )
        fig.update_layout(
            legend_title_text="장르",
            margin=dict(t=20, b=20, l=20, r=20),
            height=500,
            annotations=[dict(
                text=f"전체<br>{len(data):,}편", x=0.5, y=0.5,
                showarrow=False, font_size=20,
            )],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("3% 이상인 조각에만 비율을 표시합니다. 모든 조각에 마우스를 올리면 장르, 영화 편수, 비율을 확인할 수 있습니다.")
        show_insight("genre_insight")
    st.divider()
    audience = data.dropna(subset=["total_audi"])
    scatter = data.dropna(subset=["first_scrn", "total_audi"])
    palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    colors = {genre: palette[i % len(palette)] for i, genre in enumerate(sorted(data["genre"].unique()))}
    labels = {
        "movieNm": "영화명", "genre": "장르", "nation": "제작 국가",
        "total_audi": "총 관객 (명)", "first_scrn": "개봉일 스크린수",
        "first_week_audi": "개봉 첫 주 관객 (명)",
    }
    if len(audience) < len(data):
        st.caption("총 관객이 누락되었거나 유효하지 않은 영화는 관객수 그래프에서 제외합니다.")

    with st.container():
        st.subheader("2. 장르 안의 영화: 총 관객 트리맵")
        positive = audience[audience["total_audi"] > 0]
        st.caption("큰 칸은 장르, 그 안의 작은 칸은 영화입니다. 면적은 총 관객수에 비례합니다. 총 관객이 0인 영화는 면적이 없어 제외합니다.")
        if positive.empty:
            st.info("트리맵에 표시할 관객 데이터가 없습니다.")
        else:
            fig = px.treemap(
                positive, path=["genre", "movieNm"], values="total_audi",
                color="genre", color_discrete_map=colors, labels=labels,
            )
            fig.update_traces(hovertemplate="<b>%{label}</b><br>총 관객: %{value:,.0f}명<extra></extra>")
            fig.update_layout(height=600, margin=dict(t=20, b=20, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        show_insight("treemap_insight")
    st.divider()

    with st.container():
        st.subheader("3. 총 관객의 분포")
        # 그래프와 설명 계산에 동일한 고정 구간을 적용합니다.
        bin_size = 1_000_000
        st.caption("구간 폭은 100만 명이며, 각 구간은 시작값 이상·끝값 미만입니다.")
        if audience.empty:
            st.info("히스토그램에 표시할 관객 데이터가 없습니다.")
        else:
            values = audience["total_audi"]
            bin_end = (int(values.max() // bin_size) + 1) * bin_size
            fig = go.Figure(go.Histogram(
                x=values, xbins=dict(start=0, end=bin_end, size=bin_size),
                marker_color="#527AC7",
                hovertemplate="총 관객 구간: %{x}<br>영화 편수: %{y}편<extra></extra>",
            ))
            fig.update_layout(
                xaxis_title="총 관객 (명)", yaxis_title="영화 편수",
                yaxis=dict(dtick=1) if len(audience) < 20 else {},
                bargap=0.05,
            )
            st.plotly_chart(fig, use_container_width=True)
            bin_counts = (values // bin_size).astype(int).value_counts()
            largest_count = int(bin_counts.max())
            modes = sorted(bin_counts[bin_counts.eq(largest_count)].index)
            intervals = ", ".join(
                f"{int(i * bin_size):,}명 이상 {int((i + 1) * bin_size):,}명 미만"
                for i in modes
            )
            st.write(
                f"영화가 가장 많이 모인 구간은 {intervals}이며, "
                f"{'각각 ' if len(modes) > 1 else ''}{largest_count}편"
                f"(유효한 영화 {len(audience)}편 중 {largest_count / len(audience):.1%})입니다."
            )
            top_names = ", ".join(audience.loc[values.eq(values.max()), "movieNm"].astype(str))
            st.write(f"총 관객이 가장 많은 영화: {top_names} — {values.max():,.0f}명.")
        show_insight("histogram_insight")
    st.divider()

    with st.container():
        st.subheader("4. 개봉일 스크린수와 총 관객의 관계")
        if scatter.empty:
            st.info("산점도에 표시할 데이터가 없습니다.")
        else:
            fig = px.scatter(
                scatter, x="first_scrn", y="total_audi", color="genre",
                hover_name="movieNm", color_discrete_map=colors, labels=labels,
            )
            st.plotly_chart(fig, use_container_width=True)
        show_insight("scatter_insight")
    st.divider()

    with st.container():
        st.subheader("5. 장르별 총 관객 상자 그림")
        eligible = data["genre"].value_counts()
        eligible = eligible[eligible >= 10].index.tolist()
        box_data = audience[audience["genre"].isin(eligible)]
        st.caption("전체 데이터에서 영화가 10편 이상인 장르만 표시합니다. 점은 상자에서 1.5×IQR 기준의 수염을 벗어난 영화입니다.")
        if box_data.empty:
            st.info("영화가 10편 이상이며 유효한 관객 데이터가 있는 장르가 없습니다.")
        else:
            fig = px.box(
                box_data, x="genre", y="total_audi", color="genre",
                points="outliers", hover_name="movieNm",
                color_discrete_map=colors, labels=labels,
                category_orders={"genre": eligible},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        show_insight("box_insight")
    st.divider()

    with st.container():
        st.subheader("6. 첫 주 관객을 함께 보는 버블 그래프")
        bubble = scatter.dropna(subset=["first_week_audi"])
        st.caption("가로축은 개봉일 스크린수, 세로축은 총 관객입니다. 색은 장르, 점의 면적은 첫 주 관객을 나타냅니다. 첫 주 관객이 0인 영화는 점 크기가 0입니다.")
        if bubble.empty or bubble["first_week_audi"].max() == 0:
            st.info("크기로 표시할 첫 주 관객 데이터가 없습니다.")
        else:
            fig = px.scatter(
                bubble, x="first_scrn", y="total_audi", size="first_week_audi",
                color="genre", hover_name="movieNm", size_max=55,
                color_discrete_map=colors, labels=labels,
            )
            st.plotly_chart(fig, use_container_width=True)
        show_insight("bubble_insight")
    st.divider()

    with st.container():
        st.subheader("7. 제작 국가에서 장르로: 영화 편수 선버스트")
        st.caption("안쪽은 제작 국가, 바깥쪽은 장르이며 칸의 크기는 영화 편수입니다. 제작 국가는 CSV의 표기를 그대로 사용합니다.")
        country_genres = data.groupby(["nation", "genre"]).size().reset_index(name="편수")
        fig = px.sunburst(
            country_genres, path=["nation", "genre"], values="편수", labels=labels,
        )
        fig.update_traces(hovertemplate="<b>%{label}</b><br>영화 편수: %{value:,}편<extra></extra>")
        fig.update_layout(height=600, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        show_insight("sunburst_insight")
    st.divider()


if __name__ == "__main__":
    main()
