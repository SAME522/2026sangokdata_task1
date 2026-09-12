import pandas as pd
import plotly.express as px
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
            textinfo="percent",
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
        show_insight("genre_insight")
    st.divider()
    # 다음 그래프도 별도 container에 넣고 show_insight(고유한 키)를 호출하세요.


if __name__ == "__main__":
    main()
