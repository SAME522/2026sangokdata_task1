"""실행: streamlit run main.py"""
import csv
import io
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"


def parse_data(raw: bytes) -> pd.DataFrame:
    """기상 자료의 인코딩, 안내문, 열 이름 단위 및 날짜 앞 공백을 처리한다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949 파일을 사용해 주세요.")

    rows = list(csv.reader(io.StringIO(text)))
    for index, row in enumerate(rows):
        names = [value.strip().lstrip("\ufeff").split("(")[0].strip() for value in row]
        if all(name in names for name in ("날짜", "지점", "평균기온", "최저기온", "최고기온")):
            break
    else:
        raise ValueError("날짜·지점·평균기온·최저기온·최고기온 열을 찾지 못했습니다.")

    frame = pd.DataFrame(
        [row for row in rows[index + 1:] if len(row) == len(names)], columns=names
    )
    frame["날짜"] = pd.to_datetime(frame["날짜"].str.strip(), format="%Y-%m-%d", errors="coerce")
    for column in ("평균기온", "최저기온", "최고기온"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["날짜"])
    frame = frame.loc[pd.to_numeric(frame["지점"], errors="coerce").eq(108)]
    if frame.empty:
        raise ValueError("서울(지점 108)의 유효한 날짜 자료가 없습니다.")
    if frame["날짜"].duplicated().any():
        raise ValueError("중복된 날짜가 있습니다. 서울 자료의 날짜별 행을 확인해 주세요.")
    return frame.sort_values("날짜")


@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> pd.DataFrame:
    local = Path(__file__).with_name("seoul.csv")
    if local.is_file():
        raw = local.read_bytes()
    else:
        with urlopen(DATA_URL, timeout=30) as response:
            raw = response.read()
    return parse_data(raw)


def scatter_data(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = frame[["날짜", "최저기온", "최고기온"]].replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna(subset=["최저기온", "최고기온"])
    pairs = pairs.loc[pairs["최저기온"].le(pairs["최고기온"])].copy()
    pairs["일교차"] = pairs["최고기온"] - pairs["최저기온"]
    return pairs


def main():
    st.set_page_config(page_title="서울 최저·최고기온의 관계", page_icon="🌡️", layout="wide")
    st.title("서울 최저·최고기온의 관계")
    st.write("점 하나는 하루입니다. 오른쪽 위에 있는 점일수록 그날의 최저기온과 최고기온이 모두 높습니다.")
    try:
        with st.spinner("서울 기온 데이터를 읽고 있습니다…"):
            daily = load_data()
    except Exception as exc:
        st.error("데이터를 불러오지 못했습니다. 잠시 후 다시 시도하거나 main.py 옆에 seoul.csv를 넣어 주세요.")
        with st.expander("오류 상세"):
            st.text(str(exc))
        st.stop()

    first_year = int(daily["날짜"].dt.year.min())
    last_year = int(daily["날짜"].dt.year.max())
    if first_year < last_year:
        start, end = st.slider("분석 기간 (연도)", first_year, last_year, (first_year, last_year))
    else:
        start = end = first_year
    selected = daily.loc[daily["날짜"].dt.year.between(start, end)]
    pairs = scatter_data(selected)
    if pairs.empty:
        st.warning("선택한 기간에 최저기온과 최고기온이 모두 유효한 자료가 없습니다. 분석 기간을 바꿔 주세요.")
        st.stop()
    st.caption(f"분석에 사용한 날짜: {pairs['날짜'].min():%Y-%m-%d} ~ {pairs['날짜'].max():%Y-%m-%d}")
    correlation = float("nan")
    if len(pairs) >= 2 and pairs["최저기온"].nunique() > 1 and pairs["최고기온"].nunique() > 1:
        correlation = pairs["최저기온"].corr(pairs["최고기온"])
    col1, col2, col3 = st.columns(3)
    col1.metric("분석에 사용한 관측일", f"{len(pairs):,}일")
    col2.metric("피어슨 상관계수", f"{correlation:.3f}" if pd.notna(correlation) else "계산 불가")
    col3.metric("평균 일교차", f"{pairs['일교차'].mean():.2f} ℃")

    fig = go.Figure(go.Scattergl(
        x=pairs["최저기온"], y=pairs["최고기온"], mode="markers",
        text=pairs["날짜"].dt.strftime("%Y-%m-%d"),
        customdata=pairs[["일교차"]].to_numpy(),
        marker=dict(size=5, opacity=0.3, color="#397CB8"),
        hovertemplate=("%{text}<br>최저기온: %{x:.1f} ℃<br>최고기온: %{y:.1f} ℃"
                       "<br>일교차: %{customdata[0]:.1f} ℃<extra></extra>"),
        name="일별 기온",
    ))
    lower = float(pairs["최저기온"].min()) - 2
    upper = float(pairs["최고기온"].max()) + 2
    fig.add_trace(go.Scatter(
        x=[lower, upper], y=[lower, upper], mode="lines",
        line=dict(color="#9CA3AF", dash="dash", width=1.5),
        name="최저기온 = 최고기온", hoverinfo="skip",
    ))
    fig.update_layout(
        height=620, xaxis_title="일별 최저기온 (℃)", yaxis_title="일별 최고기온 (℃)",
        template="plotly_white", font=dict(family="sans-serif"),
        margin=dict(l=30, r=20, t=30, b=30),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(range=[lower, upper], constrain="domain")
    fig.update_yaxes(range=[lower, upper], scaleanchor="x", scaleratio=1, constrain="domain")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("점이 겹치는 곳은 더 진하게 보입니다. 회색 점선은 최저기온과 최고기온이 같은 기준선이며, 점선보다 위로 멀수록 일교차가 큽니다.")
    excluded = len(selected) - len(pairs)
    st.caption(
        f"최저·최고기온 중 하나라도 없거나 유효하지 않은 행, 최저기온이 최고기온보다 높은 행 등 {excluded:,}개 행을 제외했습니다. "
        "평균기온의 결측 여부와 관계없이 두 기온이 유효하면 포함합니다."
    )
    with st.expander("해석 방법과 일별 자료"):
        st.write(
            "같은 날짜의 최저기온과 최고기온을 짝지어 모든 유효 관측일을 표시합니다. "
            "상관계수는 두 기온의 선형 관계를 요약하며, +1에 가까울수록 함께 높아지는 경향이 강합니다. "
            "관측일이 2일 미만이거나 어느 한 기온이 모두 같으면 상관계수를 계산하지 않습니다. "
            "여러 계절과 연도를 합친 관계에는 계절 변화도 반영되며, 상관관계만으로 인과관계나 장기 온난화 추세를 판단할 수 없습니다."
        )
        table = pairs.rename(columns={"최저기온": "최저기온 (℃)", "최고기온": "최고기온 (℃)", "일교차": "일교차 (℃)"})
        table["날짜"] = table["날짜"].dt.strftime("%Y-%m-%d")
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("산점도 자료 내려받기", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name="서울_최저기온_최고기온.csv", mime="text/csv")
    st.markdown(f"[원본 데이터 확인]({DATA_URL})")


if __name__ == "__main__":
    main()
