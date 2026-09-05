"""실행: streamlit run main.py"""
import math
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
    frame["평균기온"] = pd.to_numeric(frame["평균기온"], errors="coerce")
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


def histogram_data(values: pd.Series, width: int) -> pd.DataFrame:
    """0℃를 기준으로 같은 너비의 구간을 만들고 [하한, 상한)으로 센다."""
    lower = math.floor(values.min() / width) * width
    upper = (math.floor(values.max() / width) + 1) * width
    edges = list(range(lower, upper + 1, width))
    counts = pd.cut(values, bins=edges, right=False).value_counts(sort=False)
    return pd.DataFrame({
        "기온 구간": [f"{left}℃ 이상 ~ {right}℃ 미만" for left, right in zip(edges, edges[1:])],
        "구간 중심": [(left + right) / 2 for left, right in zip(edges, edges[1:])],
        "일수": counts.to_numpy(),
        "비율 (%)": counts.to_numpy() / len(values) * 100,
    })


def main():
    st.set_page_config(page_title="서울 일별 평균기온 분포", page_icon="🌡️", layout="wide")
    st.title("서울 일별 평균기온 분포")
    st.write("어느 기온 구간에 관측일이 많이 모여 있을까요? 막대가 높을수록 해당 기온의 날이 많습니다.")
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
    width = st.select_slider("기온 구간 너비 (℃)", options=[1, 2, 3, 5, 10], value=2)
    selected = daily.loc[daily["날짜"].dt.year.between(start, end)]
    values = selected["평균기온"].replace([float("inf"), float("-inf")], float("nan")).dropna()
    if values.empty:
        st.warning("선택한 기간에 유효한 일별 평균기온이 없습니다. 분석 기간을 바꿔 주세요.")
        st.stop()
    counts = histogram_data(values, width)
    st.caption(
        f"선택 기간 내 기록: {selected['날짜'].min():%Y-%m-%d} ~ {selected['날짜'].max():%Y-%m-%d} · "
        f"기온 구간 너비: {width}℃"
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("분석에 사용한 관측일", f"{len(values):,}일")
    col2.metric("일별 평균기온의 평균", f"{values.mean():.2f} ℃")
    col3.metric("일별 평균기온의 중앙값", f"{values.median():.2f} ℃")

    # 직접 집계한 동일 너비 구간을 붙여 그려 경계와 일수를 정확히 표시한다.
    fig = go.Figure(go.Bar(
        x=counts["구간 중심"], y=counts["일수"], width=width,
        customdata=counts[["기온 구간", "비율 (%)"]].to_numpy(),
        marker=dict(color="#4F8CC9", line=dict(color="white", width=1)),
        hovertemplate="%{customdata[0]}<br>%{y:,}일 · %{customdata[1]:.2f}%<extra></extra>",
        name="관측 일수",
    ))
    fig.update_layout(
        height=520, xaxis_title="일별 평균기온 (℃)", yaxis_title="관측 일수 (일)",
        template="plotly_white", font=dict(family="sans-serif"),
        margin=dict(l=30, r=20, t=25, b=30), bargap=0, showlegend=False,
    )
    fig.update_yaxes(rangemode="tozero", tickformat=",d")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    peaks = counts.loc[counts["일수"].eq(counts["일수"].max())]
    peak_labels = ", ".join(peaks["기온 구간"])
    st.info(f"가장 많은 구간: {peak_labels} · 구간당 {int(peaks.iloc[0]['일수']):,}일 ({peaks.iloc[0]['비율 (%)']:.2f}%)")
    excluded = len(selected) - len(values)
    st.caption(
        f"평균기온이 없거나 유효하지 않은 {excluded:,}개 행은 제외했습니다. "
        "자료에 날짜 자체가 없는 날도 집계하지 않습니다. "
        "일부 날짜만 있는 연도도 유효한 관측일은 포함하므로, 이 분포는 확보된 관측자료의 분포입니다."
    )
    with st.expander("집계 방법과 구간별 일수"):
        st.write(
            "서울 지점(108)의 일별 평균기온을 사용합니다. 각 구간은 하한을 포함하고 상한을 제외합니다. "
            "예를 들어 0℃ 이상 ~ 2℃ 미만 구간에는 0℃가 포함되고, 2℃는 다음 구간에 포함됩니다. "
            "비율은 해당 구간 일수를 선택 기간의 유효 관측일 수로 나눈 값입니다. "
            "이 히스토그램은 계절과 여러 연도가 섞인 기온 분포를 보여 주며, 시간에 따른 변화는 나타내지 않습니다."
        )
        table = counts.drop(columns="구간 중심")
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("구간별 자료 내려받기", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name="서울_일별_평균기온_분포.csv", mime="text/csv")
    st.markdown(f"[원본 데이터 확인]({DATA_URL})")


if __name__ == "__main__":
    main()
