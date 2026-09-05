"""실행: streamlit run main.py"""
import calendar
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


def annual_data(frame: pd.DataFrame) -> pd.DataFrame:
    annual = frame.groupby(frame["날짜"].dt.year)["평균기온"].agg(
        **{"연평균 기온": "mean", "유효 관측일": "count"}
    )
    annual = annual.reindex(range(int(annual.index.min()), int(annual.index.max()) + 1))
    annual.index.name = "연도"
    annual["유효 관측일"] = annual["유효 관측일"].fillna(0).astype(int)
    annual["연간 일수"] = [366 if calendar.isleap(year) else 365 for year in annual.index]
    annual["완전한 연도"] = annual["유효 관측일"].eq(annual["연간 일수"])
    # 일부 계절만 관측된 해의 평균이 장기 변화로 오해되지 않도록 제외한다.
    annual.loc[~annual["완전한 연도"], "연평균 기온"] = float("nan")
    annual["10년 이동평균"] = annual["연평균 기온"].rolling(10, min_periods=10).mean()
    return annual


def main():
    st.set_page_config(page_title="서울의 100년 기온 변화", page_icon="🌡️", layout="wide")
    st.title("서울의 100년 기온 변화")
    st.write("해마다 달라지는 기온과 긴 시간에 걸친 변화를 함께 살펴보세요.")
    try:
        with st.spinner("서울 기온 데이터를 읽고 있습니다…"):
            daily = load_data()
            annual = annual_data(daily)
    except Exception as exc:
        st.error("데이터를 불러오지 못했습니다. 잠시 후 다시 시도하거나 main.py 옆에 seoul.csv를 넣어 주세요.")
        with st.expander("오류 상세"):
            st.text(str(exc))
        st.stop()

    valid = annual.dropna(subset=["연평균 기온"])
    if valid.empty:
        st.warning("한 해의 모든 날짜에 평균기온이 있는 연도가 없어 그래프를 만들 수 없습니다.")
        st.stop()
    end = int(valid.index.max())
    start = max(int(annual.index.min()), end - 99)
    mode = st.radio("표시 기간", ["최근 100년", "전체 기록"], horizontal=True)
    shown = annual.loc[start:end].copy() if mode == "최근 100년" else annual.copy()
    st.caption(
        f"원본 날짜 범위: {daily['날짜'].min():%Y-%m-%d} ~ {daily['날짜'].max():%Y-%m-%d} · "
        f"표시 기간: {shown.index.min()}~{shown.index.max()}년"
    )
    if mode == "최근 100년" and end - start + 1 < 100:
        st.info("자료 기간이 100년보다 짧아 확인 가능한 기간만 표시합니다.")

    usable = shown.dropna(subset=["연평균 기온"])
    first, last = usable.iloc[0], usable.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric(f"첫 유효 연도 · {usable.index[0]}년", f"{first['연평균 기온']:.2f} °C")
    col2.metric(f"마지막 유효 연도 · {usable.index[-1]}년", f"{last['연평균 기온']:.2f} °C")
    col3.metric("두 연도의 기온 차이", f"{last['연평균 기온'] - first['연평균 기온']:+.2f} °C")
    st.caption("기온 차이는 두 개별 연도의 비교이며, 장기 추세의 추정값은 아닙니다.")

    fig = go.Figure()
    for name, color, width in [("연평균 기온", "#4F8CC9", 1.7), ("10년 이동평균", "#E4572E", 3.5)]:
        fig.add_trace(go.Scatter(
            x=shown.index, y=shown[name], name=name,
            mode="lines+markers" if name == "연평균 기온" else "lines",
            line=dict(color=color, width=width), marker=dict(size=4), connectgaps=False,
            hovertemplate="%{x}년<br>%{y:.2f} °C<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        height=520, xaxis_title="연도", yaxis_title="기온 (°C)",
        template="plotly_white", font=dict(family="sans-serif"),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=30, r=20, t=45, b=30), hovermode="x unified",
    )
    fig.update_xaxes(tickformat="d")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    missing = int((~shown["완전한 연도"]).sum())
    if missing:
        st.info(f"관측일이 부족하거나 자료가 없는 {missing}개 연도는 계산에서 제외하고 선을 끊어 표시했습니다.")
    with st.expander("계산 방법과 연도별 자료"):
        st.write(
            "서울 지점(108)의 일별 평균기온을 연도별로 산술평균합니다. "
            "평균기온이 365일(윤년 366일) 모두 있는 해만 사용하며, 결측값을 0으로 바꾸거나 보간하지 않습니다. "
            "10년 이동평균은 해당 연도와 직전 9년의 연평균을 평균한 값으로, 10개 연도가 모두 유효할 때만 표시합니다. "
            "최근 100년은 마지막 완전한 연도와 그 이전 99개 달력 연도입니다. 원본의 최신 연도가 현재 연도와 다를 수 있습니다."
        )
        st.dataframe(shown.reset_index(), hide_index=True, use_container_width=True)
        st.download_button("연도별 자료 내려받기", shown.to_csv().encode("utf-8-sig"),
                           file_name="서울_연평균_기온.csv", mime="text/csv")
    st.markdown(f"[원본 데이터 확인]({DATA_URL})")


if __name__ == "__main__":
    main()
