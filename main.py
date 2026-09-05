import csv
import io
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
st.set_page_config(page_title="서울 일별 최저·최고기온", layout="wide")
st.title("서울의 일별 최저기온과 최고기온")
st.write("점 하나는 하루입니다. 오른쪽 위로 점들이 모일수록 최저기온이 높은 날에 최고기온도 높은 경향을 뜻합니다.")


def parse_data(raw):
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV의 한글 인코딩을 읽을 수 없습니다.")
    lines = decoded.splitlines()
    header = next((i for i, line in enumerate(lines)
                   if "날짜" in line and "최저기온" in line and "최고기온" in line), None)
    if header is None:
        raise ValueError("날짜·최저기온·최고기온 열을 찾을 수 없습니다.")
    rows = list(csv.reader(lines[header:]))
    columns = [name.strip() for name in rows[0]]
    frame = pd.DataFrame([row for row in rows[1:] if len(row) == len(columns)], columns=columns)
    selected = {}
    for name in ("날짜", "최저기온", "최고기온"):
        source = next(col for col in columns if col.startswith(name))
        selected[name] = frame[source]
    data = pd.DataFrame(selected)
    data["날짜"] = pd.to_datetime(data["날짜"].str.strip(), errors="coerce")
    for name in ("최저기온", "최고기온"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    return data.dropna(subset=["날짜"]).sort_values("날짜")


@st.cache_data(ttl=86400)
def load_data():
    with urlopen(DATA_URL, timeout=30) as response:
        return parse_data(response.read())


try:
    data = load_data()
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다: {exc}")
    st.info("인터넷 연결과 원본 CSV 주소를 확인한 후 다시 실행해 주세요.")
    st.stop()

if data.empty:
    st.warning("날짜가 유효한 관측 자료가 없습니다.")
    st.stop()

first, last = int(data["날짜"].dt.year.min()), int(data["날짜"].dt.year.max())
start, end = (first, last)
if first < last:
    start, end = st.sidebar.slider("분석 연도", first, last, (first, last))
period = data.loc[data["날짜"].dt.year.between(start, end)]
valid = period.dropna(subset=["최저기온", "최고기온"]).copy()
missing = len(period) - len(valid)
invalid = int((valid["최저기온"] > valid["최고기온"]).sum())
valid = valid.loc[valid["최저기온"] <= valid["최고기온"]].copy()
st.caption(f"선택 기간: {start}~{end}년 · 기온 결측 {missing:,}행, 최저기온이 최고기온보다 높은 {invalid:,}행 제외")
if valid.empty:
    st.warning("선택한 기간에 두 기온을 함께 확인할 수 있는 자료가 없습니다.")
    st.stop()

valid["일교차"] = valid["최고기온"] - valid["최저기온"]
valid["날짜"] = valid["날짜"].dt.strftime("%Y-%m-%d")
c1, c2, c3 = st.columns(3)
c1.metric("표시한 관측 일수", f"{len(valid):,}일")
c2.metric("평균 일교차", f"{valid['일교차'].mean():.1f}℃")
corr = valid["최저기온"].corr(valid["최고기온"]) if len(valid) > 1 else float("nan")
c3.metric("피어슨 상관계수", f"{corr:.3f}" if pd.notna(corr) else "계산 불가")

fig = px.scatter(valid, x="최저기온", y="최고기온", hover_name="날짜",
                 hover_data={"최저기온": ":.1f", "최고기온": ":.1f", "일교차": ":.1f"},
                 labels={"최저기온": "일별 최저기온 (℃)", "최고기온": "일별 최고기온 (℃)", "일교차": "일교차 (℃)"},
                 opacity=0.3, render_mode="webgl")
fig.update_traces(marker={"size": 4, "color": "#2878B5"})
low = float(valid["최저기온"].min()) - 2
high = float(valid["최고기온"].max()) + 2
fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high,
              line={"color": "gray", "dash": "dash"}, layer="below")
fig.update_layout(height=650, xaxis={"range": [low, high]},
                  yaxis={"range": [low, high], "scaleanchor": "x", "scaleratio": 1})
st.plotly_chart(fig, use_container_width=True)
st.caption("회색 점선은 최저기온과 최고기온이 같은 위치입니다. 점이 선보다 위에 있을수록 일교차가 큽니다. 겹치는 점은 반투명으로 표시하며 모든 유효 관측을 사용합니다.")
st.caption("상관계수는 선택 기간의 관계를 요약하며 계절 변동도 반영합니다. 인과관계를 뜻하지 않습니다.")
st.download_button("산점도 자료 다운로드 (CSV)", valid.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"seoul_min_max_{start}_{end}.csv", mime="text/csv")
st.markdown(f"[원본 서울 기온 데이터]({DATA_URL})")
