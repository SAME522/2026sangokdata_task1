import math

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
CLUSTER_LABELS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

FEATURES = {
    "스크린 수 (log10)": "log_first_scrn",
    "누적 관객 (log10)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

ORIGINAL_COLUMNS = {
    "스크린 수 평균": "first_scrn",
    "누적 관객 평균": "total_audi",
    "10위권 일수 평균": "days_in_top10",
    "롱런 지수 평균": "long_run_index",
}

COLOR_MAP = {
    "㉮": "#E45756",
    "㉯": "#4C78A8",
    "㉰": "#54A24B",
    "㉱": "#F2CF5B",
    "㉲": "#B279A2",
    "㉳": "#FF9DA6",
    "㉴": "#9D755D",
}


st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")


@st.cache_data
def load_and_prepare_data(url: str):
    raw = pd.read_csv(url, encoding="utf-8")
    total_count = len(raw)

    numeric_columns = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]
    data = raw.copy()
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    valid = (
        data[numeric_columns].notna().all(axis=1)
        & (data["first_week_audi"] > 0)
        & (data["first_scrn"] > 0)
        & (data["total_audi"] > 0)
    )
    data = data.loc[valid].copy()

    data["log_first_scrn"] = data["first_scrn"].map(math.log10)
    data["log_total_audi"] = data["total_audi"].map(math.log10)
    data["long_run_index"] = (
        data["total_audi"] / data["first_week_audi"]
    ).clip(upper=20)

    data = data.dropna(subset=list(FEATURES.values()))
    return data, total_count


try:
    movies, total_count = load_and_prepare_data(DATA_URL)
except Exception as error:
    st.error(f"데이터를 불러오지 못했습니다: {error}")
    st.stop()

st.write(f"전체 영화 **{total_count:,}편** · 묶음 분석에 사용한 영화 **{len(movies):,}편**")

control_col1, control_col2 = st.columns([3, 1])
with control_col1:
    selected_names = st.multiselect(
        "묶는 데 사용할 속성 (2개 이상)",
        options=list(FEATURES.keys()),
        default=list(FEATURES.keys()),
    )
with control_col2:
    cluster_count = st.slider("묶음 수", min_value=2, max_value=7, value=3, step=1)

if len(selected_names) < 2:
    st.warning("묶는 데 사용할 속성을 두 개 이상 골라 주세요.")
    st.stop()

selected_columns = [FEATURES[name] for name in selected_names]
scaled_values = StandardScaler().fit_transform(movies[selected_columns])

model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
movies["cluster_raw"] = model.fit_predict(scaled_values)

# 누적 관객 평균이 큰 기존 묶음부터 ㉮, ㉯, … 순서로 이름을 붙인다.
cluster_order = (
    movies.groupby("cluster_raw")["total_audi"].mean().sort_values(ascending=False).index
)
active_labels = CLUSTER_LABELS[:cluster_count]
label_map = {cluster: active_labels[index] for index, cluster in enumerate(cluster_order)}
movies["묶음"] = movies["cluster_raw"].map(label_map)
movies["묶음"] = pd.Categorical(
    movies["묶음"], categories=active_labels, ordered=True
)

st.subheader("2차원 산점도")
axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_name = st.selectbox("가로축", selected_names, index=0)
with axis_col2:
    y_name = st.selectbox("세로축", selected_names, index=1)

fig_2d = px.scatter(
    movies,
    x=FEATURES[x_name],
    y=FEATURES[y_name],
    color="묶음",
    color_discrete_map=COLOR_MAP,
    category_orders={"묶음": active_labels},
    hover_name="movieNm",
    labels={FEATURES[x_name]: x_name, FEATURES[y_name]: y_name},
    opacity=0.75,
)
fig_2d.update_traces(marker={"size": 7})
st.plotly_chart(fig_2d, use_container_width=True)

st.subheader("3차원 산점도")
if len(selected_names) < 3:
    st.info("3차원 산점도를 보려면 묶는 속성을 세 개 이상 골라 주세요.")
else:
    axis_col1, axis_col2, axis_col3 = st.columns(3)
    with axis_col1:
        x3_name = st.selectbox("x축", selected_names, index=0, key="x3")
    with axis_col2:
        y3_name = st.selectbox("y축", selected_names, index=1, key="y3")
    with axis_col3:
        z3_name = st.selectbox("z축", selected_names, index=2, key="z3")

    fig_3d = px.scatter_3d(
        movies,
        x=FEATURES[x3_name],
        y=FEATURES[y3_name],
        z=FEATURES[z3_name],
        color="묶음",
        color_discrete_map=COLOR_MAP,
        category_orders={"묶음": active_labels},
        hover_name="movieNm",
        labels={
            FEATURES[x3_name]: x3_name,
            FEATURES[y3_name]: y3_name,
            FEATURES[z3_name]: z3_name,
        },
        opacity=0.75,
    )
    fig_3d.update_traces(marker={"size": 3})
    fig_3d.update_layout(scene={"aspectmode": "data"})
    st.plotly_chart(fig_3d, use_container_width=True)

st.subheader("묶음별 요약")
summary = movies.groupby("묶음", observed=False).agg(
    편수=("movieCd", "size"),
    **{label: (column, "mean") for label, column in ORIGINAL_COLUMNS.items()},
)

formatted_summary = summary.copy()
formatted_summary["편수"] = formatted_summary["편수"].map(lambda value: f"{value:,.0f}")
for column in ["스크린 수 평균", "누적 관객 평균", "10위권 일수 평균"]:
    formatted_summary[column] = formatted_summary[column].map(lambda value: f"{value:,.1f}")
formatted_summary["롱런 지수 평균"] = formatted_summary["롱런 지수 평균"].map(
    lambda value: f"{value:,.2f}"
)
st.dataframe(formatted_summary, use_container_width=True)

st.subheader("묶음별 누적 관객 상위 5편")
top_movies = (
    movies.sort_values(["묶음", "total_audi"], ascending=[True, False])
    .groupby("묶음", observed=False)
    .head(5)
)

# 묶음이 많을 때도 읽기 쉽도록 한 줄에 최대 네 묶음씩 배치한다.
for start in range(0, cluster_count, 4):
    row_labels = active_labels[start : start + 4]
    columns = st.columns(len(row_labels))
    for column, label in zip(columns, row_labels):
        with column:
            st.markdown(f"#### {label}")
            titles = top_movies.loc[top_movies["묶음"] == label, "movieNm"].tolist()
            for rank, title in enumerate(titles, start=1):
                st.write(f"{rank}. {title}")

st.subheader("묶음 수에 따른 중심까지의 거리 제곱합")
inertia_rows = []
previous_inertia = None
for k in range(1, 8):
    k_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    k_model.fit(scaled_values)
    inertia = k_model.inertia_
    decrease = None if previous_inertia is None else previous_inertia - inertia
    inertia_rows.append(
        {"묶음 수": k, "거리 제곱합": inertia, "앞 값보다 줄어든 양": decrease}
    )
    previous_inertia = inertia

inertia_data = pd.DataFrame(inertia_rows)
fig_inertia = px.line(
    inertia_data,
    x="묶음 수",
    y="거리 제곱합",
    markers=True,
    labels={"묶음 수": "묶음 수(k)", "거리 제곱합": "중심까지의 거리 제곱합"},
)
fig_inertia.add_vline(
    x=cluster_count,
    line_dash="dash",
    line_color="#E45756",
    annotation_text=f"현재 선택: {cluster_count}",
    annotation_position="top",
)
fig_inertia.update_xaxes(dtick=1)
st.plotly_chart(fig_inertia, use_container_width=True)

formatted_inertia = inertia_data.copy()
formatted_inertia["거리 제곱합"] = formatted_inertia["거리 제곱합"].map(
    lambda value: f"{value:,.2f}"
)
formatted_inertia["앞 값보다 줄어든 양"] = formatted_inertia[
    "앞 값보다 줄어든 양"
].map(lambda value: "" if pd.isna(value) else f"{value:,.2f}")
st.dataframe(formatted_inertia, hide_index=True, use_container_width=True)

score = silhouette_score(scaled_values, movies["cluster_raw"])
st.write(
    f"선택한 묶음 수 **{cluster_count}개**의 실루엣 점수는 **{score:.3f}**입니다. "
    "점수는 -1에서 1 사이이며, 1에 가까울수록 묶음이 뚜렷합니다."
)
