import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

FEATURES = {
    "개봉일": "openDt",
    "장르": "genre",
    "국가": "nation",
    "첫 관측일 스크린 수": "first_scrn",
    "첫 관측일 상영 횟수": "first_show",
    "10위권 첫 등장일": "first_date",
    "성수기 개봉 여부": "peak",
    "첫 주 관객 수": "first_week_audi",
    "10위권에 머문 일수": "days_in_top10",
}

DEFAULT_FEATURES = {
    "genre",
    "nation",
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi",
    "days_in_top10",
}


st.set_page_config(page_title="영화 흥행 예측기", page_icon="🎬", layout="wide")


@st.cache_data(show_spinner=False)
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8", dtype={"영화코드": "string"})
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8", dtype={"movieCd": "string"})
    return daily, movies


def date_parts(frame, columns):
    """날짜 열을 선형회귀에 쓸 수 있는 연·월·일 숫자로 펼친다."""
    result = frame.copy()
    expanded = []
    for column in columns:
        parsed = pd.to_datetime(
            result[column].astype("string").str.replace(r"\.0$", "", regex=True),
            format="%Y%m%d",
            errors="coerce",
        )
        for suffix, values in (
            ("year", parsed.dt.year),
            ("month", parsed.dt.month),
            ("day", parsed.dt.day),
        ):
            new_column = f"{column}_{suffix}"
            result[new_column] = values
            expanded.append(new_column)
    return result, expanded


def build_model(selected, x_train):
    categorical = [c for c in selected if c in {"genre", "nation"}]
    date_columns = [c for c in selected if c in {"openDt", "first_date"}]
    numeric = [c for c in selected if c not in categorical + date_columns]
    log_numeric = [
        c
        for c in numeric
        if c in {"first_scrn", "first_show", "first_week_audi", "days_in_top10"}
    ]
    plain_numeric = [c for c in numeric if c not in log_numeric]

    x_train, expanded_dates = date_parts(x_train, date_columns)
    transformers = []
    if log_numeric:
        transformers.append(
            (
                "log_numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("log1p", FunctionTransformer(np.log1p)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                log_numeric,
            )
        )
    if plain_numeric or expanded_dates:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                plain_numeric + expanded_dates,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", drop="first"),
                        ),
                    ]
                ),
                categorical,
            )
        )

    model = Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            ("regression", LinearRegression()),
        ]
    )
    return model, x_train, date_columns


st.title("🎬 영화 흥행 예측기")
st.caption("KOBIS 영화 정보로 총 관객 수를 예측하는 다중 선형회귀 모델")

try:
    daily, movies = load_data()
except Exception as error:
    st.error(f"데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요. ({error})")
    st.stop()

daily_dates = pd.to_datetime(daily["날짜"].astype("string"), format="%Y%m%d", errors="coerce")
period_start = daily_dates.min().strftime("%Y-%m-%d")
period_end = daily_dates.max().strftime("%Y-%m-%d")

st.sidebar.header("예측 변수 선택")
selected_features = []
for label, column in FEATURES.items():
    if st.sidebar.checkbox(label, value=column in DEFAULT_FEATURES, key=column):
        selected_features.append(column)

if not selected_features:
    st.warning("왼쪽에서 예측 변수를 하나 이상 선택해 주세요.")
    st.stop()

# 명세에 따른 고정 분할: 영화코드 정렬 후 각 10편 묶음의 앞 3편은 테스트용.
ordered = movies.sort_values("movieCd", kind="stable").reset_index(drop=True)
test_mask = (np.arange(len(ordered)) % 10) < 3
train = ordered.loc[~test_mask].copy()
test = ordered.loc[test_mask].copy()

model, x_train, date_columns = build_model(selected_features, train[selected_features])
x_test, _ = date_parts(test[selected_features], date_columns)

# 관객 수의 큰 편차를 완화하기 위해 log1p(total_audi)를 회귀한다.
y_train_log = np.log1p(train["total_audi"].astype(float))
model.fit(x_train, y_train_log)
predicted = np.maximum(0.0, np.expm1(model.predict(x_test)))
actual = test["total_audi"].astype(float).to_numpy()

r2 = r2_score(actual, predicted)
mae = mean_absolute_error(actual, predicted)
rmse = mean_squared_error(actual, predicted) ** 0.5
ape = np.abs(predicted - actual) / np.maximum(actual, 1)
median_ape = np.median(ape) * 100

st.subheader("모델 평가")
c1, c2, c3, c4 = st.columns(4)
c1.metric("학습 영화", f"{len(train):,}편")
c2.metric("평가 영화", f"{len(test):,}편")
c3.metric("결정계수 R²", f"{r2:.3f}")
c4.metric("평균 절대 오차", f"{mae:,.0f}명")

st.caption(
    f"기준 기간: {period_start} ~ {period_end} · "
    f"RMSE {rmse:,.0f}명 · 중앙 절대 백분율 오차 {median_ape:.1f}%"
)
st.info(
    "R²는 1에 가까울수록 좋습니다. 평균 절대 오차(MAE)는 예측이 실제 총 관객 수에서 "
    "평균적으로 얼마나 빗나갔는지를 관객 수로 나타냅니다. 관객 수 편차가 커서 모델은 "
    "총 관객 수의 log(1+x)를 학습했습니다."
)

below_floor = predicted < 1_000
plot_y = np.maximum(predicted, 1_000)
axis_min = 1_000.0
axis_max = max(actual.max(), plot_y.max()) * 1.15

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=actual[~below_floor],
        y=plot_y[~below_floor],
        mode="markers",
        name="예측 영화",
        text=test.loc[~below_floor, "movieNm"],
        customdata=np.c_[predicted[~below_floor]],
        hovertemplate=(
            "%{text}<br>실제: %{x:,.0f}명<br>예측: %{customdata[0]:,.0f}명<extra></extra>"
        ),
        marker=dict(size=9, opacity=0.72, color="#3B82F6"),
    )
)
fig.add_trace(
    go.Scatter(
        x=actual[below_floor],
        y=plot_y[below_floor],
        mode="markers",
        name="예측 1,000명 미만",
        text=test.loc[below_floor, "movieNm"],
        customdata=np.c_[predicted[below_floor]],
        hovertemplate=(
            "%{text}<br>실제: %{x:,.0f}명<br>예측: %{customdata[0]:,.0f}명"
            "<br>그래프에는 1,000명 위치로 표시<extra></extra>"
        ),
        marker=dict(size=10, symbol="triangle-up", color="#EF4444"),
    )
)
fig.add_trace(
    go.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        name="실제 = 예측",
        line=dict(color="#111827", dash="dash"),
        hoverinfo="skip",
    )
)
fig.update_layout(
    title="테스트 영화: 실제 총 관객 수와 예측 총 관객 수",
    xaxis=dict(title="실제 총 관객 수", type="log", range=[np.log10(axis_min), np.log10(axis_max)]),
    yaxis=dict(title="예측 총 관객 수", type="log", range=[np.log10(axis_min), np.log10(axis_max)]),
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=20, r=20, t=90, b=20),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"예측값이 1,000명보다 작아 그래프 바닥(1,000명)에 표시된 영화: {below_floor.sum():,}편")

with st.expander("테스트 영화별 오차 보기"):
    evaluation = test[["movieCd", "movieNm"]].copy()
    evaluation["실제 총 관객"] = actual.astype(int)
    evaluation["예측 총 관객"] = np.rint(predicted).astype(int)
    evaluation["절대 오차"] = np.rint(np.abs(predicted - actual)).astype(int)
    evaluation["절대 백분율 오차(%)"] = np.round(ape * 100, 1)
    st.dataframe(evaluation, use_container_width=True, hide_index=True)

st.subheader("영화별 표의 맨 위 10줄")
st.dataframe(movies.head(10), use_container_width=True, hide_index=True)

st.caption(
    f"전체 {len(ordered):,}편을 빠짐없이 사용했습니다. 영화코드 순으로 정렬한 뒤 "
    "각 10편마다 앞의 3편을 평가용으로, 나머지를 학습용으로 고정 분할했습니다."
)
