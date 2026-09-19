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


def evaluate_features(feature_names, train_frame, test_frame):
    """같은 고정 분할에서 변수 조합별 R²와 MAE를 계산한다."""
    fitted_model, x_train, date_columns = build_model(
        feature_names, train_frame[feature_names]
    )
    x_test, _ = date_parts(test_frame[feature_names], date_columns)
    fitted_model.fit(x_train, np.log1p(train_frame["total_audi"].astype(float)))
    predictions = np.maximum(0.0, np.expm1(fitted_model.predict(x_test)))
    observed = test_frame["total_audi"].astype(float).to_numpy()
    return r2_score(observed, predictions), mean_absolute_error(observed, predictions)


st.title("🎬 영화 흥행 예측기")
st.caption("KOBIS 영화 정보로 총 관객 수를 예측하는 다중 선형회귀 모델")
st.warning(
    "이 데이터에는 첫 주 관객 수처럼 개봉 후에 집계되는 값이 포함되어 있습니다. "
    "따라서 아래 결과는 사후 데이터를 이용한 평가이며, 실제 개봉 전 흥행 예측 성능을 "
    "뜻하지 않습니다."
)

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

basic_features = ["first_scrn", "first_show", "peak"]
first_week_features = basic_features + ["first_week_audi"]
basic_r2, basic_mae = evaluate_features(basic_features, train, test)
week_r2, week_mae = evaluate_features(first_week_features, train, test)

st.subheader("고정 변수 조합의 예측 점수 비교")
score_col1, score_col2 = st.columns(2)
with score_col1:
    st.metric("기본 변수 3개 · R²", f"{basic_r2:.3f}")
    st.caption(
        "첫 관측일 스크린 수 + 첫 관측일 상영 횟수 + 성수기 여부  "
        f"\n평균 절대 오차: {basic_mae:,.0f}명"
    )
with score_col2:
    st.metric("기본 변수 + 첫 주 관객 수 · R²", f"{week_r2:.3f}")
    st.caption(
        "기본 변수 3개 + 첫 주 관객 수  "
        f"\n평균 절대 오차: {week_mae:,.0f}명"
    )
st.caption("두 점수는 아래와 동일한 고정 학습·테스트 분할에서 계산했습니다.")

model, x_train, date_columns = build_model(selected_features, train[selected_features])
x_test, _ = date_parts(test[selected_features], date_columns)

# 관객 수의 큰 편차를 완화하기 위해 log1p(total_audi)를 회귀한다.
y_train_log = np.log1p(train["total_audi"].astype(float))
model.fit(x_train, y_train_log)
predicted = np.expm1(model.predict(x_test))
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
below_diagonal = predicted < actual
above_diagonal = predicted > actual
negative_prediction = predicted < 0
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
scatter_col, scatter_stats_col = st.columns([4, 1])
with scatter_col:
    st.plotly_chart(fig, use_container_width=True)
with scatter_stats_col:
    st.markdown("#### 산점도 읽기")
    st.metric("대각선 아래", f"{below_diagonal.sum():,}편")
    st.caption("예측보다 실제 관객이 많은 영화")
    st.metric("대각선 위", f"{above_diagonal.sum():,}편")
    st.caption("실제보다 예측 관객이 많은 영화")
    st.metric("음수 예측", f"{negative_prediction.sum():,}편")
    st.metric("가장 작은 예측값", f"{predicted.min():,.1f}명")
st.caption(f"예측값이 1,000명보다 작아 그래프 바닥(1,000명)에 표시된 영화: {below_floor.sum():,}편")

residuals = actual - predicted
error_detail = test[["movieCd", "movieNm", "first_scrn"]].copy()
error_detail["오차"] = residuals
error_detail["절대오차"] = np.abs(residuals)
largest_errors = error_detail.nlargest(8, "절대오차").sort_values("오차")
largest_errors["표시명"] = largest_errors.apply(
    lambda row: f"{row['movieNm']} · 첫 스크린 {int(row['first_scrn']):,}개", axis=1
)
total_absolute_error = error_detail["절대오차"].sum()
largest_error_share = (
    largest_errors["절대오차"].sum() / total_absolute_error * 100
    if total_absolute_error > 0
    else 0.0
)

error_fig = go.Figure()
for positive, label, color in (
    (True, "실제가 더 많음", "#2563EB"),
    (False, "예측이 더 많음", "#F97316"),
):
    direction_rows = largest_errors[(largest_errors["오차"] >= 0) == positive]
    error_fig.add_trace(
        go.Bar(
            x=direction_rows["오차"],
            y=direction_rows["표시명"],
            orientation="h",
            name=label,
            marker_color=color,
            text=direction_rows["오차"].map(lambda value: f"{value:+,.0f}명"),
            textposition="outside",
            customdata=np.c_[
                direction_rows["절대오차"], direction_rows["first_scrn"]
            ],
            hovertemplate=(
                "%{y}<br>실제−예측: %{x:+,.0f}명"
                "<br>절대오차: %{customdata[0]:,.0f}명"
                "<br>첫 관측일 스크린: %{customdata[1]:,.0f}개<extra></extra>"
            ),
        )
    )
error_fig.add_vline(x=0, line_width=2, line_color="#374151")
error_fig.update_layout(
    title="실제−예측 절대오차가 큰 영화 8편",
    xaxis_title="오차(명): 왼쪽은 과대예측, 오른쪽은 과소예측",
    yaxis=dict(
        title=None,
        categoryorder="array",
        categoryarray=largest_errors["표시명"].tolist(),
    ),
    barmode="overlay",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=300, r=80, t=90, b=50),
    height=520,
)
st.plotly_chart(error_fig, use_container_width=True)
st.caption(
    f"이 8편의 절대오차 합은 테스트 영화 전체 절대오차 합의 "
    f"{largest_error_share:.1f}%입니다."
)

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
