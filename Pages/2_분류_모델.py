
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree


# ============================================================
# NEURO LAB — 분류 모델 실습실
#
# 중요:
# - st.set_page_config()는 main.py에서 관리
# - 이 파일은 Pages/2_분류_모델.py 에 저장
# - 디자인용 HTML은 반드시 st.html() 사용
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/main/data/stroke.csv"
)

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
]

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "BMI",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부",
}

RED = "#F34E67"
BLUE = "#386BFA"
GREEN = "#19B58A"
YELLOW = "#FFC83D"
PINK = "#E968C6"
INK = "#202943"

SEED = 42


# ============================================================
# 01. 디자인
# ============================================================

st.html("""
<style>
.robot-hero {
    position: relative;
    overflow: hidden;
    padding: 42px;
    border-radius: 22px;
    background: linear-gradient(
        115deg,
        #F4F7FF 0%,
        #EEF9FF 52%,
        #FFF0F9 100%
    );
    border: 1px solid #DFE5FF;
    margin: 12px 0 30px;
}

.robot-hero::after {
    content: "✦";
    position: absolute;
    top: -38px;
    right: 26px;
    font-size: 200px;
    color: #E968C621;
    pointer-events: none;
}

.robot-kicker {
    position: relative;
    z-index: 1;
    color: #386BFA;
    font-weight: 900;
    font-size: 12px;
    letter-spacing: .16em;
}

.robot-title {
    position: relative;
    z-index: 1;
    color: #202943;
    font-size: clamp(34px, 5vw, 60px);
    font-weight: 900;
    line-height: 1.17;
    letter-spacing: -.065em;
    margin: 16px 0 19px;
}

.robot-title span {
    background: linear-gradient(
        90deg,
        #F34E67,
        #E968C6,
        #386BFA
    );
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.robot-description {
    position: relative;
    z-index: 1;
    max-width: 680px;
    color: #50617E;
    font-size: 15px;
    line-height: 1.9;
}

.robot-colors {
    display: flex;
    gap: 6px;
    max-width: 270px;
    margin-top: 28px;
}

.robot-colors span {
    display: block;
    flex: 1;
    height: 8px;
    border-radius: 99px;
}

.robot-section {
    margin: 48px 0 18px;
    padding-top: 19px;
    border-top: 1px solid #DFE5F1;
    display: flex;
    gap: 17px;
    align-items: baseline;
}

.robot-section .num {
    color: #386BFA;
    font-size: 12px;
    font-weight: 900;
}

.robot-section .heading {
    color: #202943;
    font-size: clamp(23px, 3vw, 30px);
    font-weight: 850;
    letter-spacing: -.04em;
}

.robot-subtitle {
    color: #64718B;
    font-size: 14px;
    line-height: 1.8;
    margin: -5px 0 23px;
}

.robot-score {
    background: white;
    border: 1px solid #DEE6F3;
    border-radius: 15px;
    padding: 22px;
    min-height: 165px;
    box-shadow: 0 8px 24px #283F7310;
}

.robot-score .ribbon {
    display: inline-block;
    color: white;
    border-radius: 5px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 900;
}

.robot-score .label {
    color: #4F5B70;
    font-size: 13px;
    margin-top: 14px;
}

.robot-score .value {
    color: #202943;
    font-size: clamp(32px, 4vw, 48px);
    line-height: 1.35;
    letter-spacing: -.06em;
    font-weight: 900;
}

.robot-score .description {
    color: #728098;
    font-size: 12px;
    line-height: 1.7;
}

.robot-note {
    background: #F0F4FF;
    border-left: 4px solid #386BFA;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    color: #43516D;
    line-height: 1.8;
    margin: 17px 0;
}

.robot-map-note {
    background: #FFF9E9;
    border: 1px solid #F3E7BF;
    border-radius: 12px;
    padding: 18px 20px;
    color: #675B40;
    line-height: 1.85;
}

.robot-footer {
    border-top: 1px solid #DFE5F1;
    margin-top: 65px;
    padding-top: 20px;
    color: #8490A4;
    font-size: 11px;
    letter-spacing: .12em;
}

@media(max-width:700px) {
    .robot-hero {
        padding: 29px 23px;
    }
    .robot-title {
        font-size: 36px;
    }
    .robot-section {
        margin-top: 36px;
    }
}
</style>
""")


def section(number, title, description):
    st.html(
        '<div class="robot-section">'
        f'<span class="num">{number}</span>'
        f'<span class="heading">{html.escape(title)}</span>'
        '</div>'
        f'<div class="robot-subtitle">{html.escape(description)}</div>'
    )


def score_card(title, score, ribbon, color, description):
    st.html(
        '<div class="robot-score">'
        f'<span class="ribbon" style="background:{color}">'
        f'{html.escape(ribbon)}</span>'
        f'<div class="label">{html.escape(title)}</div>'
        f'<div class="value">{score:.1%}</div>'
        f'<div class="description">{html.escape(description)}</div>'
        '</div>'
    )


# ============================================================
# 02. 데이터 불러오기
# ============================================================

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = ["id", "stroke"] + FEATURES
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "CSV 필수 열 누락: " + ", ".join(missing)
        )

    if df["id"].isna().any():
        raise ValueError("id 열에 빈 값이 있습니다.")

    if df["id"].duplicated().any():
        raise ValueError("id가 중복되어 있습니다.")

    if df["stroke"].isna().any():
        raise ValueError("stroke 열에 빈 값이 있습니다.")

    if not df["stroke"].isin([0, 1]).all():
        raise ValueError("stroke는 0 또는 1이어야 합니다.")

    return df


# ============================================================
# 03. 학습 / 채점 데이터 준비
# ============================================================

def split_by_id(df):
    ordered = (
        df.sort_values("id", kind="mergesort")
        .reset_index(drop=True)
    )

    # ID 순으로 10명씩 묶어서 앞 3명 채점
    positions = np.arange(len(ordered))
    test_mask = (positions % 10) < 3

    train = ordered.loc[~test_mask].copy()
    test = ordered.loc[test_mask].copy()

    return train, test


def prepare_features(train, test, columns):
    X_train = train[columns].copy()
    X_test = test[columns].copy()

    bmi_median = None

    if "bmi" in columns:
        # 훈련 데이터만으로 중앙값 계산
        bmi_median = X_train["bmi"].median()

        if pd.isna(bmi_median):
            raise ValueError(
                "훈련용 BMI 중앙값을 계산할 수 없습니다."
            )

        X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
        X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError(
            "선택된 속성에 처리되지 않은 빈 값이 있습니다."
        )

    return X_train, X_test, bmi_median


# ============================================================
# 04. 모델
# ============================================================

def make_logistic(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                max_iter=2000,
                random_state=SEED,
            ),
        ),
    ])

    model.fit(X, y)
    return model


def make_tree(X, y):
    model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=SEED,
    )

    model.fit(X, y)
    return model


def positive_probabilities(model, X):
    classes = model.named_steps["logistic"].classes_
    index = list(classes).index(1)
    return model.predict_proba(X)[:, index]


# ============================================================
# 05. 결정나무 질문 순서
# ============================================================

def tree_questions(model, features):
    tree = model.tree_
    lines = []

    def walk(node, depth, branch):
        indentation = "    " * depth

        n = int(tree.n_node_samples[node])

        if tree.feature[node] == _tree.TREE_UNDEFINED:
            label_index = int(np.argmax(tree.value[node][0]))
            prediction = int(model.classes_[label_index])

            result = (
                "뇌졸중"
                if prediction == 1
                else "뇌졸중 아님"
            )

            lines.append(
                f"{indentation}{branch} → "
                f"최종 답: {result} (학습 {n}명)"
            )
            return

        name = features[tree.feature[node]]
        threshold = float(tree.threshold[node])

        lines.append(
            f"{indentation}{branch} · "
            f"질문 {depth + 1}: "
            f"{LABELS[name]} ≤ {threshold:.2f}? "
            f"(학습 {n}명)"
        )

        walk(
            tree.children_left[node],
            depth + 1,
            "예",
        )

        walk(
            tree.children_right[node],
            depth + 1,
            "아니요",
        )

    walk(0, 0, "시작")
    return "\n".join(lines)


# ============================================================
# 06. 확률 모델의 0.5 경계 계산
# ============================================================

def calculate_boundary(model, xmin, xmax, ymin, ymax):
    scaler = model.named_steps["scaler"]
    logistic = model.named_steps["logistic"]

    coefficients = (
        logistic.coef_[0] / scaler.scale_
    )

    intercept = float(
        logistic.intercept_[0]
        - np.dot(coefficients, scaler.mean_)
    )

    a = float(coefficients[0])
    b = float(coefficients[1])

    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return None

    points = []

    if abs(b) > 1e-12:
        for x in (xmin, xmax):
            y = -(a * x + intercept) / b

            if ymin <= y <= ymax:
                points.append((float(x), float(y)))

    if abs(a) > 1e-12:
        for y in (ymin, ymax):
            x = -(b * y + intercept) / a

            if xmin <= x <= xmax:
                points.append((float(x), float(y)))

    unique = []

    for point in points:
        if not any(
            np.hypot(
                point[0] - old[0],
                point[1] - old[1],
            ) < 1e-6
            for old in unique
        ):
            unique.append(point)

    if len(unique) < 2:
        return None

    pairs = [
        (p, q)
        for i, p in enumerate(unique)
        for q in unique[i + 1:]
    ]

    return max(
        pairs,
        key=lambda pair: np.hypot(
            pair[0][0] - pair[1][0],
            pair[0][1] - pair[1][1],
        ),
    )


# ============================================================
# 07. Plotly 분류 지도
# ============================================================

def classification_map(
    X_train,
    y_train,
    X_test,
    y_test,
    test_ids,
    axes,
):
    xcol, ycol = axes

    train_two = X_train[axes]
    test_two = X_test[axes]

    # 그림 전용: 두 속성만으로 다시 학습
    logistic_two = make_logistic(train_two, y_train)
    tree_two = make_tree(train_two, y_train)

    all_x = pd.concat([
        train_two[xcol],
        test_two[xcol],
    ])

    all_y = pd.concat([
        train_two[ycol],
        test_two[ycol],
    ])

    xpad = max(float(all_x.max() - all_x.min()) * 0.06, 0.5)
    ypad = max(float(all_y.max() - all_y.min()) * 0.06, 0.5)

    xmin = float(all_x.min() - xpad)
    xmax = float(all_x.max() + xpad)
    ymin = float(all_y.min() - ypad)
    ymax = float(all_y.max() + ypad)

    gx = np.linspace(xmin, xmax, 150)
    gy = np.linspace(ymin, ymax, 150)

    xx, yy = np.meshgrid(gx, gy)

    grid = pd.DataFrame({
        xcol: xx.ravel(),
        ycol: yy.ravel(),
    })

    regions = tree_two.predict(grid).reshape(xx.shape)

    fig = go.Figure()

    # 결정나무 영역
    fig.add_trace(
        go.Heatmap(
            x=gx,
            y=gy,
            z=regions,
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "#DCE9FF"],
                [0.4999, "#DCE9FF"],
                [0.5, "#FFE1ED"],
                [1.0, "#FFE1ED"],
            ],
            opacity=0.68,
            showscale=False,
            hoverinfo="skip",
        )
    )

    # 로지스틱 회귀 0.5 경계
    boundary = calculate_boundary(
        logistic_two,
        xmin,
        xmax,
        ymin,
        ymax,
    )

    if boundary is not None:
        (x1, y1), (x2, y2) = boundary

        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                name="확률 모델: 0.5 경계",
                line=dict(
                    color=GREEN,
                    width=4,
                    dash="dash",
                ),
                hoverinfo="name",
            )
        )

    # 채점용 사람: 실제 정답별 점
    truth = y_test.to_numpy()

    for value, label, color in [
        (0, "실제: 뇌졸중 아님", BLUE),
        (1, "실제: 뇌졸중", RED),
    ]:
        mask = truth == value

        if not mask.any():
            continue

        fig.add_trace(
            go.Scatter(
                x=test_two.loc[mask, xcol],
                y=test_two.loc[mask, ycol],
                mode="markers",
                name=label,
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.85,
                    line=dict(color="white", width=1),
                ),
                customdata=test_ids.loc[mask].to_numpy(),
                hovertemplate=(
                    "ID %{customdata}<br>"
                    "가로: %{x:.2f}<br>"
                    "세로: %{y:.2f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=570,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=INK),
        margin=dict(l=20, r=20, t=55, b=30),
        legend=dict(
            orientation="h",
            x=0,
            y=1.04,
            yanchor="bottom",
        ),
        xaxis=dict(
            title=LABELS[xcol],
            range=[xmin, xmax],
            gridcolor="#EFF2F8",
        ),
        yaxis=dict(
            title=LABELS[ycol],
            range=[ymin, ymax],
            gridcolor="#EFF2F8",
        ),
    )

    return fig, boundary, tree_two


# ============================================================
# 08. 화면 시작
# ============================================================

st.html("""
<div class="robot-hero">
    <div class="robot-kicker">
        03 / MODEL ARENA · SPECTRUM FIVE
    </div>

    <div class="robot-title">
        데이터 분류 실험,<br>
        <span>모델 합체 작전!</span>
    </div>

    <div class="robot-description">
        같은 채점용 사람들을 놓고 세 가지 모델을 비교합니다.
        확률로 답하는 모델, 질문으로 답하는 모델,
        그리고 아무 속성도 보지 않는 기준 모델.
        다섯 색으로 분류 경계를 탐험해 보세요.
    </div>

    <div class="robot-colors">
        <span style="background:#F34E67"></span>
        <span style="background:#386BFA"></span>
        <span style="background:#19B58A"></span>
        <span style="background:#FFC83D"></span>
        <span style="background:#E968C6"></span>
    </div>
</div>
""")


try:
    df = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다.")
    with st.expander("오류 상세"):
        st.code(str(exc))
    st.stop()


# ============================================================
# 09. 속성 선택
# ============================================================

section(
    "01",
    "모델은 무엇을 보고 판단할까?",
    "다섯 속성 가운데 사용할 입력을 선택하세요.",
)

selected = st.multiselect(
    "모델의 입력 속성",
    FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda name: f"{LABELS[name]} ({name})",
)

if len(selected) < 2:
    st.warning(
        "최소 두 개의 속성을 선택해 주세요. "
        "현재는 모델 학습을 진행하지 않습니다."
    )
    st.stop()


# ============================================================
# 10. 고정 분할
# ============================================================

train_df, test_df = split_by_id(df)

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

if y_train.nunique() < 2:
    st.error("훈련 데이터에 0과 1이 모두 있어야 합니다.")
    st.stop()


try:
    X_train, X_test, bmi_median = prepare_features(
        train_df,
        test_df,
        selected,
    )

    logistic_model = make_logistic(X_train, y_train)
    tree_model = make_tree(X_train, y_train)

except ValueError as exc:
    st.error(str(exc))
    st.stop()


# ============================================================
# 11. 모델 채점
# ============================================================

probabilities = positive_probabilities(
    logistic_model,
    X_test,
)

logistic_answers = (probabilities >= 0.5).astype(int)
tree_answers = tree_model.predict(X_test).astype(int)

majority_class = int(y_train.value_counts().idxmax())

baseline_answers = np.full(
    len(y_test),
    majority_class,
    dtype=int,
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_answers,
)

tree_accuracy = accuracy_score(
    y_test,
    tree_answers,
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_answers,
)


with st.sidebar:
    st.divider()
    st.markdown("**MODEL ARENA / 실험 규칙**")
    st.caption("양성 = 뇌졸중(1)")
    st.caption("ID 오름차순으로 정렬")
    st.caption("매 10명 중 앞 3명: 채점")
    st.caption("나머지 7명: 훈련")
    st.caption("크기 맞추기: 훈련 데이터만 사용")
    st.caption("결정나무: 질문 최대 3단계")
    st.caption("결정나무: 잎 최소 5명")
    st.caption("난수: 42")


section(
    "02",
    "같은 시험, 세 가지 답변 전략",
    "모든 정확도는 동일한 채점용 사람들을 대상으로 계산했습니다.",
)

c1, c2, c3 = st.columns(3)

with c1:
    score_card(
        "확률로 답하는 모델",
        logistic_accuracy,
        "RED / 확률",
        RED,
        "로지스틱 회귀 · 확률 0.5 기준",
    )

with c2:
    score_card(
        "질문 순서로 답하는 모델",
        tree_accuracy,
        "BLUE / 질문",
        BLUE,
        "결정나무 · 최대 질문 3단계",
    )

with c3:
    score_card(
        "많은 쪽만 답하는 모델",
        baseline_accuracy,
        "GREEN / 기준",
        GREEN,
        "훈련용 다수 클래스만 예측",
    )


st.html(
    '<div class="robot-note">'
    f'<b>훈련용 {len(train_df):,}명 · '
    f'채점용 {len(test_df):,}명</b><br>'
    f'기준 모델의 대답은 항상 '
    f'<b>{"뇌졸중" if majority_class == 1 else "뇌졸중 아님"}</b>입니다.'
    '</div>'
)

if bmi_median is not None:
    st.info(
        f"BMI는 훈련용 중앙값 {bmi_median:.2f}로 "
        "훈련용과 채점용의 빈 값을 채웠습니다. "
        "채점용 정보로 중앙값을 계산하지 않았습니다."
    )
else:
    st.caption(
        "BMI가 선택되지 않았으므로 BMI 결측치를 처리하지 않았습니다."
    )


# ============================================================
# 12. 축 선택과 Plotly 지도
# ============================================================

section(
    "03",
    "두 속성의 분류 지도",
    "채점용 사람은 실제 정답으로 색을 구분합니다. "
    "경계와 배경은 두 속성만으로 새로 학습한 모델입니다.",
)

axis_a, axis_b = st.columns(2)

with axis_a:
    x_axis = st.selectbox(
        "가로축",
        selected,
        format_func=lambda x: LABELS[x],
    )

with axis_b:
    y_options = [col for col in selected if col != x_axis]

    y_axis = st.selectbox(
        "세로축",
        y_options,
        format_func=lambda x: LABELS[x],
    )

fig, boundary, map_tree = classification_map(
    X_train,
    y_train,
    X_test,
    y_test,
    test_df["id"],
    [x_axis, y_axis],
)

st.plotly_chart(fig, use_container_width=True)

st.html("""
<div class="robot-map-note">
    <b>그래프 읽는 법</b><br>
    파란 점 = 실제 뇌졸중 아님 ·
    빨간 점 = 실제 뇌졸중<br>
    옅은 파랑 = 질문 모델의 음성 영역 ·
    옅은 분홍 = 질문 모델의 양성 영역<br>
    초록 점선 = 확률 모델의 0.5 경계
</div>
""")

if boundary is None:
    st.warning(
        "확률 모델의 0.5 경계선이 현재 그림 범위 밖에 있거나 "
        "두 축으로 구분할 수 있는 경계가 만들어지지 않았습니다."
    )

st.caption(
    "위 정확도 카드는 선택한 모든 속성을 사용하지만, "
    "그림은 선택한 두 축만으로 모델을 다시 학습합니다."
)


# ============================================================
# 13. 결정나무 질문 순서
# ============================================================

section(
    "04",
    "질문 모델의 판단 순서",
    "각 질문에서 예 또는 아니요를 따라가면 최종 대답에 도달합니다.",
)

st.markdown("**전체 선택 속성으로 학습한 질문 모델**")

st.code(
    tree_questions(tree_model, selected),
    language="text",
)

with st.expander("두 축만으로 학습한 질문 모델도 보기"):
    st.code(
        tree_questions(map_tree, [x_axis, y_axis]),
        language="text",
    )


# ============================================================
# 14. 사람별 채점 결과
# ============================================================

section(
    "05",
    "사람별 채점 결과",
    "실제 답과 모델별 답을 비교해 보세요.",
)

results = pd.DataFrame({
    "ID": test_df["id"].to_numpy(),
    "실제 정답": y_test.to_numpy(),
    "양성 확률": probabilities,
    "확률 모델 답": logistic_answers,
    "질문 모델 답": tree_answers,
    "기준 모델 답": baseline_answers,
})

st.dataframe(
    results,
    hide_index=True,
    use_container_width=True,
    height=340,
    column_config={
        "양성 확률": st.column_config.NumberColumn(
            format="%.1%",
        ),
    },
)

st.download_button(
    "채점 결과 CSV 다운로드",
    data=results.to_csv(index=False).encode("utf-8-sig"),
    file_name="neuro_classification_results.csv",
    mime="text/csv",
)


# ============================================================
# 15. 마무리
# ============================================================

st.info(
    "이 실습은 모델의 분류 방식을 이해하기 위한 교육용입니다. "
    "정확도가 높아도 실제 환자에 대한 의료 진단에 사용할 수 없습니다."
)

st.html("""
<div class="robot-footer">
    NEURO / LAB · MODEL ARENA · EDUCATIONAL USE ONLY
</div>
""")
