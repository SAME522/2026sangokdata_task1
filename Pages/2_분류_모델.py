
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree


# =========================================================
# 01. 기본 설정
# =========================================================

# 페이지 설정과 메뉴는 main.py에서 관리합니다.

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

FEATURE_LABELS = {
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


# =========================================================
# 02. 합체로봇 스타일 디자인
# =========================================================

st.markdown("""
<style>
.model-hero {
    position: relative;
    background: linear-gradient(
        114deg,
        #F5F7FF 0%,
        #EDF9FF 45%,
        #FFF0F8 100%
    );
    border: 1px solid #DBE4FF;
    border-radius: 22px;
    padding: 39px 42px;
    overflow: hidden;
    margin: 14px 0 28px;
}

.model-hero:after {
    content: "✦";
    position: absolute;
    right: 35px;
    top: -42px;
    font-size: 210px;
    color: #E968C624;
    transform: rotate(-12deg);
    pointer-events: none;
}

.model-kicker {
    font-size: 12px;
    color: #386BFA;
    font-weight: 900;
    letter-spacing: .16em;
}

.model-title {
    color: #202943;
    font-size: clamp(34px, 5vw, 62px);
    font-weight: 950;
    letter-spacing: -.075em;
    line-height: 1.16;
    margin: 16px 0 15px;
}

.model-title span {
    background: linear-gradient(
        95deg,
        #F34E67 0%,
        #E968C6 48%,
        #386BFA 100%
    );
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.model-lead {
    color: #4A5A77;
    font-size: 15px;
    line-height: 1.85;
    max-width: 700px;
}

.color-bars {
    display: flex;
    gap: 6px;
    margin-top: 27px;
    max-width: 260px;
}

.color-bars i {
    height: 7px;
    border-radius: 20px;
    flex: 1;
}

.model-step {
    margin: 43px 0 19px;
    padding-top: 19px;
    border-top: 1px solid #DAE2F2;
    display: flex;
    align-items: baseline;
    gap: 16px;
}

.model-step b {
    color: #386BFA;
    font-size: 12px;
}

.model-step span {
    font-size: 25px;
    color: #202943;
    letter-spacing: -.045em;
    font-weight: 850;
}

.model-intro {
    color: #68758D;
    font-size: 14px;
    margin: -8px 0 22px;
    line-height: 1.8;
}

.score-card {
    border: 1px solid #DFE5F1;
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 28px #3850740D;
    min-height: 160px;
}

.score-card .tag {
    display: inline-block;
    padding: 5px 9px;
    font-size: 11px;
    font-weight: 850;
    border-radius: 5px;
    color: white;
    margin-bottom: 13px;
}

.score-card .label {
    font-size: 13px;
    color: #4A5871;
    font-weight: 700;
}

.score-card .number {
    font-size: clamp(30px, 4vw, 48px);
    font-weight: 900;
    letter-spacing: -.06em;
    line-height: 1.3;
    color: #202943;
}

.score-card .caption {
    font-size: 12px;
    color: #70809A;
    margin-top: 4px;
}

.model-note {
    border-left: 4px solid #386BFA;
    background: #F0F4FF;
    border-radius: 0 10px 10px 0;
    padding: 15px 19px;
    color: #43516D;
    line-height: 1.8;
}

.legend-note {
    background: #FFF9E8;
    border: 1px solid #F5E7B6;
    border-radius: 11px;
    padding: 15px 18px;
    color: #61573F;
    line-height: 1.8;
    font-size: 13px;
}

@media(max-width: 700px) {
    .model-hero {
        padding: 28px 24px;
    }

    .model-step {
        margin-top: 32px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 03. 데이터 읽기
# =========================================================

@st.cache_data(ttl=3600)
def load_dataset():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8",
    )

    required = [
        "id",
        "stroke",
    ] + FEATURES

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "필수 열 누락: "
            + ", ".join(missing)
        )

    if (
        not df["stroke"]
        .dropna()
        .isin([0, 1])
        .all()
        or df["stroke"].isna().any()
    ):
        raise ValueError(
            "stroke 열에는 0 또는 1만 있어야 합니다."
        )

    if df["id"].isna().any():
        raise ValueError(
            "id 열에 빈 값이 있어 정렬할 수 없습니다."
        )

    return df


# =========================================================
# 04. 화면 함수
# =========================================================

def section(number, title, subtitle):

    st.markdown(
        f"""
        <div class="model-step">
            <b>{number}</b>
            <span>{title}</span>
        </div>

        <div class="model-intro">
            {subtitle}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(
    title,
    value,
    ribbon,
    color,
    explanation,
):

    st.markdown(
        f"""
        <div class="score-card">
            <span class="tag"
                  style="background:{color}">
                {ribbon}
            </span>

            <div class="label">
                {title}
            </div>

            <div class="number">
                {value:.1%}
            </div>

            <div class="caption">
                {explanation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 05. BMI 결측치 처리
# 훈련용 중앙값만 사용
# =========================================================

def prepare_xy(
    train_df,
    test_df,
    cols,
):

    X_train = train_df[cols].copy()
    X_test = test_df[cols].copy()

    bmi_median = None

    # BMI를 선택했을 때만 결측치 처리
    if "bmi" in cols:

        bmi_median = X_train[
            "bmi"
        ].median()

        if pd.isna(bmi_median):
            raise ValueError(
                "훈련용 BMI가 모두 비어 있어 "
                "중앙값을 계산할 수 없습니다."
            )

        X_train["bmi"] = (
            X_train["bmi"]
            .fillna(bmi_median)
        )

        X_test["bmi"] = (
            X_test["bmi"]
            .fillna(bmi_median)
        )

    if (
        X_train.isna().any().any()
        or X_test.isna().any().any()
    ):
        raise ValueError(
            "선택된 BMI 이외의 속성에 빈 값이 있습니다."
        )

    return (
        X_train,
        X_test,
        bmi_median,
    )


# =========================================================
# 06. 확률 모델: 로지스틱 회귀
# 크기 맞추기는 훈련용으로만
# =========================================================

def train_probability(
    X_train,
    y_train,
):

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=SEED,
            ),
        ),
    ])

    model.fit(
        X_train,
        y_train,
    )

    return model


# =========================================================
# 07. 질문 모델: 결정나무
# 최대 깊이 3 / 잎 최소 5명
# =========================================================

def train_questions(
    X_train,
    y_train,
):

    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=SEED,
    )

    tree.fit(
        X_train,
        y_train,
    )

    return tree


# =========================================================
# 08. 질문 순서를 글로 만들기
# =========================================================

def describe_tree(
    model,
    columns,
):

    t = model.tree_

    lines = []

    def visit(
        node,
        depth,
        answer,
    ):

        indent = "    " * depth

        n = int(
            t.n_node_samples[node]
        )

        # 마지막 마디
        if (
            t.feature[node]
            == _tree.TREE_UNDEFINED
        ):

            predicted = int(
                model.classes_[
                    np.argmax(
                        t.value[node][0]
                    )
                ]
            )

            result = (
                "뇌졸중"
                if predicted == 1
                else "뇌졸중 아님"
            )

            lines.append(
                f"{indent}{answer} "
                f"→ [최종 답] {result} "
                f"· 학습 {n}명"
            )

            return

        feature = columns[
            t.feature[node]
        ]

        condition = (
            f"{FEATURE_LABELS[feature]} "
            f"≤ {t.threshold[node]:.2f}?"
        )

        lines.append(
            f"{indent}{answer} "
            f"질문 {depth + 1}: "
            f"{condition} "
            f"· 학습 {n}명"
        )

        # 예 = 왼쪽 가지
        visit(
            t.children_left[node],
            depth + 1,
            "예",
        )

        # 아니요 = 오른쪽 가지
        visit(
            t.children_right[node],
            depth + 1,
            "아니요",
        )

    visit(
        0,
        0,
        "시작",
    )

    return "\n".join(lines)


# =========================================================
# 09. 확률 0.5 경계선 계산
# 로지스틱 모델의 계수를 원래 축 단위로 변환
# =========================================================

def boundary_segment(
    logit,
    xmin,
    xmax,
    ymin,
    ymax,
):

    scaler = logit.named_steps[
        "scaler"
    ]

    clf = logit.named_steps[
        "classifier"
    ]

    coef = (
        clf.coef_[0]
        / scaler.scale_
    )

    intercept = float(
        clf.intercept_[0]
        - (
            coef
            * scaler.mean_
        ).sum()
    )

    a, b = (
        float(coef[0]),
        float(coef[1]),
    )

    eps = 1e-11

    if (
        abs(a) < eps
        and abs(b) < eps
    ):

        return (
            None,
            "두 축에 따른 확률 변화가 거의 없어 "
            "0.5 경계선이 만들어지지 않습니다.",
        )

    points = []

    # 왼쪽 / 오른쪽 테두리와 교차
    if abs(b) > eps:

        for x in (
            xmin,
            xmax,
        ):

            y = (
                -(a * x + intercept)
                / b
            )

            if (
                ymin - 1e-8
                <= y
                <= ymax + 1e-8
            ):

                points.append(
                    (
                        float(x),
                        float(y),
                    )
                )

    # 아래 / 위 테두리와 교차
    if abs(a) > eps:

        for y in (
            ymin,
            ymax,
        ):

            x = (
                -(b * y + intercept)
                / a
            )

            if (
                xmin - 1e-8
                <= x
                <= xmax + 1e-8
            ):

                points.append(
                    (
                        float(x),
                        float(y),
                    )
                )

    # 중복 교점 제거
    unique = []

    for point in points:

        duplicated = any(
            np.hypot(
                point[0] - p[0],
                point[1] - p[1],
            ) < 1e-6
            for p in unique
        )

        if not duplicated:
            unique.append(point)

    # 그림 내부에 경계선이 없는 경우
    if len(unique) < 2:

        return (
            None,
            "확률 모델의 0.5 경계선이 "
            "현재 그림 범위 밖에 있습니다.",
        )

    # 가장 멀리 떨어진 두 교점
    pairs = [
        (p, q)
        for i, p in enumerate(unique)
        for q in unique[i + 1:]
    ]

    best = max(
        pairs,
        key=lambda pair: np.hypot(
            pair[0][0] - pair[1][0],
            pair[0][1] - pair[1][1],
        ),
    )

    return best, None


# =========================================================
# 10. Plotly 분류 지도
# 선택된 두 축만으로 두 모델을 다시 훈련
# =========================================================

def model_figure(
    X_train,
    y_train,
    X_test,
    test_labels,
    test_ids,
    axes,
):

    xname, yname = axes

    two_train = X_train[
        axes
    ]

    two_test = X_test[
        axes
    ]

    # 두 축으로만 다시 학습
    line_model = train_probability(
        two_train,
        y_train,
    )

    area_model = train_questions(
        two_train,
        y_train,
    )

    # 그림 범위
    all_x = pd.concat([
        two_train[xname],
        two_test[xname],
    ])

    all_y = pd.concat([
        two_train[yname],
        two_test[yname],
    ])

    dx = max(
        float(
            all_x.max()
            - all_x.min()
        ) * .06,
        .6,
    )

    dy = max(
        float(
            all_y.max()
            - all_y.min()
        ) * .06,
        .6,
    )

    xmin = float(
        all_x.min() - dx
    )

    xmax = float(
        all_x.max() + dx
    )

    ymin = float(
        all_y.min() - dy
    )

    ymax = float(
        all_y.max() + dy
    )

    # 결정나무 영역을 위한 격자
    xx = np.linspace(
        xmin,
        xmax,
        140,
    )

    yy = np.linspace(
        ymin,
        ymax,
        140,
    )

    gx, gy = np.meshgrid(
        xx,
        yy,
    )

    grid = pd.DataFrame({
        xname: gx.ravel(),
        yname: gy.ravel(),
    })

    region = (
        area_model
        .predict(grid)
        .reshape(gx.shape)
    )

    fig = go.Figure()

    # -----------------------------------------------------
    # 결정나무 예측 영역
    # -----------------------------------------------------

    fig.add_trace(
        go.Heatmap(
            x=xx,
            y=yy,
            z=region,
            zmin=0,
            zmax=1,
            colorscale=[
                [
                    0,
                    "#DDEAFF",
                ],
                [
                    .4999,
                    "#DDEAFF",
                ],
                [
                    .5,
                    "#FFE0EA",
                ],
                [
                    1,
                    "#FFE0EA",
                ],
            ],
            opacity=.72,
            showscale=False,
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # -----------------------------------------------------
    # 확률 모델의 0.5 경계선
    # -----------------------------------------------------

    segment, message = boundary_segment(
        line_model,
        xmin,
        xmax,
        ymin,
        ymax,
    )

    if segment is not None:

        (x1, y1), (x2, y2) = segment

        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(
                    color=GREEN,
                    width=4,
                    dash="dash",
                ),
                name="확률 모델 · 0.5 경계",
                hoverinfo="name",
            )
        )

    # -----------------------------------------------------
    # 채점용 사람 점
    # 실제 정답으로 색 구분
    # -----------------------------------------------------

    for label, name, color in [
        (
            0,
            "실제: 뇌졸중 아님",
            BLUE,
        ),
        (
            1,
            "실제: 뇌졸중",
            RED,
        ),
    ]:

        mask = (
            test_labels.to_numpy()
            == label
        )

        if mask.any():

            fig.add_trace(
                go.Scatter(
                    x=two_test.loc[
                        mask,
                        xname,
                    ],
                    y=two_test.loc[
                        mask,
                        yname,
                    ],
                    mode="markers",
                    name=name,
                    marker=dict(
                        size=8,
                        color=color,
                        opacity=.83,
                        line=dict(
                            color="white",
                            width=1.2,
                        ),
                    ),
                    customdata=(
                        test_ids.loc[mask]
                        .to_numpy()
                        .reshape(-1, 1)
                    ),
                    hovertemplate=(
                        "ID %{customdata[0]}<br>"
                        "가로 %{x:.2f}<br>"
                        "세로 %{y:.2f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

    # -----------------------------------------------------
    # Plotly 스타일
    # -----------------------------------------------------

    fig.update_layout(
        height=565,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(
            l=20,
            r=25,
            t=18,
            b=30,
        ),
        xaxis=dict(
            title=FEATURE_LABELS[xname],
            range=[
                xmin,
                xmax,
            ],
            gridcolor="#EEF1F8",
        ),
        yaxis=dict(
            title=FEATURE_LABELS[yname],
            range=[
                ymin,
                ymax,
            ],
            gridcolor="#EEF1F8",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(
                size=12,
            ),
        ),
        font=dict(
            color=INK,
        ),
    )

    return (
        fig,
        message,
        area_model,
    )


# =========================================================
# 11. 상단 디자인
# =========================================================

st.markdown("""
<div class="model-hero">
    <div class="model-kicker">
        03 / MODEL ARENA · SPECTRUM FIVE
    </div>

    <div class="model-title">
        데이터 분류 실험,<br>
        <span>모델 합체 작전!</span>
    </div>

    <div class="model-lead">
        같은 채점용 사람들을 놓고
        세 가지 답변 전략을 겨룹니다.
        다섯 색으로 실제 정답과 모델의 판단을
        구분하며 분류 경계를 탐험해 보세요.
    </div>

    <div class="color-bars">
        <i style="background:#F34E67"></i>
        <i style="background:#386BFA"></i>
        <i style="background:#19B58A"></i>
        <i style="background:#FFC83D"></i>
        <i style="background:#E968C6"></i>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 12. 데이터 불러오기
# =========================================================

try:
    original = load_dataset()

except Exception as exc:

    st.error(
        "데이터를 불러오지 못했습니다."
    )

    with st.expander(
        "오류 상세"
    ):
        st.code(str(exc))

    st.stop()


# =========================================================
# 13. 입력 속성 선택
# =========================================================

section(
    "01",
    "어떤 정보를 보고 답할까?",
    "다섯 속성 중 사용할 속성을 선택하세요. "
    "처음에는 BMI를 제외한 네 속성이 선택됩니다.",
)

selected = st.multiselect(
    "모델에 입력할 속성",
    FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: (
        f"{FEATURE_LABELS[c]}  ({c})"
    ),
    key="model_features",
)

# 최소 두 개 선택
if len(selected) < 2:

    st.warning(
        "속성을 최소 두 개 선택해 주세요. "
        "그래프의 가로축·세로축에도 "
        "두 속성이 필요합니다."
    )

    st.stop()


# =========================================================
# 14. 번호순 정렬 및 고정 분할
# =========================================================

# ID 오름차순 정렬
ordered = (
    original
    .sort_values(
        "id",
        kind="mergesort",
    )
    .reset_index(drop=True)
)

# 열 명 중 앞 세 명이 채점용
position = np.arange(
    len(ordered)
)

test_mask = (
    position % 10
) < 3

train_df = ordered.loc[
    ~test_mask
].copy()

test_df = ordered.loc[
    test_mask
].copy()

y_train = (
    train_df["stroke"]
    .astype(int)
)

y_test = (
    test_df["stroke"]
    .astype(int)
)

if y_train.nunique() < 2:

    st.error(
        "훈련 데이터에 정답 0과 1이 "
        "모두 있어야 확률 모델을 학습할 수 있습니다."
    )

    st.stop()


# =========================================================
# 15. 결측치 처리와 모델 학습
# =========================================================

try:

    X_train, X_test, bmi_median = prepare_xy(
        train_df,
        test_df,
        selected,
    )

    probability_model = train_probability(
        X_train,
        y_train,
    )

    question_model = train_questions(
        X_train,
        y_train,
    )

except ValueError as exc:

    st.error(str(exc))
    st.stop()


# =========================================================
# 16. 모델별 채점
# =========================================================

# 뇌졸중(1) 양성 확률
positive_probability = (
    probability_model
    .predict_proba(X_test)[
        :,
        list(
            probability_model.classes_
        ).index(1)
    ]
)

# 확률 0.5 이상이면 1
prob_prediction = (
    positive_probability >= .5
).astype(int)

# 질문 모델
tree_prediction = (
    question_model
    .predict(X_test)
    .astype(int)
)

# 입력을 전혀 보지 않는 다수 클래스 모델
majority = int(
    y_train.value_counts().idxmax()
)

baseline_prediction = np.full(
    len(y_test),
    majority,
    dtype=int,
)

# 세 모델 정확도
prob_acc = accuracy_score(
    y_test,
    prob_prediction,
)

tree_acc = accuracy_score(
    y_test,
    tree_prediction,
)

baseline_acc = accuracy_score(
    y_test,
    baseline_prediction,
)


# =========================================================
# 17. 사이드바 실험 규칙
# =========================================================

with st.sidebar:

    st.divider()

    st.markdown(
        "**MODEL ARENA / 실험 규칙**"
    )

    st.caption(
        "정답 1 = 뇌졸중(양성) · "
        "정답 0 = 뇌졸중 아님(음성)"
    )

    st.caption(
        "정렬: ID 오름차순"
    )

    st.caption(
        "각 10명 중 1·2·3번째: 채점 / 나머지: 훈련"
    )

    st.caption(
        "로지스틱 회귀: 훈련용으로만 표준화"
    )

    st.caption(
        "결정나무: 최대 질문 3단계, 잎 최소 5명"
    )

    st.caption(
        "난수 고정: 42"
    )


# =========================================================
# 18. 모델 정확도 카드
# =========================================================

section(
    "02",
    "같은 시험, 세 가지 전략",
    "세 카드 모두 같은 채점용 데이터에서의 정확도입니다. "
    "높다고 해서 의료용 진단 모델이 되는 것은 아닙니다.",
)

a, b, c = st.columns(
    3,
    gap="medium",
)

with a:

    card(
        "확률로 답하는 모델",
        prob_acc,
        "RED · 확률",
        RED,
        "로지스틱 회귀 · 0.5 이상이면 양성",
    )

with b:

    card(
        "질문 순서로 답하는 모델",
        tree_acc,
        "BLUE · 질문",
        BLUE,
        "결정나무 · 최대 깊이 3",
    )

with c:

    card(
        "많은 쪽만 답하는 모델",
        baseline_acc,
        "GREEN · 기준",
        GREEN,
        "입력 속성을 보지 않는 훈련용 다수 클래스",
    )


# =========================================================
# 19. 훈련/채점 정보 표시
# =========================================================

majority_text = (
    "뇌졸중(1)"
    if majority == 1
    else "뇌졸중 아님(0)"
)

st.markdown(
    f"""
    <div class="model-note">
        <b>
            훈련용 {len(train_df):,}명
            · 채점용 {len(test_df):,}명
        </b>
        <br>
        훈련용에서 더 많았던 정답은
        <b>{majority_text}</b>입니다.
        세 모델은 같은 채점용 인원을 맞혀 봅니다.
    </div>
    """,
    unsafe_allow_html=True,
)

if "bmi" in selected:

    st.info(
        f"BMI 선택됨 · "
        f"훈련용 BMI 중앙값 {bmi_median:.2f}로 "
        "훈련용·채점용 빈 값만 채웠습니다. "
        "채점용 BMI로 중앙값을 계산하지 않았습니다."
    )

else:

    st.caption(
        "BMI를 선택하지 않았으므로 "
        "BMI 빈 값을 채우지 않았습니다."
    )

st.caption(
    "정확도만으로는 드문 양성(뇌졸중) 사례를 "
    "얼마나 잘 맞히는지 판단하기 어렵습니다. "
    "기준 모델과 함께 비교해 주세요."
)


# =========================================================
# 20. 가로축 / 세로축 선택
# =========================================================

section(
    "03",
    "두 속성의 분류 지도",
    "선택한 속성 중 가로·세로축을 고릅니다. "
    "배경은 두 축만으로 새로 학습한 결정나무, "
    "초록 점선은 두 축만으로 새로 학습한 "
    "확률 모델의 0.5 경계입니다.",
)

axis1, axis2 = st.columns(2)

with axis1:

    xaxis = st.selectbox(
        "가로축",
        selected,
        index=0,
        format_func=lambda c: (
            FEATURE_LABELS[c]
        ),
        key="model_xaxis",
    )

with axis2:

    y_choices = [
        c for c in selected
        if c != xaxis
    ]

    yaxis = st.selectbox(
        "세로축",
        y_choices,
        index=0,
        format_func=lambda c: (
            FEATURE_LABELS[c]
        ),
        key="model_yaxis",
    )


# =========================================================
# 21. Plotly 지도 생성
# =========================================================

fig, boundary_note, two_axis_tree = model_figure(
    X_train,
    y_train,
    X_test,
    y_test,
    test_df["id"],
    [
        xaxis,
        yaxis,
    ],
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.markdown("""
<div class="legend-note">
    <b>읽는 법</b> ·
    옅은 파랑 배경 = 질문 모델의 음성 구역,
    옅은 분홍 배경 = 질문 모델의 양성 구역.
    진한 파란 점과 빨간 점은
    채점용 사람의 <b>실제</b> 정답입니다.
    초록 점선은 확률 0.5 경계입니다.
</div>
""", unsafe_allow_html=True)

if boundary_note:
    st.warning(boundary_note)

st.caption(
    "전체 속성으로 학습한 모델의 정확도 카드와 달리, "
    "지도 위 경계·배경은 고른 두 축만으로 "
    "각각 새로 학습한 모델입니다."
)


# =========================================================
# 22. 질문 모델의 질문 순서
# =========================================================

section(
    "04",
    "질문 모델은 어떤 순서로 묻나?",
    "질문은 나무의 가지마다 달라집니다. "
    "예/아니요를 따라가며 최대 세 번 질문합니다.",
)

st.markdown(
    "**전체 선택 속성으로 훈련한 질문 나무**"
)

st.code(
    describe_tree(
        question_model,
        selected,
    ),
    language="text",
)

with st.expander(
    "지도에 사용된 두 축의 질문 순서 보기"
):

    st.code(
        describe_tree(
            two_axis_tree,
            [
                xaxis,
                yaxis,
            ],
        ),
        language="text",
    )


# =========================================================
# 23. 사람별 채점 결과
# =========================================================

section(
    "05",
    "사람별 채점 결과",
    "실제 정답과 각 모델의 답, "
    "확률 모델의 양성 확률을 확인합니다.",
)

results = pd.DataFrame({
    "ID": test_df["id"].to_numpy(),
    "실제 정답 (1=뇌졸중)": y_test.to_numpy(),
    "확률 모델 양성 확률": positive_probability,
    "확률 모델 답": prob_prediction,
    "질문 모델 답": tree_prediction,
    "다수 클래스 모델 답": baseline_prediction,
})

st.dataframe(
    results,
    hide_index=True,
    use_container_width=True,
    column_config={
        "확률 모델 양성 확률": (
            st.column_config.NumberColumn(
                format="%.1%",
            )
        ),
    },
    height=320,
)

st.download_button(
    "채점 결과 CSV 저장 ↗",
    results.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="neuro_classification_results.csv",
    mime="text/csv",
)

st.markdown("""
<div class="footer">
    NEURO / LAB · FIVE COLORS, THREE STRATEGIES
    · EDUCATIONAL USE ONLY
</div>
""", unsafe_allow_html=True)
