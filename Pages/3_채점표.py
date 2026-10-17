import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# NEURO FORCE / 04. 채점표
# main.py가 st.navigation()과 페이지 설정을 담당합니다.
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/main/data/stroke.csv"
)

FEATURES = [
    "age", "avg_glucose_level", "bmi",
    "hypertension", "heart_disease",
]
DEFAULT_FEATURES = [
    "age", "avg_glucose_level",
    "hypertension", "heart_disease",
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


# ============================================================
# 공통 전대 컬러를 이어받되, 이 페이지의 디자인은 독립 적용
# HTML은 Markdown 코드 블록으로 보이지 않도록 st.html() 사용
# ============================================================

st.html("""
<style>
.grade-hero {
    position: relative;
    overflow: hidden;
    padding: 39px 42px;
    border: 1px solid #DBE4FC;
    border-radius: 22px;
    background: linear-gradient(115deg, #F1F5FF, #F0FBF7 51%, #FFF0F8);
    margin: 12px 0 28px;
}
.grade-hero:after {
    content: "✦";
    position: absolute;
    font-size: 230px;
    top: -74px;
    right: 35px;
    color: #E968C620;
    pointer-events: none;
}
.grade-kicker {
    color: #386BFA;
    font-weight: 900;
    letter-spacing: .16em;
    font-size: 12px;
    position: relative;
    z-index: 1;
}
.grade-title {
    color: #202943;
    font-size: clamp(34px, 5vw, 61px);
    font-weight: 900;
    letter-spacing: -.065em;
    line-height: 1.15;
    margin: 17px 0;
    position: relative;
    z-index: 1;
}
.grade-title span {
    background: linear-gradient(95deg, #F34E67, #E968C6, #386BFA);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.grade-lead {
    max-width: 730px;
    color: #53617C;
    line-height: 1.9;
    font-size: 15px;
    position: relative;
    z-index: 1;
}
.grade-bars { display: flex; gap: 6px; max-width: 290px; margin-top: 25px; }
.grade-bars i { display: block; flex: 1; height: 8px; border-radius: 99px; }
.grade-section {
    display: flex; align-items: baseline; gap: 17px;
    border-top: 1px solid #DCE4F2;
    padding-top: 20px; margin: 49px 0 9px;
}
.grade-section b { color: #386BFA; font-size: 12px; }
.grade-section span {
    color: #202943; font-size: clamp(23px, 3vw, 31px);
    font-weight: 850; letter-spacing: -.045em;
}
.grade-desc { margin-bottom: 23px; color: #65738B; line-height: 1.8; font-size: 14px; }
.grade-meta {
    border: 1px solid #DBE4F5; border-left: 4px solid #386BFA;
    border-radius: 0 12px 12px 0; background: #F0F4FF;
    padding: 18px 20px; line-height: 1.85; color: #3E4D69;
    margin: 17px 0;
}
.grade-keyline { display:flex; gap:8px; flex-wrap:wrap; margin: 8px 0 16px; }
.grade-keyline span {
    border-radius: 999px; padding: 7px 12px;
    border: 1px solid #DFE6F4; background: white;
    color: #495872; font-size: 12px; font-weight: 750;
}
.grade-badge {
    border-radius: 6px; font-size: 11px; font-weight: 900;
    color: white; padding: 5px 10px; display: inline-block;
}
.grade-hint {
    border: 1px solid #F1E3B9; background: #FFF9E9;
    border-radius: 12px; padding: 16px 19px;
    color: #5C523F; line-height: 1.8; font-size: 13px;
}
.grade-footer {
    border-top: 1px solid #DFE6F2; margin-top: 60px;
    padding-top: 20px; color: #8590A6; font-size: 11px; letter-spacing: .1em;
}
@media (max-width: 700px) {
    .grade-hero { padding: 27px 22px; }
    .grade-title { font-size: 37px; }
    .grade-section { margin-top: 35px; }
}
</style>
""")


def section(number, title, description):
    st.html(
        '<div class="grade-section">'
        f'<b>{html.escape(number)}</b>'
        f'<span>{html.escape(title)}</span>'
        '</div>'
        f'<div class="grade-desc">{html.escape(description)}</div>'
    )


def note(text):
    st.html(f'<div class="grade-meta">{html.escape(text)}</div>')


# ============================================================
# 01. 분류 모델 페이지와 정확히 같은 데이터 분할과 학습
# ============================================================

@st.cache_data(ttl=3600)
def read_data():
    frame = pd.read_csv(DATA_URL, encoding="utf-8")
    required = ["id", "stroke"] + FEATURES
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError("CSV 필수 열 누락: " + ", ".join(missing))
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("ID에 빈 값 또는 중복이 있습니다.")
    if frame["stroke"].isna().any() or not frame["stroke"].isin([0, 1]).all():
        raise ValueError("stroke 열은 0과 1로만 구성되어야 합니다.")
    return frame


def split_by_id(data):
    sorted_data = data.sort_values("id", kind="mergesort").reset_index(drop=True)
    position = np.arange(len(sorted_data))
    # 정렬 순서의 10명 중 1, 2, 3번째를 채점용으로 고정
    is_test = (position % 10) < 3
    return (
        sorted_data.loc[~is_test].copy().reset_index(drop=True),
        sorted_data.loc[is_test].copy().reset_index(drop=True),
    )


def prepare_features(train, test, selected):
    X_train = train[selected].copy()
    X_test = test[selected].copy()
    bmi_median = None

    if "bmi" in selected:
        # 반드시 훈련용 BMI만 보고 중앙값을 계산
        bmi_median = X_train["bmi"].median()
        if pd.isna(bmi_median):
            raise ValueError("훈련용 BMI 중앙값을 계산할 수 없습니다.")
        X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
        X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("선택한 속성에 처리되지 않은 빈 값이 있습니다.")
    return X_train, X_test, bmi_median


def train_models(X_train, y_train):
    probability_model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    probability_model.fit(X_train, y_train)

    question_model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=SEED,
    )
    question_model.fit(X_train, y_train)
    return probability_model, question_model


# ============================================================
# 02. 혼동행렬에서 네 지표 직접 계산
# 뇌졸중(1)이 양성, 아님(0)이 음성
# ============================================================

def evaluate(y_true, predicted):
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    total = tn + fp + fn + tp
    predicted_positive = tp + fp
    actual_positive = tp + fn
    f1_denominator = 2 * tp + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / predicted_positive if predicted_positive else None
    # 실제 양성이 0명인 특수 상황에서는 재현율을 0으로 표시
    recall = tp / actual_positive if actual_positive else 0.0
    # 분모가 0인 특수 상황에서는 관례적으로 F1을 0으로 표시
    f1 = 2 * tp / f1_denominator if f1_denominator else 0.0

    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "합계": total,
        "정확도": accuracy,
        "정밀도": precision,
        "재현율": recall,
        "F1": f1,
    }


def format_percent(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value * 100:.2f}%"


# ============================================================
# 03. Plotly 혼동행렬 표: 실제 행 / 예측 열 + 합계
# ============================================================

def make_matrix_table(result, accent):
    tn, fp, fn, tp = (
        result["TN"], result["FP"], result["FN"], result["TP"]
    )
    total = result["합계"]

    columns = [
        ["실제 0 · 아님", "실제 1 · 뇌졸중", "열 합계"],
        [f"TN · {tn:,}명", f"FN · {fn:,}명", f"{tn + fn:,}명"],
        [f"FP · {fp:,}명", f"TP · {tp:,}명", f"{fp + tp:,}명"],
        [f"{tn + fp:,}명", f"{fn + tp:,}명", f"{total:,}명"],
    ]
    fills = [
        ["#F5F7FC", "#F5F7FC", "#E9EDF6"],
        ["#E4EDFF", "#FFE7EA", "#F1F4FA"],
        ["#FFF1E7", "#E5F8EE", "#F1F4FA"],
        ["#F1F4FA", "#F1F4FA", "#DFE7F7"],
    ]

    fig = go.Figure(go.Table(
        columnwidth=[130, 112, 112, 104],
        header=dict(
            values=["실제 ↓ / 예측 →", "예측 0 · 아님", "예측 1 · 뇌졸중", "행 합계"],
            fill_color=accent,
            font=dict(color="white", size=12),
            align="center", height=43,
            line_color="#FFFFFF",
        ),
        cells=dict(
            values=columns,
            fill_color=fills,
            font=dict(color=INK, size=12),
            align="center", height=49,
            line_color="#FFFFFF",
        ),
    ))
    fig.update_layout(
        height=225,
        margin=dict(l=0, r=0, t=3, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ============================================================
# 04. 오답 목록 정렬, 최대 20명 표시
# ============================================================

def mistakes_table(test, y_true, predicted, kind):
    actual_values = y_true.to_numpy()
    predicted_values = np.asarray(predicted)
    if kind == "FN":
        mask = (actual_values == 1) & (predicted_values == 0)
    else:
        mask = (actual_values == 0) & (predicted_values == 1)

    selected = test.loc[
        mask,
        ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"],
    ].copy()
    selected = selected.sort_values(
        ["age", "id"], ascending=[False, True], kind="mergesort"
    )
    return selected.rename(columns={
        "id": "번호",
        "age": "나이",
        "avg_glucose_level": "평균 혈당",
        "hypertension": "고혈압 여부",
        "heart_disease": "심장병 여부",
    })


# ============================================================
# 05. 화면
# ============================================================

st.html("""
<div class="grade-hero">
    <div class="grade-kicker">04 / NEURO FORCE · SCORE MISSION</div>
    <div class="grade-title">분류 결과 정밀 분석,<br><span>채점 전대 출동!</span></div>
    <div class="grade-lead">
        정답과 예측을 네 칸으로 나눠 오답의 종류를 확인합니다.
        세 모델의 정확도·정밀도·재현율·F1을 같은 기준으로 비교하고,
        놓친 사람과 헛짚은 사람을 직접 찾아봅니다.
    </div>
    <div class="grade-bars">
        <i style="background:#F34E67"></i>
        <i style="background:#386BFA"></i>
        <i style="background:#19B58A"></i>
        <i style="background:#FFC83D"></i>
        <i style="background:#E968C6"></i>
    </div>
</div>
""")

try:
    dataset = read_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다.")
    with st.expander("오류 상세"):
        st.code(str(exc))
    st.stop()

section(
    "01", "분류 모델과 같은 실험 조건",
    "입력 속성·분할 방식·훈련용 전처리·모델 설정을 분류 모델 페이지와 동일하게 맞춥니다.",
)
selected_features = st.multiselect(
    "모델에 입력할 속성",
    FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda f: f"{FEATURE_LABELS[f]} ({f})",
    key="grade_features",
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 골라 주세요. 그전에는 학습과 채점을 진행하지 않습니다.")
    st.stop()

train_df, test_df = split_by_id(dataset)
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

if y_train.nunique() < 2:
    st.error("훈련용 데이터에 0과 1이 모두 있어야 두 모델을 학습할 수 있습니다.")
    st.stop()

try:
    X_train, X_test, bmi_median = prepare_features(
        train_df, test_df, selected_features
    )
    logistic_model, tree_model = train_models(X_train, y_train)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# 로지스틱 회귀의 1(양성) 확률을 0.5 기준으로 분류
positive_index = list(logistic_model.classes_).index(1)
positive_prob = logistic_model.predict_proba(X_test)[:, positive_index]
prob_predictions = (positive_prob >= 0.5).astype(int)
question_predictions = tree_model.predict(X_test).astype(int)

# 입력값을 보지 않고 훈련용 다수 클래스만 예측
majority_class = int(y_train.value_counts().idxmax())
baseline_predictions = np.full(len(test_df), majority_class, dtype=int)

predictions = {
    "확률 모델": prob_predictions,
    "질문 모델": question_predictions,
    "다수 클래스 기준 모델": baseline_predictions,
}
scores = {
    model_name: evaluate(y_test, pred)
    for model_name, pred in predictions.items()
}

with st.sidebar:
    st.divider()
    st.markdown("**SCORE MISSION / 실험 설정**")
    st.caption("양성: stroke = 1 (뇌졸중)")
    st.caption("ID 오름차순 · 매 10명 앞 3명 채점")
    st.caption("표준화: 훈련용으로만 학습")
    st.caption("질문 모델: 최대 깊이 3 · 잎 최소 5명")
    st.caption("난수 고정: 42")

st.html(
    '<div class="grade-meta">'
    f'<b>채점용 사람: {len(test_df):,}명</b> · '
    f'<b>그중 실제 뇌졸중: {int(y_test.eq(1).sum()):,}명</b><br>'
    f'훈련용 {len(train_df):,}명 · '
    'ID 기준으로 10명 중 앞 3명 채점, 나머지 7명 훈련'
    '</div>'
)

if bmi_median is not None:
    st.info(
        f"BMI를 선택했으므로 훈련용 BMI 중앙값 {bmi_median:.2f}로 "
        "훈련용과 채점용 BMI 결측치를 채웠습니다."
    )
else:
    st.caption("BMI를 선택하지 않아 BMI 열의 결측치를 채우지 않았습니다.")

section(
    "02", "혼동행렬 · 네 칸의 의미",
    "행은 실제 정답, 열은 모델의 예측입니다. 각 칸에는 TP·FP·FN·TN과 인원을 표시했습니다.",
)
st.html(
    '<div class="grade-keyline">'
    '<span>🟢 TP: 실제 1 · 예측 1</span>'
    '<span>🔵 TN: 실제 0 · 예측 0</span>'
    '<span>🔴 FN: 실제 1 · 예측 0</span>'
    '<span>🟠 FP: 실제 0 · 예측 1</span>'
    '</div>'
)

left, right = st.columns(2, gap="large")
with left:
    st.html('<span class="grade-badge" style="background:#F34E67">RED / 확률 모델</span>')
    st.plotly_chart(make_matrix_table(scores["확률 모델"], RED), use_container_width=True)
with right:
    st.html('<span class="grade-badge" style="background:#386BFA">BLUE / 질문 모델</span>')
    st.plotly_chart(make_matrix_table(scores["질문 모델"], BLUE), use_container_width=True)

with st.expander("기준 모델의 혼동행렬도 확인"):
    st.plotly_chart(
        make_matrix_table(scores["다수 클래스 기준 모델"], GREEN),
        use_container_width=True,
    )

section(
    "03", "세 모델의 네 가지 채점 지표",
    "정확도·정밀도·재현율·F1을 혼동행렬의 네 칸으로 직접 계산합니다.",
)
metrics = pd.DataFrame([
    {
        "모델": name,
        "정확도": format_percent(s["정확도"]),
        "정밀도": format_percent(s["정밀도"]),
        "재현율": format_percent(s["재현율"]),
        "F1": format_percent(s["F1"]),
    }
    for name, s in scores.items()
])
st.dataframe(metrics, hide_index=True, use_container_width=True)

st.html(
    '<div class="grade-hint">'
    '<b>지표 계산법</b><br>'
    '정확도 = (TP + TN) / 전체 · '
    '정밀도 = TP / (TP + FP) · '
    '재현율 = TP / (TP + FN)<br>'
    'F1 = 2 × TP / (2 × TP + FP + FN)<br>'
    '뇌졸중이라고 예측한 사람이 0명이면 정밀도에만 '
    '<b>계산할 수 없음</b>을 표시합니다. '
    'F1은 별도의 식으로 계산하며, 분모가 0이면 0%로 표시합니다.'
    '</div>'
)

section(
    "04", "놓친 사람과 헛짚은 사람",
    "모델을 고르면 왼쪽에 FN, 오른쪽에 FP가 표시됩니다. 나이가 많은 사람부터 최대 20명까지 보여 줍니다.",
)
selected_model = st.selectbox(
    "오답 목록을 확인할 모델",
    list(predictions),
    key="grade_mistake_model",
)
selected_predictions = predictions[selected_model]
selected_score = scores[selected_model]

fn_list = mistakes_table(test_df, y_test, selected_predictions, "FN")
fp_list = mistakes_table(test_df, y_test, selected_predictions, "FP")

left, right = st.columns(2, gap="large")
with left:
    st.markdown(f"#### 🔴 놓친 사람 · FN 전체 {len(fn_list):,}명")
    st.caption("실제 뇌졸중(1)인데 뇌졸중 아님(0)으로 예측")
    if fn_list.empty:
        st.success("놓친 사람이 없습니다.")
    else:
        st.dataframe(
            fn_list.head(20), hide_index=True, use_container_width=True,
            column_config={
                "나이": st.column_config.NumberColumn(format="%.1f"),
                "평균 혈당": st.column_config.NumberColumn(format="%.2f"),
            },
        )
with right:
    st.markdown(f"#### 🩷 헛짚은 사람 · FP 전체 {len(fp_list):,}명")
    st.caption("실제 뇌졸중 아님(0)인데 뇌졸중(1)으로 예측")
    if fp_list.empty:
        st.success("헛짚은 사람이 없습니다.")
    else:
        st.dataframe(
            fp_list.head(20), hide_index=True, use_container_width=True,
            column_config={
                "나이": st.column_config.NumberColumn(format="%.1f"),
                "평균 혈당": st.column_config.NumberColumn(format="%.2f"),
            },
        )

assert len(fn_list) == selected_score["FN"]
assert len(fp_list) == selected_score["FP"]
st.success(
    f"목록 전체 인원과 혼동행렬 일치: "
    f"놓친 사람 {len(fn_list):,}명 = FN {selected_score['FN']:,}명, "
    f"헛짚은 사람 {len(fp_list):,}명 = FP {selected_score['FP']:,}명. "
    "목록이 20명을 초과하면 화면에는 앞 20명만 표시됩니다."
)

st.html(
    '<div class="grade-footer">'
    'NEURO FORCE · SCORE MISSION · EDUCATIONAL USE ONLY'
    '</div>'
)