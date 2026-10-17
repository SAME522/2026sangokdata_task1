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
# NEURO FORCE / 05. 전처리와 클래스 무게 실험
# main.py가 페이지 설정 및 왼쪽 페이지 메뉴를 관리합니다.
# ============================================================

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
FEATURES = [
    "age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"
]
DEFAULT_FEATURES = [
    "age", "avg_glucose_level", "hypertension", "heart_disease"
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
# 디자인. HTML은 Markdown 파서 대신 st.html()로 표시
# ============================================================

st.html("""
<style>
.pre-hero {
    background: linear-gradient(115deg, #F1F5FF, #F0FFF9 52%, #FFF0F9);
    border: 1px solid #DDE6FA; border-radius: 22px;
    padding: 38px 42px; margin: 12px 0 24px;
    position: relative; overflow: hidden;
}
.pre-hero::after {
    content: '✦'; position: absolute; right: 28px; top: -58px;
    font-size: 210px; color: #E968C61C; pointer-events: none;
}
.pre-kicker { font-size: 12px; color: #386BFA; font-weight: 900; letter-spacing: .16em; }
.pre-title {
    color: #202943; font-size: clamp(34px, 5vw, 60px);
    font-weight: 900; letter-spacing: -.065em; line-height: 1.16;
    margin: 14px 0 16px; position: relative; z-index: 1;
}
.pre-title span {
    background: linear-gradient(94deg,#F34E67,#E968C6,#386BFA);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.pre-lead { color: #52617C; line-height: 1.85; max-width: 750px; font-size: 15px; }
.pre-colors { display: flex; max-width: 275px; gap: 6px; margin-top: 24px; }
.pre-colors i { flex: 1; height: 8px; border-radius: 999px; }
.pre-section {
    display: flex; align-items: baseline; gap: 17px;
    border-top: 1px solid #DCE4F2; padding-top: 19px; margin: 44px 0 9px;
}
.pre-section b { color: #386BFA; font-size: 12px; }
.pre-section span { color: #202943; font-size: clamp(22px, 3vw, 30px); font-weight: 850; letter-spacing: -.045em; }
.pre-desc { color: #65738B; font-size: 14px; line-height: 1.8; margin-bottom: 22px; }
.pre-info {
    background: #F0F4FF; border: 1px solid #DFE8F9;
    border-left: 4px solid #386BFA; border-radius: 0 10px 10px 0;
    padding: 17px 20px; color: #41516D; line-height: 1.85; margin: 16px 0;
}
.pre-summary {
    background: #FFF8E8; border: 1px solid #F5E4B2; border-radius: 12px;
    padding: 16px 19px; color: #66543A; line-height: 1.8; margin-top: 13px;
}
.pre-label {
    display: inline-block; padding: 6px 11px; color: #FFFFFF;
    font-weight: 850; font-size: 12px; border-radius: 7px; margin: 5px 0 12px;
}
.pre-footer {
    border-top: 1px solid #DFE6F2; margin-top: 55px; padding-top: 20px;
    font-size: 11px; letter-spacing: .1em; color: #8490A4;
}
@media(max-width:700px) {
    .pre-hero { padding: 26px 23px; }
    .pre-title { font-size: 36px; }
    .pre-section { margin-top: 34px; }
}
</style>
""")


def section(index, title, description):
    st.html(
        '<div class="pre-section">'
        f'<b>{html.escape(index)}</b>'
        f'<span>{html.escape(title)}</span>'
        '</div>'
        f'<div class="pre-desc">{html.escape(description)}</div>'
    )


def info(content):
    st.html('<div class="pre-info">' + content + '</div>')


# ============================================================
# 채점표 페이지와 동일한 CSV, ID 정렬 및 고정 3:7 분할
# ============================================================

@st.cache_data(ttl=3600)
def read_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    required = ["id", "stroke"] + FEATURES
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError("CSV 필수 열이 없습니다: " + ", ".join(missing))
    if data["id"].isna().any() or data["id"].duplicated().any():
        raise ValueError("ID에 빈 값 또는 중복이 있습니다.")
    if data["stroke"].isna().any() or not data["stroke"].isin([0, 1]).all():
        raise ValueError("stroke 열에는 0과 1만 있어야 합니다.")
    return data


def split_by_id(data):
    ordered = data.sort_values("id", kind="mergesort").reset_index(drop=True)
    test_mask = (np.arange(len(ordered)) % 10) < 3
    return (
        ordered.loc[~test_mask].copy().reset_index(drop=True),
        ordered.loc[test_mask].copy().reset_index(drop=True),
    )


def prepare_features(train, test, selected):
    X_train = train[selected].copy()
    X_test = test[selected].copy()
    bmi_median = None
    if "bmi" in selected:
        # 삭제 모드라면 결측치가 없고, 유지 모드라면 훈련용만 보고 계산
        bmi_median = X_train["bmi"].median()
        if pd.isna(bmi_median):
            raise ValueError("훈련용 BMI 중앙값을 계산할 수 없습니다.")
        X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
        X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("선택한 속성에 처리되지 않은 결측치가 있습니다.")
    return X_train, X_test, bmi_median


def make_probability_model(X, y, balanced=False):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            class_weight="balanced" if balanced else None,
        )),
    ])
    model.fit(X, y)
    return model


def make_question_model(X, y, balanced=False):
    model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=SEED,
        class_weight="balanced" if balanced else None,
    )
    model.fit(X, y)
    return model


def predict_positive(model, X):
    # 양성: stroke = 1
    if isinstance(model, Pipeline):
        labels = model.named_steps["logistic"].classes_
    else:
        labels = model.classes_
    positive_index = list(labels).index(1)
    return (model.predict_proba(X)[:, positive_index] >= 0.5).astype(int)


def confusion_scores(y_true, y_pred):
    tn, fp, fn, tp = map(
        int, confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    )
    n = tn + fp + fn + tp
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "정확도": (tp + tn) / n if n else 0.0,
        "정밀도": tp / (tp + fp) if tp + fp else None,
        "재현율": tp / (tp + fn) if tp + fn else 0.0,
        "F1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
    }


def format_percent(value):
    return "계산할 수 없음" if value is None else f"{value * 100:.2f}%"


def confusion_table(score, color):
    # 행은 실제, 열은 예측, 마지막 행/열은 합계
    tn, fp, fn, tp = (score[k] for k in ("TN", "FP", "FN", "TP"))
    fig = go.Figure(go.Table(
        columnwidth=[125, 112, 112, 110],
        header=dict(
            values=["실제 ↓ / 예측 →", "예측 0: 아님", "예측 1: 뇌졸중", "행 합계"],
            fill_color=color,
            font=dict(color="white", size=12),
            align="center", height=41,
        ),
        cells=dict(
            values=[
                ["실제 0: 아님", "실제 1: 뇌졸중", "열 합계"],
                [f"TN · {tn:,}명", f"FN · {fn:,}명", f"{tn+fn:,}명"],
                [f"FP · {fp:,}명", f"TP · {tp:,}명", f"{fp+tp:,}명"],
                [f"{tn+fp:,}명", f"{fn+tp:,}명", f"{tn+fp+fn+tp:,}명"],
            ],
            fill_color=[
                ["#F5F7FC", "#F5F7FC", "#E9EDF6"],
                ["#E4EDFF", "#FFE7EA", "#F0F3FA"],
                ["#FFF1E7", "#E5F8EE", "#F0F3FA"],
                ["#F0F3FA", "#F0F3FA", "#DFE7F7"],
            ],
            align="center", height=47,
            font=dict(size=12, color=INK),
        ),
    ))
    fig.update_layout(height=215, margin=dict(l=0, r=0, t=2, b=5), paper_bgcolor="white")
    return fig


def metric_comparison(before, after):
    rows = []
    for metric in ("정확도", "정밀도", "재현율", "F1"):
        rows.append({
            "지표": metric,
            "무게 맞추기 전": format_percent(before[metric]),
            "무게 맞춘 후": format_percent(after[metric]),
        })
    return pd.DataFrame(rows)


def counts_change(before, after):
    return (
        f"찾아낸 환자(TP) {before['TP']:,}명 → {after['TP']:,}명 · "
        f"놓친 환자(FN) {before['FN']:,}명 → {after['FN']:,}명 · "
        f"헛짚은 사람(FP) {before['FP']:,}명 → {after['FP']:,}명"
    )


# ============================================================
# 로지스틱 회귀 경계: 표준화된 계수를 원래 축 단위로 환산
# ============================================================

def raw_coefficients(model):
    scaler = model.named_steps["scaler"]
    logistic = model.named_steps["logistic"]
    coefficients = np.asarray(logistic.coef_[0], dtype=float) / scaler.scale_
    intercept = float(logistic.intercept_[0] - np.dot(coefficients, scaler.mean_))
    return coefficients, intercept


def clipped_line(coefficients, intercept, ranges):
    (xmin, xmax), (ymin, ymax) = ranges
    a, b = map(float, coefficients)
    tiny = 1e-12
    if abs(a) < tiny and abs(b) < tiny:
        return None
    pts = []
    if abs(b) > tiny:
        for x in (xmin, xmax):
            y = -(a * x + intercept) / b
            if ymin - 1e-8 <= y <= ymax + 1e-8:
                pts.append((float(x), float(np.clip(y, ymin, ymax))))
    if abs(a) > tiny:
        for y in (ymin, ymax):
            x = -(b * y + intercept) / a
            if xmin - 1e-8 <= x <= xmax + 1e-8:
                pts.append((float(np.clip(x, xmin, xmax)), float(y)))
    unique = []
    for p in pts:
        if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < 1e-6 for q in unique):
            unique.append(p)
    if len(unique) < 2:
        return None
    pairs = [(p, q) for i, p in enumerate(unique) for q in unique[i+1:]]
    return max(pairs, key=lambda pq: np.hypot(pq[0][0]-pq[1][0], pq[0][1]-pq[1][1]))


def axis_range(series):
    lo, hi = float(series.min()), float(series.max())
    pad = max((hi - lo) * 0.065, 0.12 if hi - lo < 2 else 0.5)
    return lo - pad, hi + pad


def scatter_2d(X_train, X_test, y_test, axes, before_model, after_model):
    xcol, ycol = axes
    xr = axis_range(pd.concat([X_train[xcol], X_test[xcol]]))
    yr = axis_range(pd.concat([X_train[ycol], X_test[ycol]]))
    fig = go.Figure()
    notes = []
    for name, model, color in (
        ("무게 전 0.5 경계", before_model, GREEN),
        ("무게 후 0.5 경계", after_model, PINK),
    ):
        w, bias = raw_coefficients(model)
        segment = clipped_line(w, bias, (xr, yr))
        if segment is None:
            notes.append(f"{name}선이 현재 2차원 그림 범위 밖에 있거나 경계를 만들 수 없습니다.")
        else:
            (x1, y1), (x2, y2) = segment
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2], mode="lines", name=name,
                line=dict(color=color, width=4, dash="dash"),
                hoverinfo="name",
            ))
    true_values = y_test.to_numpy()
    for outcome, name, color in (
        (0, "실제: 뇌졸중 아님", BLUE),
        (1, "실제: 뇌졸중", RED),
    ):
        mask = true_values == outcome
        if mask.any():
            fig.add_trace(go.Scatter(
                x=X_test.loc[mask, xcol], y=X_test.loc[mask, ycol],
                mode="markers", name=name,
                marker=dict(color=color, size=8, opacity=0.75,
                            line=dict(color="white", width=1)),
                hovertemplate=("가로 %{x:.2f}<br>세로 %{y:.2f}"
                               "<extra>%{fullData.name}</extra>"),
            ))
    fig.update_layout(
        template="plotly_white", height=530,
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title=LABELS[xcol], range=xr, gridcolor="#EBEFF8"),
        yaxis=dict(title=LABELS[ycol], range=yr, gridcolor="#EBEFF8"),
        legend=dict(orientation="h", y=1.06, yanchor="bottom", x=0),
        margin=dict(l=25, r=25, t=70, b=35),
        font=dict(color=INK),
    )
    return fig, notes


def clipped_plane(coef, bias, limits):
    """Return a bounded planar polygon formed by plane / 3D box intersection."""
    w = np.asarray(coef, dtype=float)
    low = np.array([pair[0] for pair in limits], dtype=float)
    high = np.array([pair[1] for pair in limits], dtype=float)
    if np.linalg.norm(w) < 1e-12:
        return None

    # The 12 edges of an axis-aligned box
    corners = [np.array([x, y, z], dtype=float)
               for x in (low[0], high[0])
               for y in (low[1], high[1])
               for z in (low[2], high[2])]
    hits = []
    for i, first in enumerate(corners):
        for second in corners[i+1:]:
            changed = np.flatnonzero(np.abs(first - second) > 1e-10)
            if len(changed) != 1:
                continue
            d0 = float(np.dot(w, first) + bias)
            d1 = float(np.dot(w, second) + bias)
            if abs(d0 - d1) < 1e-12:
                if abs(d0) < 1e-9:
                    hits.extend([first, second])
                continue
            t = d0 / (d0 - d1)
            if -1e-10 <= t <= 1.0 + 1e-10:
                hits.append(first + np.clip(t, 0, 1) * (second - first))

    points = []
    for p in hits:
        if not any(np.linalg.norm(p - q) < 1e-7 for q in points):
            points.append(p)
    if len(points) < 3:
        return None

    array = np.asarray(points)
    center = array.mean(axis=0)
    normal = w / np.linalg.norm(w)
    reference = np.array([1., 0., 0.])
    if abs(np.dot(normal, reference)) > 0.9:
        reference = np.array([0., 1., 0.])
    e1 = np.cross(normal, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    relative = array - center
    angles = np.arctan2(relative @ e2, relative @ e1)
    return array[np.argsort(angles)]


def scatter_3d(X_train, X_test, y_test, axes, before_model, after_model):
    limits = [axis_range(pd.concat([X_train[c], X_test[c]])) for c in axes]
    fig = go.Figure()
    notes = []
    for label, model, color in (
        ("무게 전 0.5 평면", before_model, GREEN),
        ("무게 후 0.5 평면", after_model, PINK),
    ):
        w, bias = raw_coefficients(model)
        polygon = clipped_plane(w, bias, limits)
        if polygon is None:
            notes.append(f"{label}이 현재 3차원 그림 범위 밖에 있거나 평면을 만들 수 없습니다.")
        else:
            n = len(polygon)
            fig.add_trace(go.Mesh3d(
                x=polygon[:, 0], y=polygon[:, 1], z=polygon[:, 2],
                i=[0] * (n - 2), j=list(range(1, n - 1)),
                k=list(range(2, n)),
                color=color, opacity=0.26, flatshading=True,
                name=label, showlegend=True, hoverinfo="name",
            ))
    truth = y_test.to_numpy()
    for actual, title, color in (
        (0, "실제: 뇌졸중 아님", BLUE),
        (1, "실제: 뇌졸중", RED),
    ):
        mask = truth == actual
        if mask.any():
            fig.add_trace(go.Scatter3d(
                x=X_test.loc[mask, axes[0]],
                y=X_test.loc[mask, axes[1]],
                z=X_test.loc[mask, axes[2]],
                mode="markers", name=title,
                marker=dict(size=3.2, color=color, opacity=0.76,
                            line=dict(color="white", width=0.5)),
                hovertemplate=("%{x:.2f}, %{y:.2f}, %{z:.2f}"
                               "<extra>%{fullData.name}</extra>"),
            ))
    fig.update_layout(
        template="plotly_white", height=630,
        paper_bgcolor="white",
        scene=dict(
            xaxis=dict(title=LABELS[axes[0]], range=limits[0]),
            yaxis=dict(title=LABELS[axes[1]], range=limits[1]),
            zaxis=dict(title=LABELS[axes[2]], range=limits[2]),
            bgcolor="white",
            aspectmode="cube",
        ),
        legend=dict(orientation="h", x=0, y=1.06, yanchor="bottom"),
        margin=dict(l=0, r=0, t=70, b=0),
        font=dict(color=INK),
    )
    return fig, notes


# ============================================================
# 화면 시작: 맨 위 세 가지 선택 사항
# ============================================================

st.html(
    '<div class="pre-hero">'
    '<div class="pre-kicker">05 / NEURO FORCE · PREPROCESS MODE</div>'
    '<div class="pre-title">데이터 전처리 작전,<br>'
    '<span>균형의 힘을 깨워라!</span></div>'
    '<div class="pre-lead">드문 뇌졸중 사례의 무게를 높이기 전과 후를 비교합니다. '
    '같은 채점용 사람에게 같은 시험을 실시하고, '
    '놓친 환자가 얼마나 바뀌었는지 살펴보세요.</div>'
    '<div class="pre-colors">'
    '<i style="background:#F34E67"></i>'
    '<i style="background:#386BFA"></i>'
    '<i style="background:#19B58A"></i>'
    '<i style="background:#FFC83D"></i>'
    '<i style="background:#E968C6"></i>'
    '</div></div>'
)

controls = st.columns([0.9, 1.1, 1.9], gap="large")
with controls[0]:
    weighted_active = st.toggle(
        "양쪽 무게 같게 학습", value=False, key="pre_active_weight",
        help="전·후 모델은 항상 둘 다 학습합니다. 이 스위치는 아래 놓친 환자 목록의 기준만 결정합니다.",
    )
with controls[1]:
    bmi_mode = st.radio(
        "BMI가 비어 있는 사람", ["그대로 두기", "삭제하기"],
        horizontal=True, key="pre_missing_choice",
    )
with controls[2]:
    selected = st.multiselect(
        "입력에 사용할 속성", FEATURES, default=DEFAULT_FEATURES,
        format_func=lambda key: f"{LABELS[key]} ({key})",
        key="pre_selected_features",
    )

try:
    original = read_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다.")
    with st.expander("오류 상세"):
        st.code(str(exc))
    st.stop()

# 먼저 채점표와 똑같이 ID 순으로 나누어 채점용 배정 고정
train_df, test_df = split_by_id(original)
removed = original.loc[original["bmi"].isna()]
removed_positive = int(removed["stroke"].eq(1).sum())

if bmi_mode == "삭제하기":
    train_df = train_df.dropna(subset=["bmi"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["bmi"]).reset_index(drop=True)
    st.warning(
        f"BMI 결측 행 {len(removed):,}명 삭제 · "
        f"그중 실제 뇌졸중 {removed_positive:,}명. "
        "ID 기준 분할을 먼저 고정한 뒤 훈련·채점 양쪽에서 제외했습니다."
    )
else:
    st.caption(
        f"BMI 결측 행 {len(removed):,}명 유지 · "
        "BMI를 입력 속성으로 선택한 경우에만 훈련용 중앙값으로 채웁니다."
    )

if len(selected) < 2:
    st.info("입력 속성을 두 개 이상 선택해야 모델을 학습하고 좌표축을 정할 수 있습니다.")
    st.stop()

if train_df.empty or test_df.empty:
    st.error("훈련용 또는 채점용 데이터가 비어 있습니다.")
    st.stop()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)
if y_train.nunique() != 2:
    st.error("훈련용 데이터에는 정답 0과 1이 모두 있어야 합니다.")
    st.stop()

try:
    X_train, X_test, bmi_median = prepare_features(train_df, test_df, selected)
    # 스위치와 무관하게 두 설정을 모두 학습/채점
    models = {
        "확률 모델": {
            False: make_probability_model(X_train, y_train, balanced=False),
            True: make_probability_model(X_train, y_train, balanced=True),
        },
        "질문 모델": {
            False: make_question_model(X_train, y_train, balanced=False),
            True: make_question_model(X_train, y_train, balanced=True),
        },
    }
except ValueError as exc:
    st.error(str(exc))
    st.stop()

predictions = {
    name: {mode: predict_positive(model, X_test)
           for mode, model in variants.items()}
    for name, variants in models.items()
}
scores = {
    name: {mode: confusion_scores(y_test, answer)
           for mode, answer in variants.items()}
    for name, variants in predictions.items()
}

with st.sidebar:
    st.divider()
    st.markdown("**PREPROCESS / 실험 규칙**")
    st.caption("양성 = 뇌졸중(1) · 음성 = 아님(0)")
    st.caption("ID 오름차순, 10명 중 앞 3명 채점")
    st.caption("로지스틱: 훈련용만 표준화")
    st.caption("결정나무: 깊이 최대 3, 잎 최소 5명")
    st.caption("무게 전: 기본 설정 · 무게 후: class_weight='balanced'")
    st.caption("난수 고정: 42")

info(
    f"<b>훈련용 {len(train_df):,}명 · 채점용 {len(test_df):,}명</b> · "
    f"채점용 실제 뇌졸중 {int(y_test.eq(1).sum()):,}명. "
    "두 모델 모두 전·후를 동일한 데이터로 학습·채점합니다."
)
if bmi_median is not None and bmi_mode == "그대로 두기":
    st.info(
        f"선택된 BMI는 훈련용 중앙값 {bmi_median:.2f}로 채웠습니다. "
        "채점용 BMI로 중앙값을 계산하지 않았습니다."
    )

# ============================================================
# 01. 두 모델의 전후 혼동행렬과 지표
# ============================================================

section(
    "01", "균형 무게 적용 전후의 채점표",
    "왼쪽이 기본 학습, 오른쪽이 양성·음성 가중치를 같게 한 학습입니다. "
    "행은 실제, 열은 예측입니다.",
)

for number, model_name, accent in (
    ("A", "확률 모델", RED),
    ("B", "질문 모델", BLUE),
):
    st.html(
        f'<span class="pre-label" style="background:{accent}">'
        f'{number} / {html.escape(model_name)}</span>'
    )
    before, after = scores[model_name][False], scores[model_name][True]
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**무게 맞추기 전**")
        st.plotly_chart(confusion_table(before, BLUE), use_container_width=True)
    with right:
        st.markdown("**무게 맞춘 후**")
        st.plotly_chart(confusion_table(after, PINK), use_container_width=True)

    st.dataframe(metric_comparison(before, after), hide_index=True, use_container_width=True)
    st.html(
        '<div class="pre-summary"><b>인원 변화</b><br>'
        + html.escape(counts_change(before, after))
        + '</div>'
    )
    st.write("")

st.caption(
    "정밀도 분모(TP+FP)가 0이면 ‘계산할 수 없음’을 표시합니다. "
    "F1은 2TP/(2TP+FP+FN)으로 직접 계산하며, 식의 분모가 0이면 0%로 표시합니다."
)

# ============================================================
# 02. 두 축으로 다시 학습한 로지스틱의 전후 경계선
# ============================================================

section(
    "02", "두 속성으로 보는 0.5 분류 경계",
    "색깔별 점은 채점용 사람의 실제 정답입니다. "
    "초록 점선은 무게 전, 분홍 점선은 무게 후 확률 0.5 경계입니다.",
)

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_axis = st.selectbox(
        "가로축", selected, format_func=lambda c: LABELS[c], key="pre_x_axis"
    )
with axis_col2:
    y_axis = st.selectbox(
        "세로축", [c for c in selected if c != x_axis],
        format_func=lambda c: LABELS[c], key="pre_y_axis",
    )

axis_two = [x_axis, y_axis]
map_before = make_probability_model(X_train[axis_two], y_train, balanced=False)
map_after = make_probability_model(X_train[axis_two], y_train, balanced=True)
figure2d, notes2d = scatter_2d(
    X_train, X_test, y_test, axis_two, map_before, map_after
)
st.plotly_chart(figure2d, use_container_width=True)
for message in notes2d:
    st.warning(message)

# ============================================================
# 03. 세 축만으로 다시 학습한 전후 로지스틱 평면
# ============================================================

if len(selected) >= 3:
    section(
        "03", "3차원 분류 평면 · 마우스로 회전",
        "세 가지 속성만으로 새로 학습한 무게 전·후 0.5 평면을 표시합니다. "
        "드래그하여 회전하고 휠로 확대할 수 있습니다.",
    )
    default_axes = list(selected[:3])
    axes_three = st.multiselect(
        "3차원 축으로 쓸 속성 세 개", selected,
        default=default_axes, max_selections=3,
        format_func=lambda c: LABELS[c], key="pre_3d_axes",
    )
    if len(axes_three) != 3:
        st.info("3차원 그림은 서로 다른 속성 세 개를 선택해야 나타납니다.")
    else:
        model3_before = make_probability_model(X_train[axes_three], y_train, balanced=False)
        model3_after = make_probability_model(X_train[axes_three], y_train, balanced=True)
        figure3d, notes3d = scatter_3d(
            X_train, X_test, y_test, axes_three, model3_before, model3_after
        )
        st.plotly_chart(figure3d, use_container_width=True)
        for message in notes3d:
            st.warning(message)

# ============================================================
# 04. 스위치에 따라 선택되는 놓친 환자(FN) 목록
# ============================================================

section(
    "04", "확률 모델이 놓친 환자",
    "스위치가 켜져 있으면 무게를 맞춘 모델, 꺼져 있으면 기본 모델의 FN을 표시합니다. "
    "나이가 어린 사람부터 정렬합니다.",
)
chosen_prediction = predictions["확률 모델"][weighted_active]
fn_mask = (y_test.to_numpy() == 1) & (chosen_prediction == 0)
missed = (
    test_df.loc[fn_mask, ["id", "age", "hypertension", "heart_disease"]]
    .sort_values(["age", "id"], ascending=[True, True], kind="mergesort")
    .rename(columns={
        "id": "번호", "age": "나이",
        "hypertension": "고혈압 여부", "heart_disease": "심장병 여부",
    })
)
chosen_score = scores["확률 모델"][weighted_active]
assert len(missed) == chosen_score["FN"]
mode_name = "무게 맞춘 후" if weighted_active else "무게 맞추기 전"
st.markdown(f"**{mode_name} · 놓친 환자 전체 {len(missed):,}명 (FN)**")
if missed.empty:
    st.success("이 설정에서 놓친 환자가 없습니다.")
else:
    st.dataframe(
        missed, hide_index=True, use_container_width=True, height=380,
        column_config={
            "나이": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.download_button(
        "놓친 환자 목록 CSV 다운로드",
        missed.to_csv(index=False).encode("utf-8-sig"),
        file_name="neuro_preprocessing_missed.csv",
        mime="text/csv",
    )
st.caption(
    f"전체 목록 {len(missed):,}명 = 혼동행렬 FN {chosen_score['FN']:,}명. "
    "이 페이지는 모델 비교를 위한 교육용 실습이며 의료 진단에 사용할 수 없습니다."
)
st.html(
    '<div class="pre-footer">NEURO FORCE · PREPROCESS MODE · EDUCATIONAL USE ONLY</div>'
)
