
"""급식 규칙 찾기 | Streamlit 메뉴 연관 규칙 분석 앱."""

import re
from collections import Counter, defaultdict
from io import BytesIO
from itertools import combinations

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_CSV_URL = (
    "https://raw.githubusercontent.com/teacher188-netizen/"
    "2026-datascience/refs/heads/main/dorim_meals.csv"
)
RULE_COLUMNS = ["조건", "결과", "동시", "지지도", "신뢰도", "향상도"]

st.set_page_config(
    page_title="급식 규칙 찾기 | 도림 급식 데이터",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1250px; padding-top: 2rem; padding-bottom: 3rem;}
    h1, h2, h3 {letter-spacing: -0.035em;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 15px;
        padding: 0.95rem 1.1rem;
        background: rgba(128,128,128,0.055);
    }
    div[data-testid="stMetricLabel"] {font-weight: 650;}
    .helper-text {opacity: .8; font-size: .93rem; line-height: 1.65;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=3600)
def read_csv_from_url(url: str) -> pd.DataFrame:
    """UTF-8 또는 UTF-8 BOM CSV를 읽습니다."""
    return pd.read_csv(url, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(data), encoding="utf-8-sig")


def clean_menu(value: str) -> str:
    """메뉴 이름의 () 및 （） 괄호와 내부 내용(중첩 포함)을 제거합니다."""
    text = str(value)
    patterns = (r"\([^()]*\)", r"（[^（）]*）")
    while True:
        before = text
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        if text == before:
            break
    return " ".join(text.split()).strip()


@st.cache_data(show_spinner=False)
def analyze(raw_df: pd.DataFrame):
    """CSV 한 행(하루 식단)을 하나의 장바구니로 분석합니다."""
    df = raw_df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    required = {"날짜", "식사", "메뉴"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필요한 열이 없습니다: {', '.join(sorted(missing))}")

    # 같은 날 같은 메뉴가 두 번 있어도 그날 1회로만 계산합니다.
    transactions = []
    for raw in df["메뉴"].dropna():
        basket = {clean_menu(part) for part in str(raw).split("|")}
        basket.discard("")
        if basket:
            transactions.append(basket)

    day_count = len(transactions)
    if day_count == 0:
        raise ValueError("분석할 메뉴가 없습니다. 메뉴 열을 확인해 주세요.")

    menu_days = Counter(item for basket in transactions for item in basket)
    pair_days = Counter(
        pair
        for basket in transactions
        for pair in combinations(sorted(basket), 2)
    )

    records = []
    for (a, b), together in pair_days.items():
        for condition, result in ((a, b), (b, a)):
            support = together / day_count
            confidence = together / menu_days[condition]
            lift = confidence / (menu_days[result] / day_count)
            records.append(
                {
                    "조건": condition,
                    "결과": result,
                    "동시": together,
                    "지지도": support,
                    "신뢰도": confidence,
                    "향상도": lift,
                }
            )

    rules = pd.DataFrame(records, columns=RULE_COLUMNS)

    # "함께 나온 적 없는 짝"을 빠르게 찾기 위한 동시 등장 메뉴 집합
    neighbors = defaultdict(set)
    for a, b in pair_days:
        neighbors[a].add(b)
        neighbors[b].add(a)

    return day_count, menu_days, pair_days, rules, dict(neighbors)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    # 엑셀에서 한글이 깨지지 않도록 UTF-8 BOM으로 내려받습니다.
    return frame.to_csv(index=False).encode("utf-8-sig")


st.title("🍱 급식 규칙 찾기")
st.markdown(
    '<div class="helper-text">한 날의 식단을 하나의 장바구니로 보고, '
    '메뉴들의 동시 등장 규칙과 함께 등장하지 않은 조합을 알아봅니다.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📁 데이터 설정")
    source_url = st.text_input(
        "CSV 데이터 주소",
        value=DEFAULT_CSV_URL,
        help="날짜, 식사, 메뉴 열이 있는 UTF-8 CSV의 직접 다운로드 주소를 사용하세요.",
    )
    uploaded_file = st.file_uploader(
        "또는 CSV 파일 업로드", type=["csv"], help="업로드한 파일이 CSV 주소보다 우선합니다."
    )
    st.caption("CSV 파일을 업로드하면 입력한 주소 대신 업로드 파일로 분석합니다.")
    st.divider()
    st.markdown("**🔎 메뉴 정리 기준**")
    st.caption("메뉴명에서 괄호와 그 안의 설명을 모두 제거합니다.")
    st.caption("예: 배추김치(완) → 배추김치")
    st.caption("한 행의 같은 메뉴는 한 번만 계산합니다.")

try:
    if uploaded_file is not None:
        raw_df = read_csv_from_bytes(uploaded_file.getvalue())
        source_name = f"업로드 파일 · {uploaded_file.name}"
    elif source_url.strip():
        raw_df = read_csv_from_url(source_url.strip())
        source_name = "CSV 웹 주소"
    else:
        st.info("왼쪽에서 CSV 주소를 입력하거나 파일을 업로드해 주세요.")
        st.stop()
    day_count, menu_days, pair_days, rules_df, neighbors = analyze(raw_df)
except Exception as exc:
    st.error(f"데이터를 분석하지 못했습니다: {exc}")
    st.info("CSV 주소가 직접 다운로드 링크인지, 열 이름이 날짜·식사·메뉴인지 확인해 주세요.")
    st.stop()

menus = sorted(menu_days)
st.caption(f"✅ 데이터 불러오기 완료 · {source_name} · 괄호 내용 제거 적용")

# 기본 통계: 아래의 필터와 무관한 전체 데이터 기준
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("🗓️ 급식 일수", f"{day_count:,}일")
kpi2.metric("🥗 메뉴 종류 수", f"{len(menu_days):,}개")
kpi3.metric("🤝 함께 나온 메뉴 쌍", f"{len(pair_days):,}쌍")

st.divider()
st.header("📊 메뉴 연관 규칙")
st.caption("조건 → 결과와 결과 → 조건을 각각 별개의 규칙으로 보여 줍니다.")

# 메뉴 선택과 정렬은 독립적이며, 슬라이더는 표 및 그래프에 함께 적용됩니다.
c1, c2, c3 = st.columns([1.3, 1.0, 1.3], gap="large")
with c1:
    selected_rule_menu = st.selectbox(
        "🔍 규칙에서 메뉴 좁혀 보기",
        ["전체 메뉴"] + menus,
        key="rule_menu_filter",
        help="선택한 메뉴가 조건 또는 결과인 규칙만 표시합니다.",
    )
with c2:
    sort_by = st.selectbox("↕️ 규칙 정렬 기준", ["향상도", "신뢰도", "동시"], key="rule_sort")
with c3:
    min_together = st.slider(
        "🎚️ 최소 동시 일수",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="min_together_days",
        help="함께 등장한 일수가 선택한 값보다 적은 규칙을 제외합니다.",
    )

filtered = rules_df[rules_df["동시"] >= min_together].copy()
if selected_rule_menu != "전체 메뉴":
    filtered = filtered.loc[
        (filtered["조건"] == selected_rule_menu)
        | (filtered["결과"] == selected_rule_menu)
    ]

# 동일한 동률 처리로 정렬 결과를 일정하게 유지합니다.
priority = [sort_by] + [col for col in ("향상도", "신뢰도", "동시") if col != sort_by]
filtered = filtered.sort_values(
    priority + ["조건", "결과"],
    ascending=[False, False, False, True, True],
    kind="stable",
)

st.markdown(
    f"**📌 현재 표에 남은 규칙: {len(filtered):,}개** "
    f"(동시 {min_together}일 이상 · "
    f"메뉴: {selected_rule_menu})"
)

# 수치 계산은 0~1 비율로 보관하고, 표에서만 % 단위로 표시합니다.
display_rules = filtered.copy()
display_rules["지지도"] = display_rules["지지도"] * 100
display_rules["신뢰도"] = display_rules["신뢰도"] * 100

if filtered.empty:
    st.info("현재 조건에 해당하는 규칙이 없습니다. 최소 동시 일수를 낮추거나 전체 메뉴를 선택해 보세요.")
else:
    st.dataframe(
        display_rules,
        use_container_width=True,
        hide_index=True,
        height=min(540, 48 + len(display_rules) * 35),
        column_config={
            "조건": st.column_config.TextColumn("조건"),
            "결과": st.column_config.TextColumn("결과"),
            "동시": st.column_config.NumberColumn("동시", format="%d일"),
            "지지도": st.column_config.NumberColumn("지지도", format="%.2f%%"),
            "신뢰도": st.column_config.NumberColumn("신뢰도", format="%.2f%%"),
            "향상도": st.column_config.NumberColumn("향상도", format="%.3f"),
        },
    )
    st.download_button(
        "⬇️ 현재 규칙 표 CSV 다운로드",
        data=csv_bytes(display_rules),
        file_name="meal_association_rules.csv",
        mime="text/csv",
        key="download_rules",
    )

st.subheader("📈 향상도 상위 10개 규칙")
st.caption("위에서 선택한 메뉴와 최소 동시 일수 조건을 그래프에도 동일하게 적용합니다.")
if filtered.empty:
    st.info("표시할 규칙이 없어 그래프가 비어 있습니다.")
else:
    top10 = (
        filtered.sort_values(
            ["향상도", "신뢰도", "동시", "조건", "결과"],
            ascending=[False, False, False, True, True],
        )
        .head(10)
        .copy()
    )
    top10["규칙"] = top10["조건"] + " → " + top10["결과"]
    top10["지지도(%)"] = top10["지지도"] * 100
    top10["신뢰도(%)"] = top10["신뢰도"] * 100
    top10 = top10.iloc[::-1]
    chart = px.bar(
        top10,
        x="향상도",
        y="규칙",
        orientation="h",
        text="향상도",
        color="향상도",
        color_continuous_scale="Tealgrn",
        hover_data={
            "동시": True,
            "지지도(%)": ":.2f",
            "신뢰도(%)": ":.2f",
            "향상도": ":.3f",
        },
    )
    chart.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    chart.update_layout(
        height=max(330, len(top10) * 49 + 90),
        margin=dict(l=10, r=60, t=15, b=10),
        xaxis_title="향상도 (Lift)",
        yaxis_title=None,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    chart.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    st.plotly_chart(chart, use_container_width=True)

with st.expander("ℹ️ 지지도·신뢰도·향상도 계산 방법"):
    st.markdown(
        """
        - **동시:** 조건 메뉴와 결과 메뉴가 같은 날 함께 나온 일수
        - **지지도:** 동시 등장 일수 ÷ 전체 급식 일수
        - **신뢰도:** 동시 등장 일수 ÷ 조건 메뉴의 등장 일수
        - **향상도:** 신뢰도 ÷ 결과 메뉴의 등장 비율

        향상도 **1 초과**는 두 메뉴가 우연한 독립 등장보다 더 자주 함께 나온다는 뜻입니다.
        함께 나온 일수가 아주 적은 규칙은 향상도가 높아도 해석에 주의하세요.
        """
    )

st.divider()
st.header("🧩 함께 나온 적 없는 짝")
st.caption(
    "선택한 메뉴와 같은 날 **단 한 번도 함께 등장하지 않았고**, "
    "자신은 **10일 이상 등장한** 메뉴를 찾습니다. "
    "이 분석은 위의 규칙 필터·슬라이더와 별개입니다."
)

# 규칙 필터와 다른 이름, 다른 key로 완전히 독립적으로 관리합니다.
selected_absent_menu = st.selectbox(
    "🍽️ 함께 나온 적 없는 짝을 찾을 기준 메뉴",
    menus,
    key="absent_pair_menu",
    help="위의 '규칙에서 메뉴 좁혀 보기'와 별개로 선택합니다.",
)

never_together = [
    (menu, count)
    for menu, count in menu_days.items()
    if menu != selected_absent_menu
    and count >= 10
    and menu not in neighbors.get(selected_absent_menu, set())
]
never_together.sort(key=lambda value: (-value[1], value[0]))

m1, m2 = st.columns(2)
m1.metric("🍽️ 선택 메뉴가 나온 날", f"{menu_days[selected_absent_menu]:,}일")
m2.metric("🧩 함께 나온 적 없는 짝", f"{len(never_together):,}개")

if never_together:
    absent_df = pd.DataFrame(never_together, columns=["함께 나온 적 없는 메뉴", "나온 날 수"])
    st.dataframe(
        absent_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "함께 나온 적 없는 메뉴": st.column_config.TextColumn("함께 나온 적 없는 메뉴"),
            "나온 날 수": st.column_config.NumberColumn("나온 날 수", format="%d일"),
        },
    )
    st.download_button(
        "⬇️ 함께 나온 적 없는 짝 CSV 다운로드",
        data=csv_bytes(absent_df),
        file_name="meal_never_together.csv",
        mime="text/csv",
        key="download_absent",
    )
else:
    st.info("이 메뉴와 한 번도 함께 나오지 않으면서 10일 이상 등장한 다른 메뉴가 없습니다.")

st.caption(
    "※ 모든 등장 일수는 분석한 CSV에서 메뉴가 포함된 식단 행 수를 기준으로 계산합니다. "
    "메뉴 이름의 괄호 및 괄호 안 설명은 제거하고 집계합니다."
)
