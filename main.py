import re
from collections import Counter
from itertools import combinations
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="급식 규칙 찾기", page_icon="🍱", layout="wide")
st.title("🍱 급식 규칙 찾기")
st.caption("하루 식단을 하나의 장바구니로 보고, 함께 나온 메뉴의 연관 규칙을 분석합니다.")

url = st.text_input("CSV 데이터 주소", placeholder="https://example.com/school_lunch.csv", help="패들렛에서 학교별 CSV의 직접 다운로드 주소를 입력하세요.")
upload = st.file_uploader("또는 CSV 파일 업로드", type=["csv"])

@st.cache_data(show_spinner=False)
def read_url(source):
    return pd.read_csv(source, encoding="utf-8-sig")

@st.cache_data(show_spinner=False)
def read_upload(contents):
    return pd.read_csv(BytesIO(contents), encoding="utf-8-sig")

def clean_menu(value):
    # 일반 괄호와 전각 괄호 및 내부 내용 제거 (중첩 괄호도 반복 처리)
    name = str(value)
    patterns = [r"\([^()]*\)", r"（[^（）]*）"]
    while True:
        previous = name
        for pattern in patterns:
            name = re.sub(pattern, "", name)
        if name == previous:
            break
    return " ".join(name.split()).strip()

if not url.strip() and upload is None:
    st.info("CSV 주소를 입력하거나 파일을 올리면 분석을 시작합니다.")
    st.stop()

try:
    df = read_upload(upload.getvalue()) if upload is not None else read_url(url.strip())
except Exception as exc:
    st.error(f"CSV를 읽지 못했습니다. 주소가 CSV 파일을 직접 가리키는지 확인하세요. ({exc})")
    st.stop()

required = {"날짜", "식사", "메뉴"}
df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
if not required.issubset(df.columns):
    st.error("CSV에 '날짜', '식사', '메뉴' 열이 모두 있어야 합니다.")
    st.stop()

# 한 행 = 하루 식단 한 건. 행 안에서 중복된 메뉴는 한 번만 셉니다.
transactions = []
for raw in df["메뉴"].dropna():
    items = {clean_menu(part) for part in str(raw).split("|")}
    items.discard("")
    if items:
        transactions.append(items)

n = len(transactions)
if n == 0:
    st.warning("분석할 메뉴가 없습니다.")
    st.stop()

item_counts = Counter(item for basket in transactions for item in basket)
pair_counts = Counter(pair for basket in transactions for pair in combinations(sorted(basket), 2))

rules = []
for (a, b), count in pair_counts.items():
    support = count / n
    for antecedent, consequent in ((a, b), (b, a)):
        confidence = count / item_counts[antecedent]
        lift = confidence / (item_counts[consequent] / n)
        rules.append({"조건": antecedent, "결과": consequent, "동시": count,
                      "지지도": support, "신뢰도": confidence, "향상도": lift})

st.write(f"**급식 일수: {n:,}일　|　메뉴 종류 수: {len(item_counts):,}개　|　함께 나온 메뉴 쌍: {len(pair_counts):,}쌍**")

left, right = st.columns(2)
with left:
    chosen = st.selectbox("메뉴 필터", ["전체 메뉴"] + sorted(item_counts), index=0)
with right:
    sort_by = st.selectbox("정렬 기준", ["향상도", "신뢰도", "동시"], index=0)

rules_df = pd.DataFrame(rules, columns=["조건", "결과", "동시", "지지도", "신뢰도", "향상도"])
if chosen != "전체 메뉴":
    rules_df = rules_df[(rules_df["조건"] == chosen) | (rules_df["결과"] == chosen)]

rules_df = rules_df.sort_values([sort_by, "동시", "조건", "결과"], ascending=[False, False, True, True], kind="stable")
st.subheader("메뉴 연관 규칙")
st.dataframe(rules_df, use_container_width=True, hide_index=True,
             column_config={
                 "동시": st.column_config.NumberColumn("동시", format="%d회"),
                 "지지도": st.column_config.NumberColumn("지지도", format="%.2%%"),
                 "신뢰도": st.column_config.NumberColumn("신뢰도", format="%.2%%"),
                 "향상도": st.column_config.NumberColumn("향상도", format="%.3f"),
             })

st.subheader("향상도 상위 10개 규칙")
if rules_df.empty:
    st.info("함께 등장한 메뉴 쌍이 없어 표시할 규칙이 없습니다.")
else:
    top = rules_df.sort_values(["향상도", "신뢰도", "동시"], ascending=False).head(10).copy()
    top["규칙"] = top["조건"] + " → " + top["결과"]
    top = top.iloc[::-1]
    fig = px.bar(top, x="향상도", y="규칙", orientation="h", hover_data=["동시", "지지도", "신뢰도"], text="향상도")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis_title=None, xaxis_title="향상도 (Lift)", margin=dict(l=10, r=35, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.caption("지지도 = 동시 등장 일수 / 전체 급식 일수 · 신뢰도 = 동시 등장 일수 / 조건 메뉴 등장 일수 · 향상도 = 신뢰도 / 결과 메뉴 등장 비율")
