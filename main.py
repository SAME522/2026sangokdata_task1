from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="어제의 박스오피스",
    page_icon="🎬",
    layout="wide",
)

KOBIS_API_URL = (
    "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/"
    "searchDailyBoxOfficeList.json"
)


# -----------------------------
# KOBIS API에서 발생한 상황을 구분하기 위한 예외
# -----------------------------
class KobisRequestError(Exception):
    """네트워크 오류나 HTTP 오류처럼 요청 자체가 실패했을 때 사용합니다."""


class KobisFaultError(Exception):
    """KOBIS가 HTTP 200과 함께 faultInfo를 돌려줬을 때 사용합니다."""


class KobisEmptyError(Exception):
    """응답은 정상처럼 보이지만 영화 목록이 비어 있을 때 사용합니다."""


class KobisResponseError(Exception):
    """JSON 형식이나 응답 구조가 예상과 다를 때 사용합니다."""


# -----------------------------
# 한국 시간 기준으로 '어제' 계산
# 서버가 UTC 등 다른 시간대를 쓰더라도 항상 한국 날짜를 기준으로 합니다.
# -----------------------------
def get_yesterday_in_korea():
    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    return now_kst.date() - timedelta(days=1)


# -----------------------------
# KOBIS 일별 박스오피스 조회
#
# ttl=3600:
# 같은 날짜의 성공한 결과를 약 1시간 동안 기억합니다.
# Streamlit이 다시 실행되어도 그동안은 API를 다시 호출하지 않습니다.
#
# _api_key:
# 앞에 밑줄(_)이 붙은 인자는 Streamlit 캐시 키 계산에서 제외됩니다.
# 따라서 실제 캐시 기준은 target_dt(조회 날짜)입니다.
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily_box_office(target_dt: str, _api_key: str):
    params = {
        "key": _api_key,
        "targetDt": target_dt,
    }

    try:
        response = requests.get(
            KOBIS_API_URL,
            params=params,
            timeout=10,
        )
    except requests.RequestException:
        # requests 예외 메시지에는 요청 URL과 인증키가 포함될 수 있으므로
        # 원문을 화면에 그대로 노출하지 않습니다.
        raise KobisRequestError(
            "KOBIS 서버에 연결하지 못했습니다."
        ) from None

    if response.status_code != 200:
        raise KobisRequestError(
            f"KOBIS API가 HTTP {response.status_code} 상태를 반환했습니다."
        )

    try:
        data = response.json()
    except ValueError:
        raise KobisResponseError(
            "KOBIS 응답을 JSON으로 해석하지 못했습니다."
        ) from None

    # KOBIS는 인증키 오류 등이 있어도 HTTP 상태코드 200을 반환하면서
    # 최상위에 faultInfo를 넣어 줄 수 있으므로 반드시 따로 확인합니다.
    if data.get("faultInfo"):
        fault_info = data["faultInfo"]

        # 화면에 보여 줄 수 있는 짧은 오류 설명만 꺼냅니다.
        if isinstance(fault_info, dict):
            fault_message = (
                fault_info.get("message")
                or fault_info.get("faultString")
                or "인증키 또는 요청값 관련 오류가 반환되었습니다."
            )
        else:
            fault_message = "인증키 또는 요청값 관련 오류가 반환되었습니다."

        raise KobisFaultError(str(fault_message))

    box_office_result = data.get("boxOfficeResult")
    if not isinstance(box_office_result, dict):
        raise KobisResponseError(
            "응답에서 boxOfficeResult를 찾지 못했습니다."
        )

    movies = box_office_result.get("dailyBoxOfficeList")
    if not isinstance(movies, list):
        raise KobisResponseError(
            "응답에서 dailyBoxOfficeList를 찾지 못했습니다."
        )

    if not movies:
        raise KobisEmptyError(
            "해당 날짜의 일별 박스오피스 영화 목록이 비어 있습니다."
        )

    return movies


# -----------------------------
# 문자열로 온 숫자를 실제 숫자로 변환
# 정렬, 지표 계산, 그래프에는 이 숫자형 데이터를 사용합니다.
# -----------------------------
def make_dataframe(movies):
    df = pd.DataFrame(movies)

    required_columns = [
        "rank",
        "movieNm",
        "openDt",
        "audiCnt",
        "audiAcc",
        "scrnCnt",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KobisResponseError(
            "KOBIS 응답에 필요한 항목이 빠져 있습니다: "
            + ", ".join(missing_columns)
        )

    # 순위는 결측값이 있더라도 표시할 수 있도록 nullable integer를 사용합니다.
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")

    # 관객수/누적관객/스크린수는 숫자로 바꿔 정렬과 그래프에 사용합니다.
    for column in ["audiCnt", "audiAcc", "scrnCnt"]:
        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(0)
            .astype("int64")
        )

    # 공식 순위를 기준으로 표를 정렬합니다.
    df = df.sort_values(
        by=["rank", "audiCnt"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    return df


# -----------------------------
# 화면 표시용 도우미
# -----------------------------
def format_number(value):
    """숫자를 12,345처럼 보기 좋게 표시합니다."""
    if pd.isna(value):
        return "-"
    return f"{int(value):,}"


def format_open_date(value):
    """개봉일 문자열을 YYYY-MM-DD 모양으로 정리합니다."""
    if value is None or str(value).strip() == "":
        return "-"

    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        return str(value)

    return parsed.strftime("%Y-%m-%d")


# -----------------------------
# 앱 본문
# -----------------------------
st.title("🎬 어제의 박스오피스")

yesterday_kst = get_yesterday_in_korea()
target_dt = yesterday_kst.strftime("%Y%m%d")

st.caption(
    f"조회 기준: 한국 시간(KST) {yesterday_kst.strftime('%Y년 %m월 %d일')} · "
    "KOBIS 일별 박스오피스"
)

# 인증키는 코드에 넣지 않고 Streamlit Secrets에서만 읽습니다.
try:
    kobis_key = st.secrets["KOBIS_KEY"]
except Exception:
    st.error("KOBIS 인증키를 찾지 못했습니다.")
    st.info(
        "Streamlit Community Cloud의 App settings → Secrets에 "
        '`KOBIS_KEY = "발급받은_키"` 형식으로 등록했는지 확인해 주세요. '
        "인증키를 main.py나 GitHub 저장소에 직접 넣으면 안 됩니다."
    )
    st.stop()

if not str(kobis_key).strip():
    st.error("KOBIS_KEY가 비어 있습니다.")
    st.info(
        "Streamlit Community Cloud의 Secrets에 올바른 KOBIS 인증키가 "
        "들어 있는지 확인해 주세요."
    )
    st.stop()

try:
    with st.spinner("어제의 박스오피스를 불러오는 중입니다..."):
        movies = fetch_daily_box_office(target_dt, str(kobis_key))

    df = make_dataframe(movies)

except KobisFaultError as exc:
    st.error("KOBIS가 오류 응답(faultInfo)을 반환했습니다.")
    st.info(
        "Secrets의 KOBIS_KEY가 정확한지, 해당 키가 사용 가능한 상태인지, "
        "그리고 조회 날짜가 yyyymmdd 형식인지 확인해 주세요."
    )
    st.caption(f"KOBIS 메시지: {exc}")
    st.stop()

except KobisEmptyError:
    st.error("해당 날짜의 박스오피스 영화 목록이 비어 있습니다.")
    st.info(
        "KOBIS의 전일 집계가 아직 준비되지 않았거나 일시적으로 데이터가 "
        "제공되지 않는 경우가 있습니다. KOBIS 서비스 상태와 조회 날짜를 "
        "확인한 뒤 다시 시도해 주세요."
    )
    st.stop()

except KobisRequestError as exc:
    st.error("KOBIS API 요청에 실패했습니다.")
    st.info(
        "인터넷 연결, KOBIS 서비스 상태, 요청 주소가 정상인지 확인해 주세요. "
        "인증키가 화면이나 로그에 노출되지 않도록 원래 요청 URL은 표시하지 않습니다."
    )
    st.caption(str(exc))
    st.stop()

except KobisResponseError as exc:
    st.error("KOBIS 응답 형식을 처리하지 못했습니다.")
    st.info(
        "KOBIS API 응답 구조가 바뀌었거나 일시적인 비정상 응답일 수 있습니다. "
        "공식 API 문서와 KOBIS 서비스 상태를 확인해 주세요."
    )
    st.caption(str(exc))
    st.stop()

except Exception:
    st.error("박스오피스 데이터를 처리하는 중 예상하지 못한 오류가 발생했습니다.")
    st.info(
        "Streamlit 로그를 확인하고, requirements.txt가 설치되었는지와 "
        "KOBIS_KEY 설정이 올바른지 확인해 주세요."
    )
    st.stop()


# 데이터가 정상적으로 준비된 뒤부터 화면을 그립니다.
if df.empty:
    # 위에서 빈 목록을 막았지만, 혹시 변환 과정에서 비게 되는 상황도 방어합니다.
    st.error("표시할 박스오피스 데이터가 없습니다.")
    st.info("KOBIS 응답 내용과 조회 날짜를 확인해 주세요.")
    st.stop()


# -----------------------------
# 1위 영화: 큰 지표 카드 3장
# -----------------------------
winner = df.iloc[0]

st.subheader("🏆 1위 영화")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("영화", winner["movieNm"], border=True)

with col2:
    st.metric("어제 관객수", f"{winner['audiCnt']:,}명", border=True)

with col3:
    st.metric("누적 관객수", f"{winner['audiAcc']:,}명", border=True)


# -----------------------------
# 관객수 상위 5편 막대그래프
# 숫자형 audiCnt를 기준으로 내림차순 정렬합니다.
# -----------------------------
st.subheader("📊 관객수 상위 5편")

top5 = (
    df.sort_values("audiCnt", ascending=False)
    .head(5)
    [["movieNm", "audiCnt"]]
    .copy()
)

st.bar_chart(
    top5,
    x="movieNm",
    y="audiCnt",
    x_label="영화명",
    y_label="관객수",
    horizontal=True,
    sort="-audiCnt",
    use_container_width=True,
)


# -----------------------------
# 전체 순위 표
# 내부 데이터는 숫자형으로 유지하고,
# 표에 보여 줄 때만 쉼표가 있는 문자열로 복사해 표시합니다.
# -----------------------------
st.subheader("📋 전체 순위")

table_df = pd.DataFrame(
    {
        "순위": df["rank"].map(
            lambda x: "-" if pd.isna(x) else str(int(x))
        ),
        "영화명": df["movieNm"],
        "개봉일": df["openDt"].map(format_open_date),
        "관객수": df["audiCnt"].map(format_number),
        "누적관객": df["audiAcc"].map(format_number),
        "스크린수": df["scrnCnt"].map(format_number),
    }
)

st.dataframe(
    table_df,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "성공한 API 결과는 조회 날짜별로 약 1시간 캐시됩니다."
)
