from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st


# =========================================================
# 1) 앱 기본 설정
# =========================================================
st.set_page_config(
    page_title="KOBIS 박스오피스 데일리",
    page_icon="🎬",
    layout="wide",
)


# =========================================================
# 2) 디자인
#    데이터에는 손대지 않고 화면만 깔끔하게 꾸밉니다.
# =========================================================
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        .hero {
            padding: 2rem 2.1rem;
            border-radius: 26px;
            background:
                radial-gradient(circle at 92% 12%, rgba(251, 191, 36, .23), transparent 26%),
                linear-gradient(135deg, #111827 0%, #24101a 52%, #7f1d1d 100%);
            color: white;
            box-shadow: 0 18px 50px rgba(15, 23, 42, .18);
            margin-bottom: 1.1rem;
        }

        .hero-kicker {
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .16em;
            color: #fbbf24;
            margin-bottom: .55rem;
        }

        .hero-title {
            font-size: clamp(2rem, 5vw, 3.6rem);
            line-height: 1.08;
            font-weight: 850;
            letter-spacing: -.045em;
            margin: 0;
        }

        .hero-copy {
            max-width: 720px;
            color: rgba(255, 255, 255, .78);
            font-size: 1rem;
            line-height: 1.65;
            margin-top: .9rem;
            margin-bottom: 0;
        }

        .section-kicker {
            margin-top: 1.8rem;
            margin-bottom: .25rem;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .12em;
            color: #b91c1c;
        }

        .section-title {
            margin-top: 0;
            margin-bottom: .9rem;
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -.03em;
        }

        .micro-note {
            opacity: .72;
            font-size: .88rem;
        }

        /* 지표 카드 */
        div[data-testid="stMetric"] {
            padding: 1.15rem 1.15rem;
            border-radius: 18px;
        }

        div[data-testid="stMetric"] label {
            font-weight: 700;
        }

        /* 날짜 입력 상자 */
        div[data-testid="stDateInput"] > div {
            border-radius: 16px;
        }

        /* 표 모서리를 조금 더 부드럽게 */
        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
        }

        /* 모바일에서는 여백을 조금 줄입니다. */
        @media (max-width: 700px) {
            .block-container {
                padding-top: 1rem;
            }

            .hero {
                padding: 1.45rem;
                border-radius: 20px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


KOBIS_API_URL = (
    "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/"
    "searchDailyBoxOfficeList.json"
)


# =========================================================
# 3) 오류 종류를 나눠서 사용자에게 정확히 안내하기
# =========================================================
class KobisRequestError(Exception):
    """네트워크 문제나 HTTP 오류처럼 요청 자체가 실패한 경우."""


class KobisFaultError(Exception):
    """KOBIS가 HTTP 200과 함께 faultInfo를 반환한 경우."""


class KobisEmptyError(Exception):
    """영화 목록이 빈 배열로 반환된 경우."""


class KobisResponseError(Exception):
    """KOBIS 응답 구조가 예상과 다른 경우."""


# =========================================================
# 4) 한국 시간 기준 날짜 계산
#    Streamlit Cloud 서버의 시간대가 한국이 아니어도 안전합니다.
# =========================================================
def get_korea_today():
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


def get_korea_yesterday():
    return get_korea_today() - timedelta(days=1)


# =========================================================
# 5) KOBIS API 호출
#
#    ttl=3600:
#    같은 날짜 + 같은 인증키의 정상 응답은 약 1시간 동안 캐시합니다.
#    날짜를 바꿨다가 다시 돌아와도 1시간 안이라면 API를 다시 부르지 않습니다.
#
#    빈 영화 목록도 "정상적으로 받아온 응답"이므로 그대로 반환합니다.
#    따라서 빈 목록 역시 1시간 동안 캐시됩니다.
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily_box_office(target_dt: str, api_key: str):
    params = {
        "key": api_key,
        "targetDt": target_dt,
    }

    try:
        response = requests.get(
            KOBIS_API_URL,
            params=params,
            timeout=10,
        )
    except requests.RequestException:
        # 예외 원문에는 인증키가 포함된 URL이 들어갈 수 있으므로
        # 사용자 화면에는 안전한 메시지만 전달합니다.
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

    # KOBIS는 인증키 오류 등이 있어도 상태코드 200을 반환하면서
    # faultInfo 상자를 보낼 수 있으므로 반드시 따로 확인합니다.
    if data.get("faultInfo"):
        fault_info = data["faultInfo"]

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

    return movies


# =========================================================
# 6) 문자열 숫자를 실제 숫자로 변환
#    KOBIS 숫자 필드는 문자열이므로 정렬/그래프 전에 바꿉니다.
# =========================================================
def make_dataframe(movies):
    df = pd.DataFrame(movies)

    required_columns = [
        "rank",
        "rankInten",
        "movieNm",
        "openDt",
        "audiCnt",
        "audiAcc",
        "scrnCnt",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KobisResponseError(
            "KOBIS 응답에 필요한 항목이 빠져 있습니다: "
            + ", ".join(missing_columns)
        )

    # 순위와 전일 대비 순위 증감을 숫자로 변환합니다.
    df["rank"] = pd.to_numeric(
        df["rank"], errors="coerce"
    ).astype("Int64")

    df["rankInten"] = (
        pd.to_numeric(df["rankInten"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )

    # 관객수 / 누적관객 / 스크린수도 실제 숫자로 변환합니다.
    for column in ["audiCnt", "audiAcc", "scrnCnt"]:
        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(0)
            .astype("int64")
        )

    # 공식 순위가 먼저 오도록 정렬합니다.
    df = df.sort_values(
        by=["rank", "audiCnt"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    return df


# =========================================================
# 7) 표시용 도우미 함수
# =========================================================
def format_open_date(value):
    """개봉일을 YYYY-MM-DD 모양으로 정리합니다."""
    if value is None or str(value).strip() == "":
        return "-"

    parsed = pd.to_datetime(str(value), errors="coerce")

    if pd.isna(parsed):
        return str(value)

    return parsed.strftime("%Y-%m-%d")


def format_rank_change(value):
    """
    rankInten > 0 : 전날보다 순위 상승
    rankInten < 0 : 전날보다 순위 하락
    rankInten = 0 : 변동 없음

    표에서는 ▲/▼ 문자를 쓰고 Styler로 각각 빨강/파랑을 칠합니다.
    """
    value = int(value)

    if value > 0:
        return f"▲ +{value}"

    if value < 0:
        return f"▼ {value}"

    return "—"


def style_rank_change(value):
    """순위 상승은 빨강, 하락은 파랑으로 표시합니다."""
    text = str(value)

    if text.startswith("▲"):
        return "color: #dc2626; font-weight: 800;"

    if text.startswith("▼"):
        return "color: #2563eb; font-weight: 800;"

    return "color: #94a3b8; font-weight: 700;"


def trophy_movie_name(movie_name, audience_acc):
    """누적관객이 100만 명을 '넘으면' 영화명 뒤에 트로피를 붙입니다."""
    if int(audience_acc) > 1_000_000:
        return f"{movie_name} 🏆"

    return movie_name


# =========================================================
# 8) 상단 히어로 영역
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">KOBIS · DAILY BOX OFFICE</div>
        <h1 class="hero-title">그날의 극장가를<br>한눈에.</h1>
        <p class="hero-copy">
            날짜를 고르면 KOBIS 일별 박스오피스를 불러와
            순위, 관객 흐름, 누적 성과를 보기 좋게 정리합니다.
            오늘 데이터는 아직 집계 전이므로 선택 가능한 마지막 날짜는 어제입니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 9) 날짜 선택
#    max_value를 한국 시간 기준 '어제'로 지정해 오늘/미래 선택을 막습니다.
# =========================================================
yesterday_kst = get_korea_yesterday()

date_col, info_col = st.columns([1, 1.65], gap="large")

with date_col:
    with st.container(border=True):
        selected_date = st.date_input(
            "📅 박스오피스 날짜",
            value=yesterday_kst,
            max_value=yesterday_kst,
            format="YYYY-MM-DD",
            help="오늘 데이터는 아직 집계 전이므로 어제까지만 선택할 수 있습니다.",
        )

with info_col:
    with st.container(border=True):
        st.markdown("**데이터 읽는 법**")
        st.caption(
            "▲ 빨강 = 전날보다 순위 상승 · "
            "▼ 파랑 = 전날보다 순위 하락 · "
            "🏆 = 누적관객 100만 명 초과"
        )
        st.caption(
            "같은 날짜의 정상 응답은 약 1시간 캐시되어 "
            "불필요한 API 재호출을 줄입니다."
        )

target_dt = selected_date.strftime("%Y%m%d")


# =========================================================
# 10) 인증키 읽기
#     인증키는 코드가 아니라 Streamlit Secrets에서만 읽습니다.
# =========================================================
try:
    kobis_key = st.secrets["KOBIS_KEY"]
except Exception:
    st.error("KOBIS 인증키를 찾지 못했습니다.")
    st.info(
        "Streamlit Community Cloud의 App settings → Secrets에 "
        '`KOBIS_KEY = "발급받은_키"` 형식으로 등록했는지 확인해 주세요. '
        "인증키는 main.py나 GitHub 저장소에 직접 넣지 마세요."
    )
    st.stop()

if not str(kobis_key).strip():
    st.error("KOBIS_KEY가 비어 있습니다.")
    st.info(
        "Streamlit Community Cloud의 Secrets에 "
        "올바른 KOBIS 인증키가 들어 있는지 확인해 주세요."
    )
    st.stop()


# =========================================================
# 11) API 호출 + 오류 처리
# =========================================================
try:
    with st.spinner(
        f"{selected_date.strftime('%Y년 %m월 %d일')} 박스오피스를 불러오는 중..."
    ):
        movies = fetch_daily_box_office(
            target_dt,
            str(kobis_key),
        )

    # 사용자가 요청한 빈 목록 안내 문구입니다.
    if not movies:
        raise KobisEmptyError

    df = make_dataframe(movies)

except KobisEmptyError:
    st.warning("그날은 아직 집계 전입니다")
    st.info(
        "다른 날짜를 선택해 보세요. "
        "KOBIS 쪽에서 해당 날짜의 영화 목록을 빈 배열로 반환했습니다."
    )
    st.stop()

except KobisFaultError as exc:
    st.error("KOBIS가 오류 응답(faultInfo)을 반환했습니다.")
    st.info(
        "Secrets의 KOBIS_KEY가 정확한지, 해당 키가 사용 가능한 상태인지, "
        "그리고 요청 날짜가 올바른지 확인해 주세요."
    )
    st.caption(f"KOBIS 메시지: {exc}")
    st.stop()

except KobisRequestError as exc:
    st.error("KOBIS API 요청에 실패했습니다.")
    st.info(
        "인터넷 연결과 KOBIS 서비스 상태를 확인해 주세요. "
        "인증키 노출을 막기 위해 실제 요청 URL은 화면에 표시하지 않습니다."
    )
    st.caption(str(exc))
    st.stop()

except KobisResponseError as exc:
    st.error("KOBIS 응답 형식을 처리하지 못했습니다.")
    st.info(
        "KOBIS API 응답 구조가 바뀌었거나 일시적인 비정상 응답일 수 있습니다. "
        "공식 API 문서와 서비스 상태를 확인해 주세요."
    )
    st.caption(str(exc))
    st.stop()

except Exception:
    st.error("박스오피스 데이터를 처리하는 중 예상하지 못한 오류가 발생했습니다.")
    st.info(
        "Streamlit 로그, requirements.txt 설치 상태, "
        "그리고 KOBIS_KEY 설정을 확인해 주세요."
    )
    st.stop()


# =========================================================
# 12) 선택한 날짜 정보
# =========================================================
st.markdown(
    '<div class="section-kicker">SELECTED DATE</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"## {selected_date.strftime('%Y년 %m월 %d일')} 박스오피스"
)

st.caption(
    f"KOBIS 일별 박스오피스 · 총 {len(df):,}편 · "
    "숫자 데이터는 실제 숫자형으로 변환해 정렬과 그래프에 사용합니다."
)


# =========================================================
# 13) 1위 영화 지표 카드 3장
# =========================================================
winner = df.iloc[0]
winner_name = trophy_movie_name(
    winner["movieNm"],
    winner["audiAcc"],
)

st.markdown(
    '<div class="section-kicker">NO. 1 MOVIE</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-title">그날의 1위</div>',
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3 = st.columns(3, gap="medium")

with metric_1:
    st.metric(
        "🥇 1위 영화",
        winner_name,
        border=True,
    )

with metric_2:
    st.metric(
        "👥 당일 관객수",
        int(winner["audiCnt"]),
        format="%,d명",
        border=True,
    )

with metric_3:
    st.metric(
        "🏟️ 누적 관객수",
        int(winner["audiAcc"]),
        format="%,d명",
        border=True,
    )

st.caption(
    f"개봉일 {format_open_date(winner['openDt'])} · "
    f"스크린 {int(winner['scrnCnt']):,}개"
)


# =========================================================
# 14) 관객수 상위 5편 막대그래프
#     audiCnt는 이미 실제 숫자형이므로 숫자 크기대로 정확히 정렬됩니다.
# =========================================================
st.markdown(
    '<div class="section-kicker">AUDIENCE TOP 5</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-title">관객을 가장 많이 모은 영화</div>',
    unsafe_allow_html=True,
)

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
    color="primary",
    height=340,
)


# =========================================================
# 15) 전체 순위 표
#
#     - rankInten > 0 : 빨간 ▲
#     - rankInten < 0 : 파란 ▼
#     - 누적관객 > 1,000,000 : 영화명 뒤 🏆
#
#     숫자 칼럼은 문자열로 바꾸지 않습니다.
#     NumberColumn의 표시 형식만 이용하므로 정렬은 계속 숫자 기준입니다.
# =========================================================
st.markdown(
    '<div class="section-kicker">FULL RANKING</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-title">전체 순위</div>',
    unsafe_allow_html=True,
)

table_df = pd.DataFrame(
    {
        "순위": df["rank"],
        "변동": df["rankInten"].map(format_rank_change),
        "영화명": [
            trophy_movie_name(movie_name, audience_acc)
            for movie_name, audience_acc in zip(
                df["movieNm"],
                df["audiAcc"],
            )
        ],
        "개봉일": df["openDt"].map(format_open_date),
        "관객수": df["audiCnt"],
        "누적관객": df["audiAcc"],
        "스크린수": df["scrnCnt"],
    }
)

# 변동 칼럼만 색을 적용합니다.
styled_table = table_df.style.map(
    style_rank_change,
    subset=["변동"],
)

st.dataframe(
    styled_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "순위": st.column_config.NumberColumn(
            "순위",
            format="%d",
            width="small",
        ),
        "변동": st.column_config.TextColumn(
            "전일 대비",
            width="small",
            help="빨간 ▲는 순위 상승, 파란 ▼는 순위 하락입니다.",
        ),
        "영화명": st.column_config.TextColumn(
            "영화명",
            width="large",
            help="🏆 표시는 누적관객 100만 명을 넘은 영화입니다.",
        ),
        "개봉일": st.column_config.TextColumn(
            "개봉일",
            width="medium",
        ),
        "관객수": st.column_config.NumberColumn(
            "관객수",
            format="%,d",
            width="medium",
        ),
        "누적관객": st.column_config.NumberColumn(
            "누적관객",
            format="%,d",
            width="medium",
        ),
        "스크린수": st.column_config.NumberColumn(
            "스크린수",
            format="%,d",
            width="small",
        ),
    },
)

st.caption(
    "ⓘ 같은 날짜의 정상 응답은 약 1시간 동안 캐시됩니다. "
    "오늘과 미래 날짜는 달력에서 선택할 수 없습니다."
)
