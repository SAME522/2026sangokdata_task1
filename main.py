"""학교 급식 찾아보기 — 실행: streamlit run main.py (Python 3.10 이상)."""

import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
import html
import json
import re
from zoneinfo import ZoneInfo

import requests
import streamlit as st
import streamlit.components.v1 as components


BASE_URL = "https://open.neis.go.kr/hub/"
ALLERGENS = {
    1: "난류(계란)", 2: "우유", 3: "메밀", 4: "땅콩", 5: "대두(콩)",
    6: "밀", 7: "고등어", 8: "게", 9: "새우", 10: "돼지고기",
    11: "복숭아", 12: "토마토", 13: "아황산류", 14: "호두", 15: "닭고기",
    16: "쇠고기", 17: "오징어", 18: "조개류(굴,전복,홍합 포함)",
}
GOLD_NUMBERS = {10, 15, 16}
ALLERGY_PATTERN = re.compile(r"\(\s*(\d+(?:\s*[.,·]\s*\d+)*[.,·]?\s*)\)")


class NeisError(Exception):
    """조회 실패와 정상적인 조회 결과 없음(INFO-200)을 구분한다."""


def korea_today():
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


def extract_rows(payload, service):
    if not isinstance(payload, dict):
        raise NeisError("급식 서비스의 응답 형식을 확인할 수 없습니다.")
    blocks = payload.get(service, [])
    if not isinstance(blocks, list):
        raise NeisError("급식 서비스의 응답 형식이 예상과 다릅니다.")
    results = [payload.get("RESULT", {})]
    rows, total = [], 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        results.append(block.get("RESULT", {}))
        for header in block.get("head", []):
            results.append(header.get("RESULT", {}))
            if "list_total_count" in header:
                total = int(header["list_total_count"])
        rows.extend(block.get("row", []))
    for result in results:
        code = result.get("CODE")
        if code == "INFO-200":
            return [], 0
        if code and code != "INFO-000":
            raise NeisError(f"나이스 서비스에서 조회를 처리하지 못했습니다. (응답 코드: {code})")
    if not blocks:
        raise NeisError("나이스 서비스에서 올바른 조회 결과를 받지 못했습니다.")
    return rows, total or len(rows)


def request_rows(service, **params):
    try:
        response = requests.get(
            BASE_URL + service,
            params={"Type": "json", **params},
            timeout=(4, 12),
        )
        response.raise_for_status()
        return extract_rows(response.json(), service)
    except requests.RequestException as exc:
        raise NeisError("나이스 서비스에 연결하지 못했습니다. 잠시 후 다시 조회해 주세요.") from exc
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise NeisError("나이스 서비스의 응답을 읽지 못했습니다. 잠시 후 다시 조회해 주세요.") from exc


def expand_school_name(query):
    query = re.sub(r"\s+", "", query.strip())
    if query.endswith("여고"):
        return query[:-2] + "여자고등학교"
    if query.endswith("고"):
        return query[:-1] + "고등학교"
    return query


@st.cache_data(ttl=3600, show_spinner=False, max_entries=512)
def search_schools(query):
    query = query.strip()
    rows, total = request_rows("schoolInfo", SCHUL_NM=query)
    used_query = query
    if not rows:
        expanded = expand_school_name(query)
        if expanded != query:
            rows, total = request_rows("schoolInfo", SCHUL_NM=expanded)
            used_query = expanded
    schools = []
    for row in rows:
        if not all(row.get(key) for key in ("SCHUL_NM", "ATPT_OFCDC_SC_CODE", "SD_SCHUL_CODE")):
            continue
        schools.append({
            "id": f'{row["ATPT_OFCDC_SC_CODE"]}:{row["SD_SCHUL_CODE"]}',
            "name": row["SCHUL_NM"], "region": row.get("LCTN_SC_NM", "지역 미제공"),
            "office": row["ATPT_OFCDC_SC_CODE"], "code": row["SD_SCHUL_CODE"],
        })
    return schools, total, used_query


def parse_menu(raw):
    # HTML 태그는 실행하지 않는다. 숫자로만 된 괄호를 알레르기 표기로 해석한다.
    result = []
    for line in re.split(r"<br\s*/?\s*>|[\r\n]+", html.unescape(raw or ""), flags=re.I):
        plain = re.sub(r"<[^>]*>", "", line).strip()
        if not plain:
            continue
        numbers = sorted({int(n) for match in ALLERGY_PATTERN.finditer(plain)
                          for n in re.findall(r"\d+", match.group(1))})
        name = ALLERGY_PATTERN.sub("", plain).strip()
        if name:
            result.append({"name": name, "allergens": numbers,
                           "gold": bool(GOLD_NUMBERS.intersection(numbers))})
    return result


@st.cache_data(ttl=1800, show_spinner=False, max_entries=10000)
def fetch_day(office, code, ymd):
    # 무인증 조회는 pSize/pIndex를 무시하고 첫 5건만 반환한다.
    # 월 범위를 한 번에 조회하거나 페이지를 반복하면 누락/무한 반복이 발생한다.
    rows, total = request_rows(
        "mealServiceDietInfo", ATPT_OFCDC_SC_CODE=office, SD_SCHUL_CODE=code,
        MMEAL_SC_CODE="2", MLSV_FROM_YMD=ymd, MLSV_TO_YMD=ymd,
        pSize=1000, pIndex=1,
    )
    entries = []
    for row in rows:
        if str(row.get("MLSV_YMD")) != ymd:
            continue
        if str(row.get("MMEAL_SC_CODE", "2")) != "2":
            continue
        entries.append({"items": parse_menu(row.get("DDISH_NM", "")),
                        "calories": row.get("CAL_INFO", "")})
    return {"status": "ok" if entries else "empty", "entries": entries,
            "partial": total > len(rows)}


def load_month(schools, year, month):
    days = [date(year, month, n).strftime("%Y%m%d")
            for n in range(1, calendar.monthrange(year, month)[1] + 1)]
    data = {day: {} for day in days}
    progress = st.progress(0, text="선택한 학교의 급식 달력을 채우고 있어요…")
    # ThreadPool의 작업 함수에서는 Streamlit UI/session_state를 사용하지 않는다.
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(fetch_day, school["office"], school["code"], day):
                (day, school["id"]) for school in schools for day in days}
        for count, future in enumerate(as_completed(jobs), 1):
            day, school_id = jobs[future]
            try:
                data[day][school_id] = future.result()
            except NeisError as exc:
                # 실패 결과는 캐시하지 않아 다음 조회에서 재시도한다.
                data[day][school_id] = {"status": "error", "message": str(exc), "entries": []}
            progress.progress(count / len(jobs), text=f"급식 달력 준비 중 · {count}/{len(jobs)}")
    progress.empty()
    return data


HTML_TEMPLATE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--ink:#193d32;--muted:#6b7b73;--green:#21694e;--paper:#fffdf7;--line:#dfe7dd}
*{box-sizing:border-box}body{margin:0;padding:8px 6px 24px;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,'Malgun Gothic',sans-serif}
button{font:inherit;cursor:pointer}button:focus-visible{outline:3px solid #ea9a27;outline-offset:3px}
.arena{height:146px;overflow:hidden;position:relative;display:flex;align-items:center;justify-content:center;border-radius:24px;background:#e9efe3;margin-bottom:26px}
.arena:before{content:'';position:absolute;width:170px;height:170px;border:1px solid #d4decd;border-radius:50%}
.teams{width:min(880px,94%);display:grid;grid-template-columns:1fr 44px 1fr;align-items:center;gap:0;position:relative}
.team{padding:16px 14px;border:1px solid #d2dfd0;background:#fffef9;border-radius:16px;text-align:center;overflow-wrap:anywhere;box-shadow:0 5px 0 #cbd8c8}
.base{background:#21694e;color:white;border-color:#21694e}.team small{display:block;font-size:10px;letter-spacing:2px;margin-bottom:7px;opacity:.75}.team strong{font-size:clamp(12px,2vw,18px)}
.opponents{display:grid;gap:6px}.opponents .team{padding:8px 12px;border-color:#e1cabc;background:#fcece0;box-shadow:0 3px 0 #e9d8c6}.opponents small{margin-bottom:2px}
.vs{text-align:center;font-weight:900;font-size:14px;z-index:2}.impact{position:absolute;left:50%;top:50%;font-size:64px;opacity:0;transform:translate(-50%,-50%);pointer-events:none}
.battle .base{animation:flyLeft .75s cubic-bezier(.2,.7,.3,1) both}.battle .opponents{animation:flyRight .75s cubic-bezier(.2,.7,.3,1) both}.battle .impact{animation:impact .55s .42s both}
@keyframes flyLeft{0%{transform:translateX(-110vw) rotate(-9deg)}65%{transform:translateX(24px) rotate(3deg)}82%{transform:translateX(-12px)}100%{transform:none}}
@keyframes flyRight{0%{transform:translateX(110vw) rotate(9deg)}65%{transform:translateX(-24px) rotate(-3deg)}82%{transform:translateX(12px)}100%{transform:none}}
@keyframes impact{0%{opacity:0;transform:translate(-50%,-50%) scale(.3)}30%{opacity:1;transform:translate(-50%,-50%) scale(1.4)}100%{opacity:0;transform:translate(-50%,-50%) scale(1.8)}}
.heading{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:0 4px 20px}.eyebrow{font-size:11px;letter-spacing:3px;color:var(--muted)}h2{font-size:32px;letter-spacing:-1px;margin:6px 0}.hint{font-size:13px;color:var(--muted);line-height:1.7}.tag{background:#e9efe3;padding:9px 13px;border-radius:30px;font-size:12px;white-space:nowrap}
.week,.calendar{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:7px}.week{text-align:center;font-size:12px;color:var(--muted);margin-bottom:12px}.week span:first-child{color:#bb6350}.week span:last-child{color:#477ca6}
.day{min-height:98px;text-align:left;background:#fffefb;border:1px solid var(--line);border-radius:14px;padding:13px;display:flex;flex-direction:column;gap:12px;transition:transform .18s,box-shadow .18s;position:relative;color:var(--ink)}
.day:hover{transform:translateY(-3px);box-shadow:0 6px 14px #254a3212;border-color:#8cb39a}.day.today{border:2px solid var(--green);background:#eff6e8}.day.selected{box-shadow:inset 0 0 0 2px #85a98a}.day .num{font-size:19px;font-weight:700}.day .status{font-size:11px;color:var(--muted)}.day.sun .num{color:#bb6350}.day.sat .num{color:#477ca6}.todayLabel{font-size:10px;position:absolute;right:10px;top:17px;color:var(--green)}
.dots{display:flex;gap:5px;flex-wrap:wrap}.dot{height:6px;width:6px;border-radius:50%;background:#32795a}.dot.other{background:#d4976e}.empty{background:#f3f4ed;border-radius:14px;opacity:.65}.legend{margin:18px 4px;font-size:12px;color:var(--muted);line-height:1.8}
dialog{padding:0;border:1px solid #d5dfd1;border-radius:24px;background:var(--paper);color:var(--ink);width:min(1100px,96vw);max-height:92vh;box-shadow:0 24px 90px #133a3440;overflow:auto}dialog::backdrop{background:#193d3288;backdrop-filter:blur(4px)}
.dialogHead{position:sticky;top:0;z-index:3;background:var(--paper);display:flex;align-items:center;justify-content:space-between;padding:20px 24px;border-bottom:1px solid var(--line)}.dialogHead h2{font-size:24px;margin:0}.close{border:1px solid var(--line);border-radius:50%;background:white;width:36px;height:36px;font-size:22px}.dialogHint{padding:0 24px;margin:16px 0;font-size:12px;color:var(--muted)}
.cards{display:grid;gap:14px;padding:0 24px 24px;grid-template-columns:repeat(var(--count),minmax(0,1fr));align-items:stretch}.mealCard{background:white;border:1px solid var(--line);border-radius:18px;display:flex;flex-direction:column;overflow:hidden;min-width:0}.schoolHead{padding:16px;border-bottom:1px solid var(--line);background:#f6eade}.mealCard:first-child .schoolHead{background:#e5efdf}.schoolHead small{font-size:11px;color:var(--muted)}.schoolHead h3{font-size:16px;line-height:1.5;margin:6px 0}.menuList{list-style:none;padding:14px;margin:0;display:flex;flex-direction:column;gap:10px;flex:1}.dish{position:relative;min-height:57px;border:1px solid #e7e9df;border-radius:11px;display:grid;place-items:center;padding:12px 10px;overflow:visible}.dishName{visibility:hidden;opacity:0;font-size:14px;text-align:center;line-height:1.55;overflow-wrap:anywhere}.dish.revealed .dishName{visibility:visible;opacity:1;transition:opacity .25s}.dishName small{display:block;color:#728176;font-size:11px;margin-top:3px}.cover{position:absolute;inset:0;border:1px solid #9dac8e;border-radius:10px;background:repeating-linear-gradient(135deg,#e7eddd 0px,#e7eddd 12px,#dfe7d3 12px,#dfe7d3 24px);color:#466440;font-weight:700;box-shadow:0 3px 0 #c8d3bd;font-size:13px}.cover.gold{color:#755009;border-color:#ddb03f;background:linear-gradient(115deg,#f7d271,#fff1a8,#eeb94e,#fff1a8);background-size:250% 100%;animation:gold 2s ease-in-out infinite;box-shadow:0 0 16px #e9ba4e88,0 3px 0 #d7aa41}.cover.opening{animation:shake .48s ease-in-out forwards!important;pointer-events:none}.cover:disabled{opacity:1}
@keyframes gold{0%,100%{background-position:0% 50%;box-shadow:0 0 10px #e9ba4e70}50%{background-position:100% 50%;box-shadow:0 0 22px #efbd43bb}}
@keyframes shake{0%{transform:translateX(0)}12%{transform:translateX(-6px) rotate(-3deg)}25%{transform:translateX(6px) rotate(3deg)}40%{transform:translateX(-6px) rotate(-3deg)}55%{transform:translateX(5px) rotate(2deg)}72%{transform:scale(1.05);opacity:1}100%{transform:translateY(-9px) scale(1.12);opacity:0}}
.allergy{margin:0 14px 14px;border-top:2px solid #bbcab6;padding-top:13px;font-size:11px;line-height:1.9;color:#64745e}.allergy strong{color:#284b32;display:block;margin-bottom:6px}.allergy span{display:inline-block;margin-right:7px}.message{padding:20px 16px;color:var(--muted);font-size:13px;line-height:1.8}.calories{font-size:11px;margin:4px 14px 12px;color:var(--muted)}.entryLabel{font-size:11px;color:var(--muted);padding:4px}
@media(max-width:720px){.cards{grid-template-columns:repeat(2,minmax(0,1fr));padding:0 14px 14px}.day{padding:9px;min-height:82px}.day .num{font-size:16px}.day .status{font-size:10px}.todayLabel{position:static;font-size:9px}.calendar,.week{gap:4px}.heading h2{font-size:26px}.tag{display:none}}
@media(max-width:450px){.cards{grid-template-columns:1fr}.day .status{font-size:9px}.day{min-height:88px;padding:7px}.arena{height:170px}.team{padding:12px 7px}.heading .hint{font-size:11px}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;transition:none!important}.cover.gold{box-shadow:0 0 18px #e9ba4e99}}
</style></head><body>
<section class="arena" id="arena" aria-label="선택한 학교"><div class="teams" id="teams"></div><span class="impact" aria-hidden="true">💥</span></section>
<div class="heading"><div><div class="eyebrow">SCHOOL LUNCH CLUB</div><h2 id="monthTitle"></h2><div class="hint">궁금한 날짜를 누르고, 오늘의 메뉴 상자를 열어 보세요.</div></div><span class="tag">🍱 중식 · 한국 시간 기준</span></div>
<div class="week" aria-hidden="true"><span>일</span><span>월</span><span>화</span><span>수</span><span>목</span><span>금</span><span>토</span></div>
<div class="calendar" id="calendar" aria-label="급식 달력"></div>
<p class="legend">🟢 기준 학교 · 🟠 비교 학교 &nbsp; | &nbsp; 점은 급식 정보가 있는 학교를 뜻해요.<br>✨ 황금 상자: 10 돼지고기 · 15 닭고기 · 16 쇠고기 중 하나 이상이 표시된 메뉴</p>
<dialog id="detail" aria-labelledby="detailTitle"><div class="dialogHead"><h2 id="detailTitle"></h2><button class="close" id="close" aria-label="달력으로 돌아가기">×</button></div><p class="dialogHint">상자를 하나씩 누르면 흔들린 뒤 메뉴가 나타나요. 알레르기 정보는 각 학교의 아래쪽에서 확인하세요.</p><div class="cards" id="cards"></div></dialog>
<script>
const DATA = __PAYLOAD__;
const $ = id => document.getElementById(id);
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function node(tag, cls, text) {const el=document.createElement(tag);if(cls)el.className=cls;if(text!==undefined)el.textContent=text;return el;}
const teams=$('teams');
const base=node('div','team base');base.append(node('small','','기준 학교'),node('strong','',DATA.schools[0].name));teams.append(base,node('div','vs',DATA.schools.length>1?'VS':'🍱'));
const opponents=node('div','opponents');
DATA.schools.slice(1).forEach((s,i)=>{const card=node('div','team');card.append(node('small','',`비교 학교 ${i+1}`),node('strong','',s.name));opponents.append(card);});
if(DATA.schools.length===1)opponents.append(node('div','hint','비교 학교를 추가해 함께 살펴보세요.'));
teams.append(opponents);if(DATA.animate)$('arena').classList.add('battle');
$('monthTitle').textContent=`${DATA.year}년 ${DATA.month}월`;
const grid=$('calendar'), dialog=$('detail');let activeButton=null;
const offset=new Date(DATA.year,DATA.month-1,1).getDay();
for(let i=0;i<offset;i++)grid.append(node('div','empty'));
const dateKeys=Object.keys(DATA.days).sort();
dateKeys.forEach(ymd=>{
 const n=Number(ymd.slice(-2)), weekday=new Date(DATA.year,DATA.month-1,n).getDay();
 const btn=node('button',`day ${weekday===0?'sun':weekday===6?'sat':''}`);
 btn.type='button';btn.dataset.ymd=ymd;btn.setAttribute('aria-haspopup','dialog');
 btn.append(node('span','num',String(n)));
 if(ymd===DATA.today){btn.classList.add('today');btn.append(node('span','todayLabel','오늘'));}
 if(ymd===DATA.selected)btn.classList.add('selected');
 let count=0,failed=false;const dots=node('div','dots');
 DATA.schools.forEach((school,i)=>{const meal=DATA.days[ymd][school.id];if(meal.status==='ok'){count++;const dot=node('span',i===0?'dot':'dot other');dot.title=school.name;dots.append(dot);}if(meal.status==='error')failed=true;});
 const status=failed?'일부 조회 실패':count?`${count}개 학교 급식`:'급식 정보 없음';
 btn.append(node('span','status',status),dots);btn.setAttribute('aria-label',`${DATA.month}월 ${n}일, ${status}, 메뉴 보기`);
 btn.addEventListener('click',()=>openDay(ymd,btn));grid.append(btn);
});
while(grid.children.length%7)grid.append(node('div','empty'));
function openDay(ymd,btn){
 activeButton=btn;grid.querySelectorAll('.selected').forEach(el=>el.classList.remove('selected'));btn.classList.add('selected');
 const origin=btn.getBoundingClientRect();$('detailTitle').textContent=`${DATA.month}월 ${Number(ymd.slice(-2))}일의 점심`;
 const cards=$('cards');cards.replaceChildren();cards.style.setProperty('--count',DATA.schools.length);
 DATA.schools.forEach((school,index)=>{
  const meal=DATA.days[ymd][school.id], card=node('section','mealCard'), head=node('div','schoolHead');
  head.append(node('small','',index===0?'기준 학교':`비교 학교 ${index}`),node('h3','',school.name),node('small','',school.region));card.append(head);
  if(meal.status!=='ok'){card.append(node('p','message',meal.status==='error'?meal.message:'이날은 등록된 중식 정보가 없어요.'));cards.append(card);return;}
  const list=node('ul','menuList'), allNumbers=new Set();
  meal.entries.forEach((entry,entryIndex)=>{
   if(meal.entries.length>1)list.append(node('li','entryLabel',`중식 식단 ${entryIndex+1}`));
   if(!entry.items.length)list.append(node('li','message','메뉴 상세 정보가 제공되지 않았어요.'));
   entry.items.forEach((item,i)=>{
    item.allergens.forEach(n=>allNumbers.add(n));
    const li=node('li','dish'), name=node('div','dishName',item.name);name.setAttribute('aria-hidden','true');
    if(item.allergens.length)name.append(node('small','',`알레르기 ${item.allergens.join(', ')}`));
    const cover=node('button',`cover${item.gold?' gold':''}`,`${item.gold?'✨':'📦'} 메뉴 ${i+1} 열기`);cover.type='button';
    cover.setAttribute('aria-label',`${school.name} 메뉴 ${i+1} 공개${item.gold?', 황금 상자':''}`);
    cover.addEventListener('click',()=>{
     cover.disabled=true;cover.classList.add('opening');
     setTimeout(()=>{cover.remove();li.classList.add('revealed');name.removeAttribute('aria-hidden');name.setAttribute('role','status');li.tabIndex=-1;li.focus({preventScroll:true});},reduced?0:490);
    },{once:true});
    li.append(name,cover);list.append(li);
   });
  });card.append(list);
  const calories=meal.entries.map((e,i)=>e.calories?`${meal.entries.length>1?`식단 ${i+1} · `:''}${e.calories}`:'').filter(Boolean);
  if(calories.length)card.append(node('p','calories',calories.join(' / ')));
  const allergy=node('footer','allergy');allergy.append(node('strong','','알레르기 번호 · 이름'));
  if(!allNumbers.size)allergy.append(node('span','','제공된 메뉴에 알레르기 번호 표기가 없습니다.'));
  [...allNumbers].sort((a,b)=>a-b).forEach(n=>allergy.append(node('span','',`${n}. ${DATA.allergens[n]||'명칭 미제공'}`)));
  if(meal.partial)allergy.append(node('p','','이날 식단이 5건을 초과해 일부 정보만 표시됩니다.'));
  card.append(allergy);cards.append(card);
 });
 dialog.showModal();dialog.scrollTop=0;
 if(!reduced){const end=dialog.getBoundingClientRect();dialog.animate([
  {transform:`translate(${origin.left+origin.width/2-end.left-end.width/2}px,${origin.top+origin.height/2-end.top-end.height/2}px) scale(${origin.width/end.width},${origin.height/end.height})`,opacity:.4},
  {transform:'translate(0,0) scale(1,1)',opacity:1}
 ],{duration:340,easing:'cubic-bezier(.2,.8,.2,1)'});}
}
$('close').onclick=()=>dialog.close();
dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
dialog.addEventListener('close',()=>activeButton?.focus({preventScroll:true}));
</script></body></html>'''


def render_calendar(schools, data, selected, animate):
    payload = {"schools": schools, "days": data, "year": selected.year,
               "month": selected.month, "selected": selected.strftime("%Y%m%d"),
               "today": korea_today().strftime("%Y%m%d"), "allergens": ALLERGENS,
               "animate": animate}
    # 외부 문자열은 JSON으로만 전달하고 script 종료 태그 삽입을 차단한다.
    safe_json = (json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
                 .replace(">", "\\u003e").replace("&", "\\u0026")
                 .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    components.html(HTML_TEMPLATE.replace("__PAYLOAD__", safe_json), height=1060, scrolling=True)


def main():
    st.set_page_config(page_title="학교 급식 찾아보기", page_icon="🍱", layout="wide")
    st.markdown("""<style>
    .stApp{background:#fffdf7;color:#193d32}
    .block-container{max-width:1280px;padding-top:2.5rem}
    h1{letter-spacing:-1.5px}div[data-testid="stForm"]{background:#f1f4ea;border-radius:18px}
    </style>""", unsafe_allow_html=True)
    st.caption("🍱  SCHOOL LUNCH CLUB")
    st.title("학교 급식 찾아보기")
    st.write("우리 학교의 점심, 다른 학교와 함께 열어 볼까요?")
    st.session_state.setdefault("school_registry", {})
    st.session_state.setdefault("comparisons", [])
    st.session_state.setdefault("view_date", korea_today())
    with st.form("school_search"):
        query = st.text_input("학교 이름", placeholder="예: 수도여고, 경기고, 서울교육대학교부설초등학교")
        submitted = st.form_submit_button("🔎 학교 찾기")
    if submitted:
        if not query.strip():
            st.info("검색할 학교 이름을 입력해 주세요.")
        else:
            try:
                with st.spinner("학교를 찾고 있어요…"):
                    found, total, used = search_schools(query)
                if used != query.strip():
                    st.info(f"‘{query.strip()}’의 검색 결과가 없어 ‘{used}’로 다시 검색했어요.")
                for school in found:
                    st.session_state.school_registry[school["id"]] = school
                if not found:
                    st.info("학교를 찾지 못했어요. 학교의 정식 이름이나 더 구체적인 이름으로 검색해 주세요.")
                else:
                    st.success(f"{len(found)}개 학교를 찾았어요. 아래에서 기준 학교와 비교 학교를 선택해 주세요.")
                    if total > len(found) or len(found) >= 5:
                        st.caption("인증키 없는 검색은 최대 5개 학교가 표시돼요. 원하는 학교가 없다면 이름을 더 자세히 입력해 주세요.")
            except NeisError as exc:
                st.info(str(exc))
    registry = st.session_state.school_registry
    if not registry:
        st.info("학교를 검색하면 기준 학교 1개와 비교 학교 최대 3개를 선택할 수 있어요.")
        return

    def school_label(school_id):
        school = registry[school_id]
        return f'{school["name"]} · {school["region"]} ({school["code"]})'

    left, right = st.columns([1, 2])
    with left:
        base = st.selectbox("기준 학교 · 1개", list(registry), index=None,
                            format_func=school_label, placeholder="기준 학교를 선택하세요", key="base_school")
    # 기준 학교로 바꾼 학교가 기존 비교 목록에 있으면 중복을 제거한다.
    st.session_state.comparisons = [sid for sid in st.session_state.comparisons if sid != base and sid in registry]
    with right:
        comparisons = st.multiselect("비교 학교 · 최대 3개", [sid for sid in registry if sid != base],
                                     format_func=school_label, max_selections=3, key="comparisons",
                                     placeholder="다른 학교를 추가로 검색해 선택하세요")
    st.caption("다른 학교를 추가 검색해도 선택한 학교는 유지됩니다.")
    selected = st.date_input("조회할 날짜 · 해당 월의 달력을 표시해요", key="view_date",
                             min_value=date(2000, 1, 1), max_value=date(2100, 12, 31), format="YYYY-MM-DD")
    if not base:
        st.info("기준 학교를 하나 선택해 주세요.")
        return
    schools = [registry[sid] for sid in [base, *comparisons]]
    signature = (base, *comparisons)
    animate = st.session_state.get("last_school_selection") != signature
    data = load_month(schools, selected.year, selected.month)
    failures = sum(value["status"] == "error" for day in data.values() for value in day.values())
    if failures:
        st.info("일부 날짜의 정보를 가져오지 못했어요. 해당 날짜 칸에서 안내를 확인하고 다시 조회해 주세요.")
        if st.button("가져오지 못한 급식 다시 조회"):
            st.rerun()
    render_calendar(schools, data, selected, animate)
    st.session_state.last_school_selection = signature
    st.caption("자료: 나이스 교육정보 개방 포털 · 중식 기준 · 급식 정보는 30분 동안 캐시됩니다.")


if __name__ == "__main__":
    main()
