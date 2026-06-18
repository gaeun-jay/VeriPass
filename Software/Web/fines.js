/* ===========================================================
   fines.js — VeriPass 과징금 조회 페이지 로직

   읽기 전용 페이지입니다. Firebase에 아무것도 쓰지 않습니다.
   데이터 소스:
   - veripass/logs          : 통과 기록. Count(부정승차 누적횟수) > 0 인 건이 과징금 대상
   - veripass/fare_settings : 운임/할인/배율 설정값

   단건 과징금 = (기본요금 × (1 - 조조할인율)) × fine_multiplier
   누적 과징금 = 단건 과징금 × Count
   조조할인은 세션 Timestamp 시각 기준으로 판단합니다.

   POLL_INTERVAL_MS 주기로 logs를 다시 읽어 누적값을 갱신합니다.
=========================================================== */

const POLL_INTERVAL_MS = 3000;

let logsCache = {};
let fareSettingsCache = null;

/* ---------- DOM 참조 ---------- */
const statCount = document.getElementById('stat-count');
const statCards = document.getElementById('stat-cards');
const statTotalFine = document.getElementById('stat-total-fine');
const cardSummaryBody = document.getElementById('card-summary-body');
const cardSummaryEmpty = document.getElementById('card-summary-empty');
const cardSummaryMeta = document.getElementById('card-summary-meta');
const detailBody = document.getElementById('detail-body');
const detailEmpty = document.getElementById('detail-empty');
const filterGate = document.getElementById('filter-gate');
const filterCardId = document.getElementById('filter-card-id');
const filterCardType = document.getElementById('filter-card-type');
const fareRuleText = document.getElementById('fare-rule-text');

/* ---------- 유틸 ---------- */
function fmtWon(n) {
  return n.toLocaleString('ko-KR') + '원';
}
function fmtDateTime(d) {
  const datePart = d.toLocaleDateString('ko-KR', { month: '2-digit', day: '2-digit' });
  const timePart = [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map((n) => String(n).padStart(2, '0'))
    .join(':');
  return `${datePart} ${timePart}`;
}

/* ---------- logs → 과징금 레코드 (Count > 0 인 부정 건만) ---------- */
function buildFineRecords() {
  if (!fareSettingsCache) return [];

  return Object.keys(logsCache)
    .filter((key) => {
      const v = logsCache[key];
      return v && typeof v === 'object' && v.Timestamp && Number(v.Count) > 0;
    })
    .map((key) => {
      const v = logsCache[key];
      // calcFine은 { card_type, timestamp } 형식을 받으므로 변환
      const calc = calcFine({ card_type: v.Card_Type, timestamp: v.Timestamp }, fareSettingsCache);
      const count = Number(v.Count);
      return {
        id: key,
        gate: v.Gate,
        cardId: v.Card_ID,
        cardType: v.Card_Type,
        cardTypeLabel: CARD_TYPE_LABEL[v.Card_Type] || v.Card_Type,
        name: v.Name || '알 수 없음',
        time: new Date(v.Timestamp),
        count,
        baseFare: calc.baseFare,
        isEarlyBird: calc.isEarlyBird,
        discountRate: calc.discountRate,
        unitFine: calc.fineAmount,           // 단건 과징금
        totalFine: calc.fineAmount * count   // 누적 과징금 = 단건 × Count
      };
    })
    .sort((a, b) => b.time - a.time);
}

/* ---------- 카드별 누적 집계 ---------- */
function buildCardSummary(records) {
  const map = {};
  records.forEach((rec) => {
    if (!map[rec.cardId]) {
      map[rec.cardId] = {
        cardId: rec.cardId,
        name: rec.name,
        cardTypeLabel: rec.cardTypeLabel,
        count: 0,
        totalFine: 0
      };
    }
    map[rec.cardId].count += rec.count;
    map[rec.cardId].totalFine += rec.totalFine;
  });
  return Object.values(map).sort((a, b) => b.totalFine - a.totalFine);
}

/* ---------- 렌더링: 상단 통계 ---------- */
function renderStats(records, cardSummary) {
  const totalCount = records.reduce((sum, r) => sum + r.count, 0);
  const totalFine = records.reduce((sum, r) => sum + r.totalFine, 0);
  statCount.textContent = totalCount.toLocaleString();
  statCards.textContent = cardSummary.length.toLocaleString();
  statTotalFine.textContent = fmtWon(totalFine);
}

/* ---------- 렌더링: 카드별 누적 테이블 ---------- */
function renderCardSummary(cardSummary) {
  cardSummaryBody.innerHTML = '';
  cardSummaryEmpty.style.display = cardSummary.length ? 'none' : 'block';
  cardSummaryMeta.textContent = `${cardSummary.length}명`;

  cardSummary.forEach((c) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-weight:500;">${c.name}</td>
      <td style="color:var(--kr-gray1);">${c.cardId}</td>
      <td>${c.cardTypeLabel}</td>
      <td style="text-align:right;">${c.count}건</td>
      <td style="text-align:right; font-weight:600; color:var(--kr-warn);">${fmtWon(c.totalFine)}</td>
    `;
    cardSummaryBody.appendChild(tr);
  });
}

/* ---------- 렌더링: 상세 이력 테이블 (필터 적용) ---------- */
function renderDetail(records) {
  const gateFilter = filterGate.value;
  const cardIdFilter = filterCardId.value;
  const typeFilter = filterCardType.value;

  const filtered = records.filter(
    (r) =>
      (gateFilter === 'all' || String(r.gate) === gateFilter) &&
      (cardIdFilter === 'all' || r.cardId === cardIdFilter) &&
      (typeFilter === 'all' || r.cardType === typeFilter)
  );

  detailBody.innerHTML = '';
  detailEmpty.style.display = filtered.length ? 'none' : 'block';

  filtered.forEach((r) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${fmtDateTime(r.time)}</td>
      <td style="font-weight:500;">G${String(r.gate).padStart(2, '0')}</td>
      <td>${r.name}</td>
      <td>${r.cardTypeLabel}</td>
      <td style="text-align:right;">${fmtWon(r.baseFare)}</td>
      <td style="text-align:right; color:${r.isEarlyBird ? 'var(--kr-primary)' : 'var(--kr-gray2)'};">
        ${r.isEarlyBird ? `−${Math.round(r.discountRate * 100)}%` : '해당없음'}
      </td>
      <td style="text-align:right;">${r.count}건</td>
      <td style="text-align:right; font-weight:600; color:var(--kr-warn);">${fmtWon(r.totalFine)}</td>
    `;
    detailBody.appendChild(tr);
  });
}

/* ---------- 렌더링: 산정 기준 안내 ---------- */
function renderFareRuleText() {
  if (!fareSettingsCache) return;
  const bf = fareSettingsCache.base_fare;
  const ebd = fareSettingsCache.early_bird_discount;
  const mult = fareSettingsCache.fine_multiplier;

  fareRuleText.innerHTML = `
    단건 과징금 = (기본요금 × (1 − 조조할인율)) × ${mult}배<br>
    누적 과징금 = 단건 과징금 × 부정승차 누적횟수(Count)&nbsp;&nbsp;
    <span style="color:var(--kr-gray2);">(거리별 추가요금은 본 MVP에서 미반영)</span><br><br>
    기본요금 — 일반: ${fmtWon(bf.adult)} · 청소년: ${fmtWon(bf.teen)} · 경로우대(부정사용 시): ${fmtWon(bf.senior)}<br>
    조조할인 — 영업시작 ~ ${String(ebd.end_hour).padStart(2, '0')}:${String(ebd.end_minute).padStart(2, '0')} 탑승 시 기본요금 ${Math.round(ebd.rate * 100)}% 할인 적용
  `;
}

/* ---------- 게이트 필터 옵션 채우기 ---------- */
function populateGateFilter(records) {
  const existing = new Set(Array.from(filterGate.options).map((o) => o.value));
  const gates = [...new Set(records.map((r) => String(r.gate)))].sort();
  gates.forEach((g) => {
    if (!existing.has(g)) {
      const opt = document.createElement('option');
      opt.value = g;
      opt.textContent = 'G' + String(g).padStart(2, '0');
      filterGate.appendChild(opt);
    }
  });
}

/* ---------- 카드 ID 필터 옵션 채우기 ---------- */
function populateCardIdFilter(records) {
  const existing = new Set(Array.from(filterCardId.options).map((o) => o.value));
  const cardIds = [...new Set(records.map((r) => r.cardId))].sort();
  cardIds.forEach((id) => {
    if (!existing.has(id)) {
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = id;
      filterCardId.appendChild(opt);
    }
  });
}

/* ---------- 전체 갱신 ---------- */
function refreshAll() {
  const records = buildFineRecords();
  const cardSummary = buildCardSummary(records);

  populateGateFilter(records);
  populateCardIdFilter(records);
  renderStats(records, cardSummary);
  renderCardSummary(cardSummary);
  renderDetail(records);
}

/* ---------- Firebase 폴링 ---------- */
async function pollData() {
  try {
    const [logs, fareSettings] = await Promise.all([
      fetchLogs(),
      fetchFareSettings()
    ]);
    logsCache = logs;
    fareSettingsCache = fareSettings;

    renderFareRuleText();
    refreshAll();
  } catch (err) {
    console.error('과징금 페이지 데이터 로드 오류:', err);
  }
}

/* ---------- 시계 ---------- */
function updateClock() {
  const n = new Date();
  document.getElementById('clock').textContent = n.toLocaleTimeString('ko-KR', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
}

/* ---------- 초기화 ---------- */
function init() {
  updateClock();
  setInterval(updateClock, 1000);

  filterGate.addEventListener('change', refreshAll);
  filterCardId.addEventListener('change', refreshAll);
  filterCardType.addEventListener('change', refreshAll);

  pollData();
  setInterval(pollData, POLL_INTERVAL_MS);
}

document.addEventListener('DOMContentLoaded', init);
