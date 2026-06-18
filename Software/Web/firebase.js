/* ===========================================================
   firebase.js — VeriPass 공통 Firebase 연동 모듈

   Main.html(관제 대시보드)과 Fines.html(과징금 페이지)이
   공통으로 사용하는 Firebase REST API 헬퍼 모음입니다.

   SDK 설치 없이 fetch() 기반 REST 호출만 사용합니다.
=========================================================== */

const FIREBASE_DB_URL = 'https://veripass-21ab9-default-rtdb.firebaseio.com';

/* ---------- 공통 GET 헬퍼 ---------- */
async function fbGet(path) {
  const res = await fetch(`${FIREBASE_DB_URL}/${path}.json`);
  if (!res.ok) throw new Error(`Firebase GET 실패: ${path} (${res.status})`);
  return res.json();
}

/* ---------- 공통 쓰기 헬퍼 ----------
   fbPut    : 해당 경로를 통째로 덮어씀 (PUT)
   fbPatch  : 일부 필드만 병합 갱신 (PATCH)
   fbDelete : 해당 경로 삭제 (DELETE)
   웹 → Firebase 방향 동기화에 사용합니다.
*/
async function fbPut(path, data) {
  const res = await fetch(`${FIREBASE_DB_URL}/${path}.json`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  if (!res.ok) throw new Error(`Firebase PUT 실패: ${path} (${res.status})`);
  return res.json();
}

async function fbPatch(path, data) {
  const res = await fetch(`${FIREBASE_DB_URL}/${path}.json`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  if (!res.ok) throw new Error(`Firebase PATCH 실패: ${path} (${res.status})`);
  return res.json();
}

async function fbDelete(path) {
  const res = await fetch(`${FIREBASE_DB_URL}/${path}.json`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`Firebase DELETE 실패: ${path} (${res.status})`);
  return res.json();
}

/* ---------- 카드 마스터 저장/삭제 (웹 → Firebase) ---------- */
async function saveCard(cardId, data) {
  return fbPut(`veripass/cards/${cardId}`, data);
}
async function deleteCard(cardId) {
  return fbDelete(`veripass/cards/${cardId}`);
}

/* ---------- 카드 마스터 정보 조회 ---------- */
async function fetchCards() {
  return (await fbGet('veripass/cards')) || {};
}

/* ---------- 운임 설정 조회 (base_fare, early_bird_discount, fine_multiplier) ---------- */
async function fetchFareSettings() {
  return (await fbGet('veripass/fare_settings')) || {
    base_fare: { adult: 1550, teen: 900, senior: 1550 },
    early_bird_discount: { enabled: true, rate: 0.2, end_hour: 6, end_minute: 30 },
    fine_multiplier: 30
  };
}

/* ---------- 위반(불일치) 이력 전체 조회 ---------- */
async function fetchViolations() {
  return (await fbGet('veripass/violations')) || {};
}

/* ---------- 게이트 현황 전체 조회 ---------- */
async function fetchGates() {
  return (await fbGet('veripass/gates')) || {};
}

/* ---------- 통과 기록(logs) 전체 조회 ----------
   RPI가 태깅 1건마다 push한 세션 누적.
   세션 스키마: { Timestamp, Gate, Card_ID, Name, Card_Type, Status, Gender, Age }
   Status: 'normal'(정상) | 'suspect'(부정승차 의심)
*/
async function fetchLogs() {
  return (await fbGet('veripass/logs')) || {};
}

/* ---------- 카드 타입 한글 라벨 ---------- */
const CARD_TYPE_LABEL = {
  adult: '일반',
  senior: '경로우대',
  teen: '청소년',
  disabled: '장애인'
};

/* ---------- 과징금 계산 (web에서 실시간 계산, DB에 저장하지 않음) ----------
   - base_fare: fare_settings.base_fare[card_type]
   - 조조할인: timestamp 시각이 영업시작 ~ end_hour:end_minute 이전이면 적용
   - 최종 과징금 = (base_fare * (1 - discount_rate)) * fine_multiplier
*/
function calcFine(violation, fareSettings) {
  const baseFare = fareSettings.base_fare[violation.card_type] ?? 0;
  const d = new Date(violation.timestamp);
  const hour = d.getHours();
  const minute = d.getMinutes();

  const ebd = fareSettings.early_bird_discount;
  const isEarlyBird =
    ebd && ebd.enabled &&
    (hour < ebd.end_hour || (hour === ebd.end_hour && minute <= ebd.end_minute));

  const discountRate = isEarlyBird ? ebd.rate : 0;
  const discountedFare = Math.round(baseFare * (1 - discountRate));
  const fineAmount = discountedFare * fareSettings.fine_multiplier;

  return {
    baseFare,
    isEarlyBird,
    discountRate,
    discountedFare,
    fineAmount
  };
}
