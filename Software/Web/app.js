/* ===========================================================
   VeriPass 관제 대시보드 — app.js

   데이터 소스:
   - veripass/logs        : 통과 기록(정상+부정). 통계 / 통과 기록 블럭 / 이력 누적 조회
   - veripass/gates/{n}/gate_result : 실시간 부정 판정 → 모달 팝업 + 게이트 차단 표시

   logs 세션 스키마:
     { Timestamp, Gate, Card_ID, Name, Card_Type, Status('normal'|'suspect'), Gender, Age }
=========================================================== */

const GATES = 4;
const GATE_POLL_MS = 1500;  // gate_result 실시간 폴링(모달)
const LOGS_POLL_MS = 3000;  // logs 폴링(기록/이력/통계)

const gateStates = Array(GATES).fill('ok');
let allLogs = [];           // 정규화된 통과 기록 (최신순)
let modalTimer = null;

/* ---------- DOM 참조 ---------- */
const gateGrid = document.getElementById('gates-grid');
const alertsList = document.getElementById('alerts-list');
const recordCount = document.getElementById('record-count');
const historyBody = document.getElementById('history-body');
const historyEmpty = document.getElementById('history-empty');
const filterGate = document.getElementById('filter-gate');
const filterType = document.getElementById('filter-type');
const modalOverlay = document.getElementById('modal-overlay');

/* ---------- 라벨 ---------- */
const STATUS_LABEL = { normal: '정상 통과', suspect: '정보 불일치' };
const GENDER_LABEL = { female: '여', male: '남' };

/* ---------- 유틸 ---------- */
function fmtTime(d) {
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map((n) => String(n).padStart(2, '0'))
    .join(':');
}

// 이름 마스킹: "이다영" → "이00"
function maskName(name) {
  if (!name) return '익명';
  return name[0] + '00';
}

// logs 세션 1건을 화면용 객체로 정규화
function normalizeLog(v) {
  return {
    time: new Date(v.Timestamp),
    gate: Number(v.Gate),
    cardId: v.Card_ID,
    name: maskName(v.Name),
    cardTypeLabel: CARD_TYPE_LABEL[v.Card_Type] || v.Card_Type || '알 수 없음',
    genderLabel: GENDER_LABEL[v.Gender] || '',
    age: v.Age,
    isSuspect: v.Status === 'suspect',
    reason: STATUS_LABEL[v.Status] || v.Status || ''
  };
}

/* ---------- 게이트 그리드 ---------- */
function renderGates() {
  gateGrid.innerHTML = '';
  gateStates.forEach((s, i) => {
    const d = document.createElement('div');
    d.className = 'gate' + (s === 'danger' ? ' warn' : '');
    const iconSvg = s === 'danger'
      ? ICONS.alert
      : '<img class="gate-icon-img" src="images/gate.svg" alt="게이트" />';
    d.innerHTML =
      '<span class="' + (s === 'danger' ? 'pulse' : '') + '">' + iconSvg + '</span>' +
      '<span class="gate-label">G' + String(i + 1).padStart(2, '0') + '</span>';
    gateGrid.appendChild(d);
  });
}

/* ---------- 모달 팝업 (실시간 부정 발생 시) ---------- */
function showModal(ev) {
  document.getElementById('modal-gate').textContent = 'G' + String(ev.gate).padStart(2, '0') + ' 게이트';
  document.getElementById('modal-card').textContent = ev.card;
  document.getElementById('modal-reason').textContent = ev.reason;
  document.getElementById('modal-time').textContent = fmtTime(ev.time);
  modalOverlay.classList.add('show');
  clearTimeout(modalTimer);
  modalTimer = setTimeout(() => modalOverlay.classList.remove('show'), 6000);
}

document.getElementById('modal-close').addEventListener('click', () => {
  modalOverlay.classList.remove('show');
  clearTimeout(modalTimer);
});
modalOverlay.addEventListener('click', (e) => {
  if (e.target === modalOverlay) {
    modalOverlay.classList.remove('show');
    clearTimeout(modalTimer);
  }
});

/* ---------- 통과 기록 블럭 (정상 + 부정) ---------- */
function renderRecords() {
  const recent = allLogs.slice(0, 20);
  recordCount.textContent = allLogs.length + '건';
  alertsList.innerHTML = '';

  recent.forEach((rec) => {
    const row = document.createElement('div');
    row.className = 'row';
    const dotColor = rec.isSuspect ? 'var(--kr-warn)' : 'var(--kr-primary)';
    const badge = rec.isSuspect
      ? '<span class="badge badge-warn">부정승차 의심</span>'
      : '<span class="badge badge-blue">정상</span>';
    row.innerHTML =
      '<div class="dot" style="background:' + dotColor + ';"></div>' +
      '<div style="flex:1; min-width:0;">' +
        '<div style="display:flex; align-items:center; gap:7px; margin-bottom:2px;">' +
          '<span class="row-title">G' + String(rec.gate).padStart(2, '0') + ' — ' + rec.name + '</span>' +
          badge +
        '</div>' +
        '<p class="row-reason">' + rec.cardTypeLabel + ' · ' + rec.genderLabel + ' ' + rec.age + '세</p>' +
      '</div>' +
      '<span class="row-time">' + fmtTime(rec.time) + '</span>';
    alertsList.appendChild(row);
  });
}

/* ---------- 이력 누적 조회 (필터 적용) ---------- */
function renderHistory() {
  const gateFilter = filterGate.value;
  const typeFilter = filterType.value;
  const filtered = allLogs.filter(
    (h) =>
      (gateFilter === 'all' || String(h.gate) === gateFilter) &&
      (typeFilter === 'all' || h.cardTypeLabel === typeFilter)
  );

  historyBody.innerHTML = '';
  historyEmpty.style.display = filtered.length ? 'none' : 'block';

  filtered.slice(0, 50).forEach((h) => {
    const badge = h.isSuspect
      ? '<span class="badge badge-warn">부정승차 의심</span>'
      : '<span class="badge badge-blue">정상</span>';
    const tr = document.createElement('tr');
    tr.innerHTML =
      '<td>' + fmtTime(h.time) + '</td>' +
      '<td style="font-weight:500;">G' + String(h.gate).padStart(2, '0') + '</td>' +
      '<td>' + h.cardTypeLabel + '</td>' +
      '<td style="color:var(--kr-gray1);">' + h.reason + '</td>' +
      '<td style="text-align:right;">' + badge + '</td>';
    historyBody.appendChild(tr);
  });
}

/* ---------- 통계 갱신 (logs 기준) ---------- */
function updateStats() {
  const total = allLogs.length;
  const fraud = allLogs.filter((l) => l.isSuspect).length;
  const locked = gateStates.filter((s) => s === 'danger').length;

  document.getElementById('stat-total').textContent = total.toLocaleString();
  document.getElementById('stat-fraud').textContent = fraud;
  document.getElementById('stat-locked').textContent = locked;
  document.getElementById('stat-rate').textContent =
    total ? Math.round(((total - fraud) / total) * 100) + '%' : '—';
}

/* ---------- logs 폴링 (통계 / 기록 / 이력) ---------- */
async function pollLogs() {
  try {
    const logs = await fetchLogs();
    allLogs = Object.values(logs)
      .filter((v) => v && typeof v === 'object' && v.Timestamp)
      .map(normalizeLog)
      .sort((a, b) => b.time - a.time);

    updateStats();
    renderRecords();
    renderHistory();
  } catch (err) {
    console.error('logs 폴링 오류:', err);
  }
}

/* ---------- gate_result 실시간 폴링 (모달 + 게이트 차단) ---------- */
function subscribeToGateResults() {
  const lastSeen = {}; // gate별 마지막 처리한 timestamp

  async function poll() {
    try {
      const gates = await fetchGates();
      if (!gates) return;

      Object.keys(gates).forEach((gateNum) => {
        const gateData = gates[gateNum];
        const result = gateData && gateData.gate_result;
        if (!result || !result.timestamp) return;
        if (lastSeen[gateNum] === result.timestamp) return;
        lastSeen[gateNum] = result.timestamp;

        if (result.match === false) {
          const gate = Number(gateNum);
          showModal({
            gate,
            card: CARD_TYPE_LABEL[gateData.scan_result && gateData.scan_result.card_type] || '알 수 없음',
            reason: result.reason || '정보 불일치',
            time: new Date(result.timestamp)
          });

          // 게이트 차단 표시 (8초 후 해제)
          if (gate >= 1 && gate <= GATES) {
            gateStates[gate - 1] = 'danger';
            renderGates();
            updateStats();
            setTimeout(() => {
              gateStates[gate - 1] = 'ok';
              renderGates();
              updateStats();
            }, 8000);
          }
        }
      });
    } catch (err) {
      console.error('gate_result 폴링 오류:', err);
    }
  }

  poll();
  setInterval(poll, GATE_POLL_MS);
}

/* ---------- 시간대별 바 차트 ---------- */
function renderBarChart() {
  const hours = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18'];
  const vals = hours.map(() => Math.floor(Math.random() * 80) + 10);
  const peak = Math.max(...vals);

  const barChart = document.getElementById('bar-chart');
  const barLabels = document.getElementById('bar-labels');

  vals.forEach((v, i) => {
    const pct = Math.round((v / peak) * 100);
    const outer = document.createElement('div');
    outer.className = 'bar-outer';
    outer.style.height = pct + '%';
    const inner = document.createElement('div');
    inner.className = 'bar-inner';
    outer.appendChild(inner);
    barChart.appendChild(outer);

    const label = document.createElement('div');
    label.className = 'bar-label';
    label.textContent = hours[i];
    barLabels.appendChild(label);
  });
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
  for (let i = 1; i <= GATES; i++) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = 'G' + String(i).padStart(2, '0');
    filterGate.appendChild(opt);
  }

  renderGates();
  renderHistory();
  renderBarChart();
  updateClock();
  setInterval(updateClock, 1000);

  filterGate.addEventListener('change', renderHistory);
  filterType.addEventListener('change', renderHistory);

  // logs 폴링 (기록/이력/통계)
  pollLogs();
  setInterval(pollLogs, LOGS_POLL_MS);

  // gate_result 폴링 (실시간 모달)
  subscribeToGateResults();
}

document.addEventListener('DOMContentLoaded', init);
