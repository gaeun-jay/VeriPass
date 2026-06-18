/* ===========================================================
   cards.js — VeriPass 카드 마스터 관리 (양방향 동기화)

   Firebase Realtime Database REST API를 사용합니다.
   공통 연동 함수는 firebase.js (같은 폴더, 먼저 로드).

   - Firebase → 웹 : POLL_INTERVAL_MS 주기로 cards를 다시 읽어 테이블 갱신
   - 웹 → Firebase : 추가/수정 시 saveCard(PUT), 삭제 시 deleteCard(DELETE)

   수정 모달이 열려 있는 동안에는 폴링이 모달 입력을 덮어쓰지 않도록
   목록 테이블만 다시 그립니다(모달은 한 번 채운 값을 유지).
=========================================================== */

const POLL_INTERVAL_MS = 3000;

let cardsCache = {};
let editingCardId = null; // 수정 모달에서 편집 중인 카드 ID (null이면 닫힘)

/* ---------- DOM 참조 ---------- */
const cardsBody = document.getElementById('cards-body');
const cardsEmpty = document.getElementById('cards-empty');
const cardMeta = document.getElementById('card-meta');

const addId = document.getElementById('add-id');
const addName = document.getElementById('add-name');
const addAge = document.getElementById('add-age');
const addType = document.getElementById('add-type');
const addGender = document.getElementById('add-gender');
const addBtn = document.getElementById('add-btn');
const addStatus = document.getElementById('add-status');

const editOverlay = document.getElementById('edit-overlay');
const editIdLabel = document.getElementById('edit-id-label');
const editName = document.getElementById('edit-name');
const editAge = document.getElementById('edit-age');
const editType = document.getElementById('edit-type');
const editGender = document.getElementById('edit-gender');
const editStatus = document.getElementById('edit-status');

const GENDER_LABEL = { female: '여성', male: '남성' };

/* ---------- 상태 메시지 ---------- */
function setStatus(el, msg, kind) {
  el.textContent = msg;
  el.className = 'save-status' + (kind ? ' ' + kind : '');
  if (kind === 'ok') {
    setTimeout(() => {
      if (el.textContent === msg) {
        el.textContent = '';
        el.className = 'save-status';
      }
    }, 2500);
  }
}

/* ---------- 다음 카드 ID 자동 생성 (CARD_001 형식) ---------- */
function nextCardId() {
  let max = 0;
  Object.keys(cardsCache).forEach((id) => {
    const m = id.match(/^CARD_(\d+)$/);
    if (m) max = Math.max(max, Number(m[1]));
  });
  return 'CARD_' + String(max + 1).padStart(3, '0');
}

/* ---------- 목록 테이블 렌더 (Firebase → 웹) ---------- */
function renderCards() {
  const ids = Object.keys(cardsCache).sort();
  cardsBody.innerHTML = '';
  cardsEmpty.style.display = ids.length ? 'none' : 'block';
  cardMeta.textContent = ids.length + '장';

  ids.forEach((id) => {
    const c = cardsCache[id] || {};
    const tr = document.createElement('tr');
    tr.innerHTML =
      '<td style="color:var(--kr-gray1);">' + id + '</td>' +
      '<td style="font-weight:500;">' + (c.name || '—') + '</td>' +
      '<td>' + (c.age != null ? c.age : '—') + '</td>' +
      '<td>' + (CARD_TYPE_LABEL[c.card_type] || c.card_type || '—') + '</td>' +
      '<td>' + (GENDER_LABEL[c.gender] || c.gender || '—') + '</td>' +
      '<td class="row-actions"></td>';

    const actions = tr.querySelector('.row-actions');
    const editBtn = document.createElement('button');
    editBtn.className = 'btn btn-ghost btn-sm';
    editBtn.textContent = '수정';
    editBtn.addEventListener('click', () => openEditModal(id));

    const delBtn = document.createElement('button');
    delBtn.className = 'btn btn-danger btn-sm';
    delBtn.textContent = '삭제';
    delBtn.addEventListener('click', () => removeCard(id));

    actions.appendChild(editBtn);
    actions.appendChild(delBtn);
    cardsBody.appendChild(tr);
  });
}

/* ---------- 입력값 검증 + 카드 객체 만들기 ---------- */
function buildCardData(name, age, cardType, gender) {
  const trimmedName = (name || '').trim();
  if (!trimmedName) return { error: '이름을 입력하세요.' };
  const ageNum = Number(age);
  if (!Number.isInteger(ageNum) || ageNum < 0) return { error: '나이를 올바르게 입력하세요.' };
  return {
    data: { name: trimmedName, age: ageNum, card_type: cardType, gender: gender }
  };
}

/* ---------- 추가 (웹 → Firebase) ---------- */
async function addCard() {
  const built = buildCardData(addName.value, addAge.value, addType.value, addGender.value);
  if (built.error) {
    setStatus(addStatus, built.error, 'err');
    return;
  }

  const idInput = addId.value.trim();
  const cardId = idInput || nextCardId();
  if (cardsCache[cardId]) {
    setStatus(addStatus, '이미 존재하는 카드 ID입니다: ' + cardId, 'err');
    return;
  }

  addBtn.disabled = true;
  setStatus(addStatus, '저장 중...', '');
  try {
    await saveCard(cardId, built.data);
    // 즉시 로컬 캐시 반영 후 재렌더 (폴링 기다리지 않음)
    cardsCache[cardId] = built.data;
    renderCards();
    addId.value = '';
    addName.value = '';
    addAge.value = '';
    setStatus(addStatus, cardId + ' 추가 완료', 'ok');
  } catch (err) {
    console.error(err);
    setStatus(addStatus, '저장 실패: ' + err.message, 'err');
  } finally {
    addBtn.disabled = false;
  }
}

/* ---------- 삭제 (웹 → Firebase) ---------- */
async function removeCard(cardId) {
  const c = cardsCache[cardId] || {};
  if (!confirm(`${cardId} (${c.name || '이름 없음'}) 카드를 삭제할까요?`)) return;
  try {
    await deleteCard(cardId);
    delete cardsCache[cardId];
    renderCards();
  } catch (err) {
    console.error(err);
    alert('삭제 실패: ' + err.message);
  }
}

/* ---------- 수정 모달 ---------- */
function openEditModal(cardId) {
  const c = cardsCache[cardId] || {};
  editingCardId = cardId;
  editIdLabel.textContent = cardId;
  editName.value = c.name || '';
  editAge.value = c.age != null ? c.age : '';
  editType.value = c.card_type || 'adult';
  editGender.value = c.gender || 'female';
  setStatus(editStatus, '', '');
  editOverlay.classList.add('show');
}

function closeEditModal() {
  editingCardId = null;
  editOverlay.classList.remove('show');
}

async function saveEdit() {
  if (!editingCardId) return;
  const built = buildCardData(editName.value, editAge.value, editType.value, editGender.value);
  if (built.error) {
    setStatus(editStatus, built.error, 'err');
    return;
  }
  setStatus(editStatus, '저장 중...', '');
  try {
    await saveCard(editingCardId, built.data);
    cardsCache[editingCardId] = built.data;
    renderCards();
    closeEditModal();
  } catch (err) {
    console.error(err);
    setStatus(editStatus, '저장 실패: ' + err.message, 'err');
  }
}

/* ---------- Firebase 폴링 (Firebase → 웹) ---------- */
async function pollData() {
  try {
    const cards = await fetchCards();
    cardsCache = cards || {};
    renderCards();
  } catch (err) {
    console.error('카드 관리 데이터 로드 오류:', err);
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

  addBtn.addEventListener('click', addCard);
  document.getElementById('edit-save').addEventListener('click', saveEdit);
  document.getElementById('edit-cancel').addEventListener('click', closeEditModal);
  editOverlay.addEventListener('click', (e) => {
    if (e.target === editOverlay) closeEditModal();
  });

  pollData();
  setInterval(pollData, POLL_INTERVAL_MS);
}

document.addEventListener('DOMContentLoaded', init);
