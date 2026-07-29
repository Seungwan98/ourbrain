(() => {
  'use strict';

  const ACCESS_KEY = 'remoteReviewAccessCode';
  const REVIEWER_KEY = 'remoteReviewReviewer';
  const LABELS = new Set(['N', 'C', 'U']);
  const UI_TO_API_LABEL = { N: 'negative', C: 'crack', U: 'uncertain' };
  const API_TO_UI_LABEL = { negative: 'N', crack: 'C', uncertain: 'U', N: 'N', C: 'C', U: 'U' };
  const SPLITS = ['all', 'train', 'val', 'test'];

  const els = {
    accessForm: document.querySelector('#access-form'),
    accessCode: document.querySelector('#access-code'),
    clearAccess: document.querySelector('#clear-access'),
    datasetPill: document.querySelector('#dataset-pill'),
    revisionPill: document.querySelector('#revision-pill'),
    saveStatus: document.querySelector('#save-status'),
    progressGrid: document.querySelector('#progress-grid'),
    downloadCsv: document.querySelector('#download-csv'),
    candidateImage: document.querySelector('#candidate-image'),
    imagePlaceholder: document.querySelector('#image-placeholder'),
    imagePath: document.querySelector('#image-path'),
    candidateId: document.querySelector('#candidate-id'),
    candidateSplit: document.querySelector('#candidate-split'),
    positionLabel: document.querySelector('#position-label'),
    currentLabel: document.querySelector('#current-label'),
    labelButtons: [...document.querySelectorAll('[data-label]')],
    clearLabel: document.querySelector('#clear-label'),
    reviewer: document.querySelector('#reviewer'),
    note: document.querySelector('#note'),
    prev: document.querySelector('#prev-candidate'),
    next: document.querySelector('#next-candidate'),
    nextUnreviewed: document.querySelector('#next-unreviewed'),
    metadataTable: document.querySelector('#metadata-table'),
    toast: document.querySelector('#toast'),
  };

  const state = {
    datasetId: '',
    revision: null,
    candidates: [],
    decisions: {},
    conflicts: {},
    index: 0,
    saveTimer: null,
    loading: false,
    imageAbort: null,
    imageObjectUrl: '',
    renderedCandidateId: '',
  };

  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    els.accessCode.value = sessionStorage.getItem(ACCESS_KEY) || '';
    els.reviewer.value = sessionStorage.getItem(REVIEWER_KEY) || '';
    bindEvents();
    setSaveStatus('loading', '검수 상태 로딩');
    try {
      await syncReviewState();
      render();
      setSaveStatus('idle', '동기화 완료');
    } catch (error) {
      console.error(error);
      setSaveStatus('error', friendlyError(error));
      showToast(friendlyError(error));
      render();
    }
  }

  function bindEvents() {
    els.accessForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      sessionStorage.setItem(ACCESS_KEY, els.accessCode.value.trim());
      await syncWithFeedback();
    });

    els.clearAccess.addEventListener('click', () => {
      sessionStorage.removeItem(ACCESS_KEY);
      els.accessCode.value = '';
      setSaveStatus('idle', '접근 코드 삭제됨');
      showToast('접근 코드를 sessionStorage에서 지웠습니다.');
    });

    els.labelButtons.forEach((button) => {
      button.addEventListener('click', () => setLabel(button.dataset.label));
    });
    els.clearLabel.addEventListener('click', () => setLabel(null));

    els.reviewer.addEventListener('input', () => {
      sessionStorage.setItem(REVIEWER_KEY, els.reviewer.value);
      scheduleSave();
    });
    els.note.addEventListener('input', scheduleSave);
    els.note.addEventListener('blur', flushScheduledSave);
    els.reviewer.addEventListener('blur', flushScheduledSave);

    els.prev.addEventListener('click', () => move(-1));
    els.next.addEventListener('click', () => move(1));
    els.nextUnreviewed.addEventListener('click', goNextUnreviewed);
    els.downloadCsv.addEventListener('click', downloadCsv);

    document.addEventListener('keydown', (event) => {
      if (event.defaultPrevented || hasEditableTarget(event)) return;
      const key = event.key.toUpperCase();
      if (LABELS.has(key)) {
        event.preventDefault();
        setLabel(key);
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault();
        move(-1);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        move(1);
      }
    });
  }

  async function syncWithFeedback() {
    setSaveStatus('loading', '동기화 중');
    try {
      await syncReviewState();
      render();
      setSaveStatus('idle', '동기화 완료');
      showToast('서버 상태를 동기화했습니다.');
    } catch (error) {
      console.error(error);
      setSaveStatus('error', friendlyError(error));
      showToast(friendlyError(error));
    }
  }

  async function syncReviewState() {
    const response = await apiFetch('/api/review-state');
    if (!response.ok) throw new Error(`검수 상태 로드 실패 (${response.status})`);
    applyServerState(await response.json());
  }

  function applyServerState(payload) {
    if (state.datasetId && payload.datasetId && state.datasetId !== payload.datasetId) {
      throw new Error('서버 dataset ID가 현재 검수 데이터와 다릅니다.');
    }
    state.datasetId = payload.datasetId || state.datasetId;
    state.revision = payload.revision ?? null;
    state.decisions = normalizeDecisions(payload.decisions || {});
    state.conflicts = payload.conflicts || {};
    if (Array.isArray(payload.candidates)) {
      state.candidates = payload.candidates.map((row, index) => ({
        ...row,
        id: String(row.id ?? index),
      }));
      state.index = clamp(
        state.index,
        0,
        Math.max(state.candidates.length - 1, 0),
      );
    }
  }

  function normalizeDecisions(decisions) {
    if (Array.isArray(decisions)) {
      return decisions.reduce((acc, item) => {
        const id = item.candidateId ?? item.id;
        if (id != null) acc[String(id)] = normalizeDecision(item);
        return acc;
      }, {});
    }
    return Object.fromEntries(
      Object.entries(decisions).map(([id, decision]) => [String(id), normalizeDecision(decision)])
    );
  }

  function normalizeDecision(decision) {
    if (!decision || typeof decision !== 'object') return {};
    return {
      label: toUiLabel(decision.label),
      note: decision.note || '',
      reviewer: decision.reviewer || '',
    };
  }

  function render() {
    renderChrome();
    renderProgress();
    renderCandidate();
  }

  function renderChrome() {
    const visibleDatasetId = state.datasetId
      ? `${state.datasetId.slice(0, 12)}…`
      : '-';
    els.datasetPill.textContent = `dataset: ${visibleDatasetId}`;
    els.datasetPill.title = state.datasetId || '';
    els.revisionPill.textContent = `revision: ${state.revision ?? '-'}`;
    els.prev.disabled = state.index <= 0 || !state.candidates.length;
    els.next.disabled = state.index >= state.candidates.length - 1 || !state.candidates.length;
    els.nextUnreviewed.disabled = findNextUnreviewedIndex() == null;
    const progress = summarize('all');
    const conflictCount = Object.keys(state.conflicts).length;
    els.downloadCsv.disabled =
      !getAccessCode() ||
      progress.reviewed !== progress.total ||
      conflictCount > 0;
    els.downloadCsv.title =
      progress.reviewed === progress.total && conflictCount === 0
        ? '완료된 검수 CSV 다운로드'
        : conflictCount > 0
          ? `동시 저장 충돌 ${conflictCount}건을 다시 판정해야 합니다`
          : `모든 후보를 검수해야 합니다 (${progress.reviewed}/${progress.total})`;
  }

  function renderProgress() {
    const summaries = SPLITS.map((split) => summarize(split));
    els.progressGrid.replaceChildren(...summaries.map(progressCard));
  }

  function progressCard(summary) {
    const card = document.createElement('article');
    card.className = 'progress-item';

    const title = document.createElement('div');
    title.className = 'progress-title';
    const name = document.createElement('strong');
    name.textContent = summary.split;
    const count = document.createElement('span');
    count.textContent = `${summary.reviewed}/${summary.total}`;
    title.append(name, count);

    const bar = document.createElement('div');
    bar.className = 'progress-bar';
    bar.setAttribute('aria-label', `${summary.split} 진행률 ${summary.percent}%`);
    bar.setAttribute('role', 'progressbar');
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', '100');
    bar.setAttribute('aria-valuenow', String(summary.percent));
    const fill = document.createElement('span');
    fill.style.width = `${summary.percent}%`;
    bar.append(fill);

    const details = document.createElement('p');
    const conflicts = summary.split === 'all'
      ? Object.keys(state.conflicts).length
      : state.candidates.filter(
        (row) => getSplit(row) === summary.split && state.conflicts[row.id],
      ).length;
    details.textContent = `${summary.percent}% · N ${summary.counts.N} · C ${summary.counts.C} · U ${summary.counts.U}${conflicts ? ` · 충돌 ${conflicts}` : ''}`;

    card.append(title, bar, details);
    return card;
  }

  function summarize(split) {
    const rows = split === 'all' ? state.candidates : state.candidates.filter((row) => getSplit(row) === split);
    const counts = { N: 0, C: 0, U: 0 };
    let reviewed = 0;
    rows.forEach((row) => {
      const label = getDecision(row.id).label;
      if (LABELS.has(label)) {
        reviewed += 1;
        counts[label] += 1;
      }
    });
    const total = rows.length;
    return { split, total, reviewed, counts, percent: total ? Math.round((reviewed / total) * 100) : 0 };
  }

  function renderCandidate() {
    const row = currentCandidate();
    if (!row) {
      els.positionLabel.textContent = '0 / 0';
      els.candidateImage.removeAttribute('src');
      els.candidateImage.hidden = true;
      els.imagePlaceholder.hidden = false;
      els.imagePlaceholder.textContent = '검수할 후보가 없습니다.';
      return;
    }

    const decision = getDecision(row.id);
    els.positionLabel.textContent = `${state.index + 1} / ${state.candidates.length}`;
    els.candidateId.textContent = row.id;
    els.candidateSplit.textContent = getSplit(row) || '-';
    els.imagePath.textContent = row.imageUrl || '-';
    els.note.value = decision.note || '';
    if (!els.reviewer.value && decision.reviewer) els.reviewer.value = decision.reviewer;

    renderLabel(decision.label, Boolean(state.conflicts[row.id]));
    void renderImage(row.imageUrl, row.id);
    renderMetadata(row);
  }

  function renderLabel(label, hasConflict = false) {
    const value = LABELS.has(label) ? label : null;
    els.currentLabel.textContent = hasConflict
      ? '충돌: 다시 판정 필요'
      : value
        ? `라벨 ${value}`
        : '미검수';
    els.currentLabel.className = `label-pill ${value ? `label-${value.toLowerCase()}` : 'unset'}`;
    els.labelButtons.forEach((button) => {
      const selected = button.dataset.label === value;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
  }

  async function renderImage(url, candidateId) {
    if (
      state.renderedCandidateId === candidateId &&
      state.imageObjectUrl &&
      els.candidateImage.src === state.imageObjectUrl
    ) {
      return;
    }
    state.imageAbort?.abort();
    state.imageAbort = new AbortController();
    if (state.imageObjectUrl) URL.revokeObjectURL(state.imageObjectUrl);
    state.imageObjectUrl = '';
    state.renderedCandidateId = '';
    els.candidateImage.hidden = true;
    els.candidateImage.removeAttribute('src');
    els.imagePlaceholder.hidden = false;
    els.imagePlaceholder.textContent = url
      ? '인증된 후보 이미지를 불러오는 중입니다.'
      : '이미지 URL이 없습니다.';
    if (!url) return;

    try {
      const response = await apiFetch(url, { signal: state.imageAbort.signal });
      if (!response.ok) throw new Error(`후보 이미지 로드 실패 (${response.status})`);
      const objectUrl = URL.createObjectURL(await response.blob());
      if (currentCandidate()?.id !== candidateId) {
        URL.revokeObjectURL(objectUrl);
        return;
      }
      state.imageObjectUrl = objectUrl;
      state.renderedCandidateId = candidateId;
      els.candidateImage.src = objectUrl;
      els.candidateImage.alt = `후보 ${candidateId} 이미지`;
      els.candidateImage.hidden = false;
      els.imagePlaceholder.hidden = true;
    } catch (error) {
      if (error?.name === 'AbortError') return;
      console.error(error);
      els.imagePlaceholder.textContent = friendlyError(error);
    }
  }

  function renderMetadata(row) {
    const dl = document.createElement('dl');
    Object.entries(row).forEach(([key, value]) => {
      const wrap = document.createElement('div');
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = key;
      dd.textContent = typeof value === 'object' ? JSON.stringify(value) : String(value ?? '');
      wrap.append(dt, dd);
      dl.append(wrap);
    });
    els.metadataTable.replaceChildren(dl);
  }

  async function setLabel(label) {
    if (label !== null && !LABELS.has(label)) return;
    await saveDecision({ label });
  }

  function scheduleSave() {
    window.clearTimeout(state.saveTimer);
    state.saveTimer = window.setTimeout(() => saveDecision({}), 600);
  }

  function flushScheduledSave() {
    if (!state.saveTimer) return Promise.resolve();
    window.clearTimeout(state.saveTimer);
    state.saveTimer = null;
    return saveDecision({});
  }

  async function saveDecision(patch) {
    const row = currentCandidate();
    if (!row || state.loading) return;
    const previous = getDecision(row.id);
    const next = {
      label: Object.prototype.hasOwnProperty.call(patch, 'label') ? patch.label : previous.label || null,
      note: els.note.value,
      reviewer: els.reviewer.value.trim(),
    };
    state.decisions[row.id] = next;
    render();

    setSaveStatus('saving', '저장 중');
    try {
      state.loading = true;
      const response = await apiFetch('/api/review-state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candidateId: row.id,
          label: toApiLabel(next.label),
          note: next.note,
          reviewer: next.reviewer,
          expectedRevision: state.revision,
        }),
      });

      if (response.status === 409) {
        setSaveStatus('conflict', '충돌: 재동기화 중');
        showToast('서버 revision이 변경되어 재동기화했습니다. 변경 내용을 다시 확인하세요.');
        await syncReviewState();
      } else if (!response.ok) {
        throw new Error(`저장 실패 (${response.status})`);
      } else {
        applyServerState(await response.json());
        setSaveStatus('idle', '저장됨');
      }
      render();
    } catch (error) {
      console.error(error);
      setSaveStatus('error', friendlyError(error));
      showToast(friendlyError(error));
    } finally {
      state.loading = false;
    }
  }

  function move(delta) {
    const nextIndex = clamp(state.index + delta, 0, state.candidates.length - 1);
    if (nextIndex === state.index) return;
    flushScheduledSave();
    state.index = nextIndex;
    render();
  }

  function goNextUnreviewed() {
    const nextIndex = findNextUnreviewedIndex();
    if (nextIndex == null) {
      showToast('미검수 후보가 없습니다.');
      return;
    }
    flushScheduledSave();
    state.index = nextIndex;
    render();
  }

  function findNextUnreviewedIndex() {
    if (!state.candidates.length) return null;
    for (let offset = 1; offset <= state.candidates.length; offset += 1) {
      const index = (state.index + offset) % state.candidates.length;
      const candidateId = state.candidates[index].id;
      if (
        state.conflicts[candidateId] ||
        !LABELS.has(getDecision(candidateId).label)
      ) {
        return index;
      }
    }
    return null;
  }

  async function downloadCsv() {
    setSaveStatus('loading', 'CSV 생성 중');
    try {
      await flushScheduledSave();
      const response = await apiFetch('/api/review-state?format=csv');
      if (!response.ok) throw new Error(`CSV 다운로드 실패 (${response.status})`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${state.datasetId || 'review-state'}-decisions.csv`;
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setSaveStatus('idle', 'CSV 다운로드 완료');
    } catch (error) {
      console.error(error);
      setSaveStatus('error', friendlyError(error));
      showToast(friendlyError(error));
    }
  }

  function apiFetch(path, options = {}) {
    const accessCode = getAccessCode();
    const headers = new Headers(options.headers || {});
    if (accessCode) headers.set('Authorization', `Bearer ${accessCode}`);
    return fetch(path, { ...options, headers, cache: 'no-store' });
  }

  function getAccessCode() {
    return (sessionStorage.getItem(ACCESS_KEY) || els.accessCode.value || '').trim();
  }

  function getDecision(candidateId) {
    return state.decisions[String(candidateId)] || {};
  }

  function toUiLabel(label) {
    return API_TO_UI_LABEL[label] || null;
  }

  function toApiLabel(label) {
    return label ? UI_TO_API_LABEL[label] : '';
  }

  function currentCandidate() {
    return state.candidates[state.index] || null;
  }

  function getSplit(row) {
    return String(row.target_split || row.split || '').toLowerCase();
  }

  function setSaveStatus(kind, text) {
    els.saveStatus.className = `pill save-${kind}`;
    els.saveStatus.textContent = text;
  }

  function showToast(message) {
    els.toast.textContent = message;
    els.toast.classList.add('visible');
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => els.toast.classList.remove('visible'), 3600);
  }

  function friendlyError(error) {
    return error && error.message ? error.message : '알 수 없는 오류가 발생했습니다.';
  }

  function hasEditableTarget(event) {
    const target = event.target;
    return target instanceof HTMLElement && Boolean(target.closest('input, textarea, select, [contenteditable="true"]'));
  }

  function clamp(value, min, max) {
    if (max < min) return min;
    return Math.min(Math.max(value, min), max);
  }
})();
