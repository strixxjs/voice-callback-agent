/**
 * Calls page logic: calls list, call detail (transcript bubbles, extraction,
 * event log), and real-time SSE updates.
 * Strictly uses textContent / createElement for safe DOM rendering.
 */

let selectedCallId = null;
let callsCache = [];

document.addEventListener('DOMContentLoaded', () => {
  const params = new URLSearchParams(window.location.search);
  const initialId = params.get('id');

  loadCalls(initialId);

  const debouncedListReload = debounce(() => {
    loadCalls(selectedCallId);
  }, 300);

  initSSE({
    onEvent: (type, data) => {
      debouncedListReload();
      if (data && data.call_id && data.call_id === selectedCallId) {
        loadCallDetail(selectedCallId);
      }
    },
    onOpen: () => {
      loadCalls(selectedCallId);
      if (selectedCallId) {
        loadCallDetail(selectedCallId);
      }
    }
  });

  // Handle browser back/forward buttons
  window.addEventListener('popstate', () => {
    const p = new URLSearchParams(window.location.search);
    const id = p.get('id');
    if (id && id !== selectedCallId) {
      selectCall(id, false);
    }
  });
});

/**
 * Loads list of calls from GET /api/calls.
 */
async function loadCalls(preferredId = null) {
  try {
    const res = await fetch('/api/calls');
    if (!res.ok) return;
    const calls = await res.json();
    callsCache = calls;

    const countSpan = document.getElementById('calls-count');
    if (countSpan) {
      countSpan.textContent = `${calls.length} ${getNoun(calls.length, 'дзвінок', 'дзвінки', 'дзвінків')}`;
    }

    renderCallsList(calls);

    // Pick call to select
    let toSelect = preferredId;
    if (!toSelect && calls.length > 0) {
      toSelect = String(calls[0].call_id);
    }
    if (toSelect) {
      selectCall(toSelect, false);
    }
  } catch (err) {
    console.warn('Failed to load calls list:', err);
  }
}

function getNoun(number, one, two, five) {
  let n = Math.abs(number);
  n %= 100;
  if (n >= 5 && n <= 20) return five;
  n %= 10;
  if (n === 1) return one;
  if (n >= 2 && n <= 4) return two;
  return five;
}

/**
 * Renders the left sidebar list of calls.
 */
function renderCallsList(calls) {
  const container = document.getElementById('calls-list');
  if (!container) return;
  container.innerHTML = '';

  if (calls.length === 0) {
    const empty = el('div', 'empty-state', 'Дзвінків ще немає. Створіть перший на сторінці «Огляд».');
    container.appendChild(empty);
    return;
  }

  calls.forEach((c) => {
    const item = el('div', `call-item ${String(c.call_id) === selectedCallId ? 'selected' : ''}`);
    item.id = `call-item-${c.call_id}`;

    // Top: Name + Phone
    const top = el('div', 'call-item-top');
    const name = el('span', 'call-item-name', c.lead_name || 'Невідомий лід');
    const phone = el('span', 'call-item-phone', c.phone_masked || '—');
    top.appendChild(name);
    top.appendChild(phone);

    // Bottom: Status badge + Time + Duration
    const bottom = el('div', 'call-item-bottom');
    const badge = createStatusBadge(c.status);
    const durText = c.started_at ? formatDuration(c.duration_seconds) : '—';
    const meta = el('span', '', `${formatDateTime(c.created_at)} • ${durText}`);
    bottom.appendChild(badge);
    bottom.appendChild(meta);

    item.appendChild(top);
    item.appendChild(bottom);

    item.addEventListener('click', () => {
      selectCall(String(c.call_id), true);
    });

    container.appendChild(item);
  });
}

/**
 * Selects a call and displays its details.
 */
function selectCall(callId, updateUrl = true) {
  if (!callId) return;
  selectedCallId = String(callId);

  // Update URL query parameter
  if (updateUrl) {
    history.pushState(null, '', `/calls?id=${encodeURIComponent(selectedCallId)}`);
  }

  // Update active item styling in list
  document.querySelectorAll('.call-item').forEach((item) => {
    item.classList.remove('selected');
  });
  const currentItem = document.getElementById(`call-item-${selectedCallId}`);
  if (currentItem) {
    currentItem.classList.add('selected');
  }

  loadCallDetail(selectedCallId);
}

/**
 * Loads call details from GET /api/calls/{id}.
 */
async function loadCallDetail(callId) {
  const emptyState = document.getElementById('detail-empty');
  const detailContent = document.getElementById('detail-content');

  try {
    const res = await fetch(`/api/calls/${encodeURIComponent(callId)}`);
    if (!res.ok) {
      if (emptyState) {
        emptyState.textContent = 'Дзвінок не знайдено або видалено.';
        emptyState.style.display = 'block';
      }
      if (detailContent) detailContent.style.display = 'none';
      return;
    }

    const call = await res.json();
    renderCallDetail(call);

    if (emptyState) emptyState.style.display = 'none';
    if (detailContent) detailContent.style.display = 'flex';
  } catch (err) {
    console.warn('Failed to load call detail:', err);
  }
}

/**
 * Renders full detail view of a single call.
 */
function renderCallDetail(call) {
  // Header Info
  const nameEl = document.getElementById('detail-name');
  if (nameEl) nameEl.textContent = call.lead_name || 'Невідомий лід';

  const phoneEl = document.getElementById('detail-phone');
  if (phoneEl) phoneEl.textContent = call.phone_masked || '—';

  const statusBadge = document.getElementById('detail-status-badge');
  if (statusBadge) {
    statusBadge.innerHTML = '';
    statusBadge.appendChild(createStatusBadge(call.status));
  }

  const idEl = document.getElementById('detail-id');
  if (idEl) idEl.textContent = String(call.call_id);

  const roomEl = document.getElementById('detail-room');
  if (roomEl) roomEl.textContent = call.room || '—';

  const modeEl = document.getElementById('detail-mode');
  if (modeEl) modeEl.textContent = call.mode === 'phone' ? 'Телефон' : 'Браузер';

  const startedEl = document.getElementById('detail-started');
  if (startedEl) startedEl.textContent = call.started_at ? formatDateTime(call.started_at) : '—';

  const durationEl = document.getElementById('detail-duration');
  if (durationEl) durationEl.textContent = call.started_at ? formatDuration(call.duration_seconds) : '—';

  const endedByEl = document.getElementById('detail-ended-by');
  if (endedByEl) {
    let endedText = '—';
    if (call.ended_by === 'agent') endedText = 'Агент';
    else if (call.ended_by === 'user') endedText = 'Користувач';
    endedByEl.textContent = endedText;
  }

  // 1. Transcript
  renderTranscript(call.transcript || [], call.status);

  // 2. Extraction
  renderExtraction(call.extraction);

  // 3. Event Log
  renderEvents(call.events || []);
}

function renderTranscript(turns, callStatus) {
  const container = document.getElementById('detail-transcript');
  if (!container) return;
  container.innerHTML = '';

  if (turns.length === 0) {
    const isInProgress = callStatus === 'dialing' || callStatus === 'in_progress';
    const msg = isInProgress
      ? 'Транскрипт порожній або розмова ще триває.'
      : 'Розмови не було';
    const empty = el('div', 'empty-state', msg);
    container.appendChild(empty);
    return;
  }

  turns.forEach((turn) => {
    const role = (turn.role || '').toLowerCase();
    let speakerLabel = '';
    let bubbleClass = 'chat-bubble';

    if (role === 'agent' || role === 'assistant') {
      speakerLabel = 'Агент';
      bubbleClass += ' assistant';
    } else if (role === 'user') {
      speakerLabel = 'Клієнт';
      bubbleClass += ' user';
    } else {
      speakerLabel = turn.role || 'Невідомий';
      bubbleClass += ' neutral';
    }

    const bubble = el('div', bubbleClass);

    const meta = el('div', 'bubble-meta');
    const speaker = el('span', '', speakerLabel);
    const time = el('span', '', formatTime(turn.timestamp));
    meta.appendChild(speaker);
    meta.appendChild(time);

    const text = el('div', '', turn.text || '');

    bubble.appendChild(meta);
    bubble.appendChild(text);
    container.appendChild(bubble);
  });
}

function renderExtraction(extraction) {
  const badgeContainer = document.getElementById('detail-extraction-badge');
  const answersGrid = document.getElementById('detail-extraction-answers');
  const missingBox = document.getElementById('detail-extraction-missing-box');
  const missingSpan = document.getElementById('detail-extraction-missing');

  if (badgeContainer) badgeContainer.innerHTML = '';
  if (answersGrid) answersGrid.innerHTML = '';
  if (missingBox) missingBox.style.display = 'none';

  if (!extraction || (!extraction.answers && !extraction.status)) {
    if (badgeContainer) {
      badgeContainer.appendChild(el('span', 'badge badge-no_answer', 'ще немає'));
    }
    if (answersGrid) {
      answersGrid.appendChild(el('div', 'empty-state', 'Екстракція очікується після завершення розмови.'));
    }
    return;
  }

  // Status badge
  if (badgeContainer) {
    let statusText = extraction.status || 'ще немає';
    let badgeClass = 'badge badge-completed';
    if (extraction.status === 'extraction_failed') {
      badgeClass = 'badge badge-failed';
      statusText = 'помилка екстракції';
    } else if (extraction.status === 'ok') {
      statusText = 'успішно';
    }
    badgeContainer.appendChild(el('span', badgeClass, statusText));
  }

  // Answers grid
  const answers = extraction.answers || {};
  const standardKeys = ['name', 'service', 'timeline', 'preferred_call_time'];
  const allKeys = Array.from(new Set([...standardKeys, ...Object.keys(answers)]));

  if (Object.keys(answers).length === 0) {
    answersGrid.appendChild(el('div', 'empty-state', 'Відповіді не було зафіксовано.'));
  } else {
    allKeys.forEach((key) => {
      if (answers[key] !== undefined && answers[key] !== null) {
        const row = el('div', 'answer-row');
        const keyEl = el('span', 'answer-key', key);
        const valEl = el('span', 'answer-val', String(answers[key]));
        row.appendChild(keyEl);
        row.appendChild(valEl);
        answersGrid.appendChild(row);
      }
    });
  }

  // Missing keys
  if (extraction.missing && extraction.missing.length > 0) {
    if (missingBox && missingSpan) {
      missingSpan.textContent = extraction.missing.join(', ');
      missingBox.style.display = 'block';
    }
  }
}

function renderEvents(events) {
  const container = document.getElementById('detail-events');
  if (!container) return;
  container.innerHTML = '';

  if (events.length === 0) {
    container.appendChild(el('div', 'empty-state', 'Подій не зареєстровано.'));
    return;
  }

  events.forEach((ev) => {
    const row = el('div', 'log-entry');
    const ts = el('span', 'log-ts', `[${formatTime(ev.timestamp)}]`);
    const type = el('span', 'log-type', ev.event_type);
    const msg = el('span', 'log-msg', `— ${ev.message}`);

    row.appendChild(ts);
    row.appendChild(type);
    row.appendChild(msg);
    container.appendChild(row);
  });
}
