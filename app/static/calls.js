/**
 * Calls page logic: calls list, call detail (transcript bubbles, extraction,
 * event log), and real-time SSE updates.
 * Strictly uses textContent / createElement for safe DOM rendering.
 */

let selectedCallId = null;
let callsCache = [];
let isHeaderPanelOpen = false;
const openTurnPanels = new Set();

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
  const newId = String(callId);
  if (newId !== selectedCallId) {
    selectedCallId = newId;
    isHeaderPanelOpen = false;
    openTurnPanels.clear();
  }

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
 * Provider name normalization: google->Google, deepgram->Deepgram, elevenlabs->ElevenLabs.
 */
function formatProvider(name) {
  if (!name) return '';
  const lower = String(name).toLowerCase();
  if (lower === 'google') return 'Google';
  if (lower === 'deepgram') return 'Deepgram';
  if (lower === 'elevenlabs') return 'ElevenLabs';
  return String(name);
}

function mono(text) {
  return el('span', 'mono', text);
}

function faint(text) {
  return el('span', 'faint', text);
}

function createLatencyRow(label, valNodes) {
  const row = el('div', 'latency-row');
  const labelEl = el('span', 'latency-row-label', label);
  const valEl = el('div', 'latency-row-val');
  if (Array.isArray(valNodes)) {
    valNodes.forEach((node) => {
      if (typeof node === 'string') {
        valEl.appendChild(document.createTextNode(node));
      } else if (node instanceof Node) {
        valEl.appendChild(node);
      }
    });
  } else if (typeof valNodes === 'string') {
    valEl.textContent = valNodes;
  } else if (valNodes instanceof Node) {
    valEl.appendChild(valNodes);
  }
  row.appendChild(labelEl);
  row.appendChild(valEl);
  return row;
}

/**
 * Renders latency badge button and collapsible panel in the header card.
 */
function renderHeaderLatency(latency, callConfig) {
  const badgeBtn = document.getElementById('detail-latency-badge');
  const panelEl = document.getElementById('detail-latency-panel');
  if (!badgeBtn || !panelEl) return;

  if (!latency || latency.p50 === null || latency.p50 === undefined) {
    badgeBtn.style.display = 'none';
    panelEl.style.display = 'none';
    panelEl.innerHTML = '';
    return;
  }

  // 1. Badge button
  badgeBtn.textContent = `⏱ ${latency.p50.toFixed(1)} с`;
  badgeBtn.style.display = 'inline-flex';
  badgeBtn.setAttribute('aria-expanded', String(isHeaderPanelOpen));

  badgeBtn.onclick = () => {
    isHeaderPanelOpen = !isHeaderPanelOpen;
    badgeBtn.setAttribute('aria-expanded', String(isHeaderPanelOpen));
    panelEl.style.display = isHeaderPanelOpen ? 'flex' : 'none';
  };

  // 2. Collapsible panel
  panelEl.innerHTML = '';
  panelEl.style.display = isHeaderPanelOpen ? 'flex' : 'none';

  const rows = [];

  // - Затримка: p50 X с · max Y с (2 decimals)
  if (latency.p50 !== null || latency.max !== null) {
    const nodes = [];
    if (latency.p50 !== null && latency.p50 !== undefined) {
      nodes.push('p50 ');
      nodes.push(mono(`${latency.p50.toFixed(2)} с`));
    }
    if (latency.max !== null && latency.max !== undefined) {
      if (nodes.length > 0) nodes.push(' · ');
      nodes.push('max ');
      nodes.push(mono(`${latency.max.toFixed(2)} с`));
    }
    rows.push(createLatencyRow('Затримка:', nodes));
  }

  // - Ходів: LLM N · сценарних M
  if (latency.llm_turns !== null || latency.scripted_turns !== null) {
    const nodes = [];
    if (latency.llm_turns !== null && latency.llm_turns !== undefined) {
      nodes.push('LLM ');
      nodes.push(mono(String(latency.llm_turns)));
    }
    if (latency.scripted_turns !== null && latency.scripted_turns !== undefined) {
      if (nodes.length > 0) nodes.push(' · ');
      nodes.push('сценарних ');
      nodes.push(mono(String(latency.scripted_turns)));
    }
    rows.push(createLatencyRow('Ходів:', nodes));
  }

  // - Пауза: <endpointing_mode> <min_delay> с (show max_delay only when endpointing_mode is "dynamic"); grace <end_call_grace_period> с
  if (callConfig) {
    const hasEndpointing = Boolean(callConfig.endpointing_mode || callConfig.endpointing_min_delay !== null);
    const hasGrace = callConfig.end_call_grace_period !== null && callConfig.end_call_grace_period !== undefined;
    if (hasEndpointing || hasGrace) {
      const nodes = [];
      if (hasEndpointing) {
        const mode = callConfig.endpointing_mode || '';
        if (mode) nodes.push(`${mode} `);
        const isDynamic = String(mode).toLowerCase() === 'dynamic';
        const minDelay = callConfig.endpointing_min_delay;
        const maxDelay = callConfig.endpointing_max_delay;
        if (isDynamic && maxDelay !== null && maxDelay !== undefined && minDelay !== null && minDelay !== undefined) {
          nodes.push(mono(`${minDelay}–${maxDelay} с`));
        } else if (minDelay !== null && minDelay !== undefined) {
          nodes.push(mono(`${minDelay} с`));
        }
      }
      if (hasGrace) {
        if (nodes.length > 0) nodes.push('; ');
        nodes.push('grace ');
        nodes.push(mono(`${callConfig.end_call_grace_period} с`));
      }
      rows.push(createLatencyRow('Пауза:', nodes));
    }
  }

  // - STT: Deepgram <stt_model> (<stt_language>)
  if (callConfig) {
    const sttProv = formatProvider(callConfig.stt_provider);
    const sttModel = callConfig.stt_model || '';
    const sttLang = callConfig.stt_language || '';
    if (sttProv || sttModel) {
      let val = [sttProv, sttModel].filter(Boolean).join(' ');
      if (sttLang) val += ` (${sttLang})`;
      rows.push(createLatencyRow('STT:', val));
    }
  }

  // - LLM: Google <llm_model>; fallback <llm_fallback_model>; спрацював: ні | ходи #a, #b
  if (callConfig) {
    const llmProv = formatProvider(callConfig.llm_provider);
    const llmModel = callConfig.llm_model || '';
    const fallbackModel = callConfig.llm_fallback_model || '';
    if (llmProv || llmModel) {
      const nodes = [];
      nodes.push([llmProv, llmModel].filter(Boolean).join(' '));
      if (fallbackModel) {
        nodes.push(`; fallback ${fallbackModel}`);
      }
      nodes.push('; спрацював: ');
      const fTurns = Array.isArray(latency.fallback_turns) ? latency.fallback_turns : [];
      if (fTurns.length > 0) {
        nodes.push('ходи ' + fTurns.map((t) => '#' + t).join(', '));
      } else {
        nodes.push('ні');
      }
      rows.push(createLatencyRow('LLM:', nodes));
    }
  }

  // - TTS: ElevenLabs <tts_model>, голос <first 6 chars of voice id>…
  if (callConfig) {
    const ttsProv = formatProvider(callConfig.tts_provider);
    const ttsModel = callConfig.tts_model || '';
    const voiceId = callConfig.tts_voice_id || '';
    if (ttsProv || ttsModel) {
      const nodes = [];
      nodes.push([ttsProv, ttsModel].filter(Boolean).join(' '));
      if (voiceId) {
        const shortVoice = voiceId.length > 6 ? voiceId.slice(0, 6) + '…' : voiceId;
        nodes.push(`, голос ${shortVoice}`);
      }
      rows.push(createLatencyRow('TTS:', nodes));
    }
  }

  // - Режими: turn detection <..>, interruption <..>, preemptive так/ні
  if (callConfig) {
    const modeParts = [];
    if (callConfig.turn_detection_mode) {
      modeParts.push(`turn detection ${callConfig.turn_detection_mode}`);
    }
    if (callConfig.interruption_mode) {
      modeParts.push(`interruption ${callConfig.interruption_mode}`);
    }
    if (callConfig.preemptive_generation !== undefined && callConfig.preemptive_generation !== null) {
      modeParts.push(`preemptive ${callConfig.preemptive_generation ? 'так' : 'ні'}`);
    }
    if (modeParts.length > 0) {
      rows.push(createLatencyRow('Режими:', modeParts.join(', ')));
    }
  }

  // Append all rows
  rows.forEach((r) => panelEl.appendChild(r));

  // - Muted hint
  if (rows.length > 0) {
    const hint = el(
      'div',
      'latency-hint',
      'LLM частково працює паралельно з паузою (preemptive), тому сума компонентів не дорівнює загальній затримці.'
    );
    panelEl.appendChild(hint);
  }
}

/**
 * Creates collapsible mini panel for a single agent turn.
 */
function createTurnLatencyPanel(m) {
  const panel = el('div', 'turn-latency-panel');

  if (m.scripted) {
    // For scripted turns: show TTS TTFB; if e2e is present, show it with the muted note "включає рішення LLM і grace, не входить у p50".
    if (m.e2e !== null && m.e2e !== undefined) {
      const valNodes = [
        mono(`${m.e2e.toFixed(2)} с`),
        ' ',
        faint('(включає рішення LLM і grace, не входить у p50)')
      ];
      panel.appendChild(createLatencyRow('Загалом:', valNodes));
    }
    if (m.tts_ttfb !== null && m.tts_ttfb !== undefined) {
      panel.appendChild(createLatencyRow('TTS TTFB:', mono(`${m.tts_ttfb.toFixed(2)} с`)));
    }
    return panel;
  }

  // LLM turn
  // 1. Загалом: e2e с (if present). If e2e_source is "computed", add muted "оцінка: сума компонентів". If sdk_e2e is present and differs from e2e by > 0.3 s, add muted "SDK: X с — виправлено"
  if (m.e2e !== null && m.e2e !== undefined) {
    const valNodes = [mono(`${m.e2e.toFixed(2)} с`)];
    if (m.e2e_source === 'computed') {
      valNodes.push(' ');
      valNodes.push(faint('(оцінка: сума компонентів)'));
    }
    if (m.sdk_e2e !== null && m.sdk_e2e !== undefined && Math.abs(m.sdk_e2e - m.e2e) > 0.3) {
      valNodes.push(' ');
      valNodes.push(faint(`(SDK: ${m.sdk_e2e.toFixed(2)} с — виправлено)`));
    }
    panel.appendChild(createLatencyRow('Загалом:', valNodes));
  }

  // 2. Пауза (endpointing): X с
  if (m.endpointing_delay !== null && m.endpointing_delay !== undefined) {
    panel.appendChild(createLatencyRow('Пауза (endpointing):', mono(`${m.endpointing_delay.toFixed(2)} с`)));
  }

  // 3. Транскрипт: X с після кінця мовлення
  if (m.transcript_delay !== null && m.transcript_delay !== undefined) {
    const valNodes = [
      mono(`${m.transcript_delay.toFixed(2)} с`),
      ' після кінця мовлення'
    ];
    panel.appendChild(createLatencyRow('Транскрипт:', valNodes));
  }

  // 4. LLM TTFT: X с (<llm_model>; fallback, if llm_fallback)
  if (m.llm_ttft !== null && m.llm_ttft !== undefined) {
    const valNodes = [mono(`${m.llm_ttft.toFixed(2)} с`)];
    const details = [];
    if (m.llm_model) details.push(m.llm_model);
    if (m.llm_fallback) details.push('fallback');
    if (details.length > 0) {
      valNodes.push(` (${details.join('; ')})`);
    }
    panel.appendChild(createLatencyRow('LLM TTFT:', valNodes));
  }

  // 5. TTS TTFB: X с
  if (m.tts_ttfb !== null && m.tts_ttfb !== undefined) {
    panel.appendChild(createLatencyRow('TTS TTFB:', mono(`${m.tts_ttfb.toFixed(2)} с`)));
  }

  return panel;
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

  // Latency & Call Config Header
  renderHeaderLatency(call.latency, call.call_config);

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

  turns.forEach((turn, idx) => {
    const role = (turn.role || '').toLowerCase();
    const isAgent = role === 'agent' || role === 'assistant';
    let speakerLabel = '';
    let bubbleClass = 'chat-bubble';

    if (isAgent) {
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

    let panel = null;
    const m = turn.metrics;
    if (isAgent && m) {
      const turnKey = turn.turn_index !== undefined && turn.turn_index !== null ? turn.turn_index : idx;
      const isTurnOpen = openTurnPanels.has(turnKey);

      let btnText = '';
      if (m.scripted) {
        btnText = 'сценарна';
      } else if (m.e2e !== null && m.e2e !== undefined) {
        btnText = `${m.e2e.toFixed(1)} с`;
      } else {
        btnText = '—';
      }

      const btn = el('button', 'turn-latency-btn', btnText);
      btn.type = 'button';
      btn.setAttribute('aria-expanded', String(isTurnOpen));

      panel = createTurnLatencyPanel(m);
      panel.style.display = isTurnOpen ? 'flex' : 'none';

      btn.addEventListener('click', () => {
        if (openTurnPanels.has(turnKey)) {
          openTurnPanels.delete(turnKey);
          panel.style.display = 'none';
          btn.setAttribute('aria-expanded', 'false');
        } else {
          openTurnPanels.add(turnKey);
          panel.style.display = 'flex';
          btn.setAttribute('aria-expanded', 'true');
        }
      });

      meta.appendChild(btn);
    }

    const text = el('div', '', turn.text || '');

    bubble.appendChild(meta);
    bubble.appendChild(text);
    if (panel) {
      bubble.appendChild(panel);
    }
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
