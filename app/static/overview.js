/**
 * Overview page logic: lead submission, live LiveKit connection,
 * 60s participant-wait countdown, and overview components/stages display.
 */

let activeRoom = null;
let activeToken = null; // Stored ONLY in memory
let currentCallId = null;
let countdownTimer = null;
let attachedAudioEls = [];

document.addEventListener('DOMContentLoaded', () => {
  loadOverview();
  setupLeadForm();
  initSSE({
    onEvent: handleSSEEvent,
    onOpen: () => loadOverview()
  });
});

/**
 * Loads project stages and component statuses from GET /api/overview.
 */
async function loadOverview() {
  try {
    const res = await fetch('/api/overview');
    if (!res.ok) return;
    const data = await res.json();
    renderComponents(data.components || []);
    renderStages(data.stages || []);
  } catch (err) {
    console.warn('Failed to load overview data:', err);
  }
}

function renderComponents(components) {
  const tbody = document.getElementById('components-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (components.length === 0) {
    const tr = el('tr');
    const td = el('td', 'empty-state', 'Немає даних');
    td.colSpan = 3;
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  components.forEach((c) => {
    const tr = el('tr');

    const tdName = el('td', '', c.name);
    tdName.style.fontWeight = '500';

    const tdProvider = el('td', '', c.provider || '—');
    tdProvider.style.color = 'var(--muted)';

    const tdStatus = el('td');
    const statusLabel = COMPONENT_STATUS_LABELS[c.status] || c.status;
    let badgeClass = 'badge badge-completed';
    if (c.status === 'not_connected') {
      badgeClass = 'badge badge-no_answer';
    } else if (c.status === 'not_configured') {
      badgeClass = 'badge badge-dialing';
    }
    const badge = el('span', badgeClass, statusLabel);
    tdStatus.appendChild(badge);

    tr.appendChild(tdName);
    tr.appendChild(tdProvider);
    tr.appendChild(tdStatus);
    tbody.appendChild(tr);
  });
}

function renderStages(stages) {
  const tbody = document.getElementById('stages-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (stages.length === 0) {
    const tr = el('tr');
    const td = el('td', 'empty-state', 'Немає даних');
    td.colSpan = 3;
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  stages.forEach((s) => {
    const tr = el('tr');

    const displayId = s.id === 'phone_go_live' ? 'Phone go-live' : s.id;
    const tdId = el('td', '', displayId);
    tdId.style.fontFamily = 'var(--font-mono)';
    tdId.style.color = 'var(--accent)';
    tdId.style.fontWeight = '600';

    const tdTitle = el('td', '', s.title);

    const tdStatus = el('td');
    const statusLabel = STAGE_LABELS[s.status] || s.status;
    let badgeClass = 'badge badge-completed';
    if (s.status === 'in_progress') badgeClass = 'badge badge-in_progress';
    else if (s.status === 'planned') badgeClass = 'badge badge-dialing';
    else if (s.status === 'deferred') badgeClass = 'badge badge-no_answer';

    const badge = el('span', badgeClass, statusLabel);
    tdStatus.appendChild(badge);

    tr.appendChild(tdId);
    tr.appendChild(tdTitle);
    tr.appendChild(tdStatus);
    tbody.appendChild(tr);
  });
}

/**
 * Handles submission of the test lead form.
 */
function setupLeadForm() {
  const form = document.getElementById('lead-form');
  const errorBox = document.getElementById('form-error');
  const submitBtn = document.getElementById('submit-btn');

  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorBox.style.display = 'none';
    errorBox.textContent = '';
    submitBtn.disabled = true;
    submitBtn.textContent = 'Створюємо лід...';

    const name = document.getElementById('lead-name').value.trim();
    const phone = document.getElementById('lead-phone').value.trim();

    try {
      const res = await fetch('/leads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, phone, mode: 'browser' })
      });

      if (!res.ok) {
        let msg = 'Помилка під час створення ліда.';
        try {
          const errData = await res.json();
          if (res.status === 400) {
            msg = 'Телефонний режим наразі недоступний (доступно після Phone go-live).';
          } else if (res.status === 422) {
            msg = "Перевірте дані: ім'я та номер телефону у форматі E.164 (наприклад, +380501234567).";
          } else if (res.status === 502) {
            msg = 'Помилка LiveKit сесії: не вдалося створити кімнату або агент не відповів.';
          } else if (errData && errData.detail) {
            msg = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
          }
        } catch (_) {}

        errorBox.textContent = msg;
        errorBox.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Створити лід і підключитися';
        return;
      }

      const data = await res.json();
      currentCallId = String(data.call_id);
      activeToken = data.token; // In-memory only

      // Display the active session panel
      showActiveCall(data);

      // Connect to LiveKit room in the browser immediately
      await connectLiveKit(data.livekit_url, data.token);
    } catch (err) {
      console.error('Lead submit failed:', err);
      errorBox.textContent = 'Помилка з\'єднання із сервером: ' + (err.message || String(err));
      errorBox.style.display = 'block';
      submitBtn.disabled = false;
      submitBtn.textContent = 'Створити лід і підключитися';
    }
  });

  // Disconnect button
  const disconnectBtn = document.getElementById('disconnect-btn');
  if (disconnectBtn) {
    disconnectBtn.addEventListener('click', async () => {
      disconnectBtn.disabled = true;
      disconnectBtn.textContent = 'Завершення...';
      if (activeRoom) {
        try {
          await activeRoom.disconnect();
        } catch (e) {
          console.warn('Error disconnecting room:', e);
        }
      }
      stopCountdown();
      const statusBadge = document.getElementById('call-status-badge');
      if (statusBadge) {
        statusBadge.innerHTML = '';
        statusBadge.appendChild(createStatusBadge('completed'));
      }
      const audioStatus = document.getElementById('conn-audio-status');
      if (audioStatus) {
        audioStatus.textContent = 'Відключено';
        audioStatus.style.color = 'var(--muted)';
      }
    });
  }

  // Audio enable button
  const startAudioBtn = document.getElementById('start-audio-btn');
  if (startAudioBtn) {
    startAudioBtn.addEventListener('click', async () => {
      if (activeRoom) {
        try {
          await activeRoom.startAudio();
          checkAudioBlocked(activeRoom);
        } catch (e) {
          console.warn('Failed to start audio playback:', e);
        }
      }
    });
  }
}

/**
 * Renders the active call panel with details and links.
 */
function showActiveCall(data) {
  const panel = document.getElementById('connection-panel');
  if (!panel) return;
  panel.style.display = 'block';

  const callIdSpan = document.getElementById('conn-call-id');
  if (callIdSpan) callIdSpan.textContent = String(data.call_id);

  const roomSpan = document.getElementById('conn-room');
  if (roomSpan) roomSpan.textContent = String(data.room);

  const statusBadge = document.getElementById('call-status-badge');
  if (statusBadge) {
    statusBadge.innerHTML = '';
    statusBadge.appendChild(createStatusBadge(data.status || 'dialing'));
  }

  const detailsBtn = document.getElementById('call-details-btn');
  if (detailsBtn) detailsBtn.href = `/calls?id=${encodeURIComponent(data.call_id)}`;

  const meetBtn = document.getElementById('meet-btn');
  if (meetBtn) {
    meetBtn.href = `https://meet.livekit.io/custom?liveKitUrl=${encodeURIComponent(data.livekit_url)}&token=${encodeURIComponent(data.token)}`;
  }

  startCountdown();
}

/**
 * 60 s countdown until participant / agent joins.
 */
function startCountdown() {
  stopCountdown();
  let remaining = 60;
  const box = document.getElementById('countdown-box');
  const num = document.getElementById('countdown-num');
  const text = document.getElementById('countdown-text');

  if (box) {
    box.className = 'countdown-hint';
  }

  countdownTimer = setInterval(() => {
    remaining -= 1;
    if (num) num.textContent = String(remaining);

    if (remaining <= 0) {
      stopCountdown();
      if (text) {
        text.textContent = 'Таймаут 60 с минув. Перевірте, чи запущений LiveKit агент.';
      }
    }
  }, 1000);
}

function stopCountdown() {
  if (countdownTimer) {
    clearInterval(countdownTimer);
    countdownTimer = null;
  }
}

function markAgentConnected() {
  stopCountdown();
  const box = document.getElementById('countdown-box');
  const text = document.getElementById('countdown-text');
  if (box) box.className = 'countdown-hint connected';
  if (text) text.textContent = 'Агент у кімнаті! Говоріть у мікрофон.';
}

/**
 * Connects to the LiveKit room, enables microphone, and attaches audio tracks.
 */
async function connectLiveKit(livekitUrl, token) {
  if (typeof LivekitClient === 'undefined') {
    console.error('LivekitClient UMD library not loaded.');
    return;
  }

  if (activeRoom) {
    try {
      await activeRoom.disconnect();
    } catch (_) {}
  }

  const room = new LivekitClient.Room({
    adaptiveStream: true,
    dynacast: true
  });
  activeRoom = room;

  // Handle incoming remote audio tracks from the agent
  room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
    if (track.kind === LivekitClient.Track.Kind.Audio) {
      const audioEl = track.attach();
      attachedAudioEls.push(audioEl);
      document.body.appendChild(audioEl);
      markAgentConnected();
      checkAudioBlocked(room);
    }
  });

  room.on(LivekitClient.RoomEvent.ParticipantConnected, () => {
    markAgentConnected();
  });

  room.on(LivekitClient.RoomEvent.AudioPlaybackStatusChanged, () => {
    checkAudioBlocked(room);
  });

  room.on(LivekitClient.RoomEvent.Disconnected, () => {
    const audioStatus = document.getElementById('conn-audio-status');
    if (audioStatus) {
      audioStatus.textContent = 'Відключено';
      audioStatus.style.color = 'var(--muted)';
    }
    // Clean up audio elements
    attachedAudioEls.forEach((el) => {
      try {
        el.remove();
      } catch (_) {}
    });
    attachedAudioEls = [];
  });

  try {
    await room.connect(livekitUrl, token);
    await room.localParticipant.setMicrophoneEnabled(true);
    checkAudioBlocked(room);
  } catch (err) {
    console.error('Failed to connect to LiveKit room:', err);
    const audioStatus = document.getElementById('conn-audio-status');
    if (audioStatus) {
      audioStatus.textContent = 'Помилка підключення';
      audioStatus.style.color = 'var(--danger)';
    }
  }
}

function checkAudioBlocked(room) {
  const alert = document.getElementById('audio-blocked-alert');
  if (!alert) return;
  if (room && !room.canPlaybackAudio) {
    alert.style.display = 'block';
  } else {
    alert.style.display = 'none';
  }
}

/**
 * SSE event handler for the overview page.
 */
function handleSSEEvent(type, data) {
  if (!data || !data.call_id) return;
  if (String(data.call_id) === currentCallId) {
    const statusBadge = document.getElementById('call-status-badge');
    if (statusBadge && data.status) {
      statusBadge.innerHTML = '';
      statusBadge.appendChild(createStatusBadge(data.status));
    }

    if (data.status === 'in_progress') {
      markAgentConnected();
    } else if (['completed', 'no_answer', 'failed'].includes(data.status)) {
      stopCountdown();
    }
  }
}
