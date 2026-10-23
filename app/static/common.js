/**
 * Common utilities and UI helpers for the Voice Callback Agent console.
 * Strictly avoids innerHTML for any API-derived or dynamic data.
 */

const STATUS_LABELS = {
  dialing: 'дзвонимо',
  in_progress: 'у розмові',
  completed: 'завершено',
  no_answer: 'не відповів',
  failed: 'помилка'
};

const STAGE_LABELS = {
  done: 'Завершено',
  in_progress: 'В роботі',
  planned: 'Заплановано',
  deferred: 'Відкладено'
};

const COMPONENT_STATUS_LABELS = {
  configured: 'налаштовано',
  not_configured: 'не налаштовано',
  not_connected: 'не підключено'
};

/**
 * Creates an element with optional class name and safe text content.
 */
function el(tag, className = '', textContent = '') {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (textContent !== undefined && textContent !== null && textContent !== '') {
    element.textContent = String(textContent);
  }
  return element;
}

/**
 * Creates a status badge element.
 */
function createStatusBadge(status) {
  const normStatus = (status || 'dialing').toLowerCase();
  const label = STATUS_LABELS[normStatus] || normStatus;
  const badge = el('span', `badge badge-${normStatus}`);
  badge.textContent = label;
  return badge;
}

/**
 * Formats an ISO datetime string into human readable time / date.
 */
function formatTime(isoString) {
  if (!isoString) return '—';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleTimeString('uk-UA', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch (e) {
    return isoString;
  }
}

function formatDateTime(isoString) {
  if (!isoString) return '—';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    const dateStr = d.toLocaleDateString('uk-UA', { day: '2-digit', month: '2-digit' });
    const timeStr = d.toLocaleTimeString('uk-UA', { hour: '2-digit', minute: '2-digit' });
    return `${dateStr} ${timeStr}`;
  } catch (e) {
    return isoString;
  }
}

/**
 * Formats duration in seconds into 'X хв Y с' or 'X с'.
 */
function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || seconds === '') return '—';
  const sec = Math.max(0, parseInt(seconds, 10));
  if (isNaN(sec)) return '—';
  if (sec < 60) return `${sec} с`;
  const mins = Math.floor(sec / 60);
  const remSec = sec % 60;
  return `${mins} хв ${remSec} с`;
}

/**
 * Debounce helper to coalesce rapid events.
 */
function debounce(fn, delayMs = 300) {
  let timer = null;
  return function (...args) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      fn.apply(this, args);
    }, delayMs);
  };
}

/**
 * Sets up a persistent EventSource with automatic reconnection monitoring.
 */
function initSSE({ onEvent, onOpen, onError, callId = null } = {}) {
  const url = callId ? `/api/events?call_id=${encodeURIComponent(callId)}` : '/api/events';
  const dot = document.getElementById('sse-dot');
  const text = document.getElementById('sse-text');

  function updateStatus(connected) {
    if (dot) {
      dot.className = `status-dot ${connected ? 'connected' : 'disconnected'}`;
    }
    if (text) {
      text.textContent = connected ? 'SSE: підключено' : 'SSE: відключено';
    }
  }

  let es = null;

  try {
    es = new EventSource(url);
  } catch (e) {
    console.error('Failed to create EventSource:', e);
    updateStatus(false);
    return null;
  }

  es.onopen = () => {
    updateStatus(true);
    if (typeof onOpen === 'function') {
      onOpen();
    }
  };

  es.onerror = (err) => {
    updateStatus(false);
    if (typeof onError === 'function') {
      onError(err);
    }
  };

  const handleMessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (typeof onEvent === 'function') {
        onEvent(e.type, data);
      }
    } catch (err) {
      console.warn('Invalid JSON from SSE event:', e.data, err);
    }
  };

  // Listen for specific event types as well as standard message
  const kinds = ['ready', 'call_created', 'status', 'event', 'turn', 'extraction', 'message'];
  kinds.forEach((kind) => {
    es.addEventListener(kind, handleMessage);
  });

  return es;
}
