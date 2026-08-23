// =========================================================
// TALOS AI AGENT - VANILLA JS CLIENT (ZERO DEPENDENCIES)
// =========================================================

(function () {
  'use strict';

  // --- STATE ---
  let activeTaskId = null;
  let activeTaskDetail = null;
  let events = [];
  let rawLogs = [];
  let tasks = [];
  let ws = null;
  let reconnectTimer = null;
  let activeFilter = 'all';
  let activeView = 'split';

  // --- DOM ELEMENTS ---
  const elSidebar = document.getElementById('sidebar');
  const elBtnToggleSidebar = document.getElementById('btn-toggle-sidebar');
  const elBtnNewTask = document.getElementById('btn-new-task');
  const elBtnClearHistory = document.getElementById('btn-clear-history');
  const elBtnRefreshHistory = document.getElementById('btn-refresh-history');
  const elHistorySearch = document.getElementById('history-search');
  const elHistoryList = document.getElementById('history-list');
  const elConnectionStatus = document.getElementById('connection-status');
  
  const elTaskTitleDisplay = document.getElementById('task-title-display');
  const elTaskIdBadge = document.getElementById('task-id-badge');
  const elMetricSteps = document.getElementById('metric-steps');
  const elMetricTokens = document.getElementById('metric-tokens');
  const elMetricCost = document.getElementById('metric-cost');
  const elMetricDuration = document.getElementById('metric-duration');
  const elBtnExportJson = document.getElementById('btn-export-json');
  
  const elWorkspaceBody = document.getElementById('workspace-body');
  const elEventsContainer = document.getElementById('events-container');
  const elEmptyState = document.getElementById('empty-state');
  const elStreamSearch = document.getElementById('stream-search-input');
  const elFilterPills = document.getElementById('filter-pills');
  
  const elTerminalOutput = document.getElementById('terminal-output');
  const elBtnCopyTerminal = document.getElementById('btn-copy-terminal');
  const elBtnClearTerminal = document.getElementById('btn-clear-terminal');
  
  const elComposerForm = document.getElementById('composer-form');
  const elPromptInput = document.getElementById('prompt-input');
  const elBtnSubmit = document.getElementById('btn-submit');

  // --- INITIALIZATION ---
  function init() {
    setupEventListeners();
    fetchTasks();
  }

  // --- EVENT LISTENERS ---
  function setupEventListeners() {
    // Sidebar toggle
    elBtnToggleSidebar.addEventListener('click', () => {
      elSidebar.classList.toggle('collapsed');
    });

    // New task
    elBtnNewTask.addEventListener('click', () => {
      resetActiveSession();
    });

    // Refresh history
    elBtnRefreshHistory.addEventListener('click', fetchTasks);

    // Clear all history
    elBtnClearHistory.addEventListener('click', handleClearAllHistory);

    // Search history
    elHistorySearch.addEventListener('input', renderHistoryList);

    // View switchers
    document.querySelectorAll('.view-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.view-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        activeView = btn.dataset.view;
        elWorkspaceBody.className = 'workspace-body view-' + activeView;
      });
    });

    // Filter pills
    elFilterPills.addEventListener('click', (e) => {
      const pill = e.target.closest('.filter-pill');
      if (!pill) return;
      document.querySelectorAll('.filter-pill').forEach((p) => p.classList.remove('active'));
      pill.classList.add('active');
      activeFilter = pill.dataset.filter;
      renderEvents();
    });

    // Stream search
    elStreamSearch.addEventListener('input', renderEvents);

    // Export trajectory
    elBtnExportJson.addEventListener('click', handleExportTrajectory);

    // Terminal actions
    elBtnCopyTerminal.addEventListener('click', () => {
      const text = rawLogs.map((l) => `[${l.time}] ${l.raw}`).join('\n');
      navigator.clipboard.writeText(text);
      elBtnCopyTerminal.textContent = 'Copied!';
      setTimeout(() => (elBtnCopyTerminal.textContent = 'Copy'), 1500);
    });

    elBtnClearTerminal.addEventListener('click', () => {
      rawLogs = [];
      elTerminalOutput.innerHTML = '';
    });

    // Quick prompt chips
    document.addEventListener('click', (e) => {
      const chip = e.target.closest('.prompt-chip');
      if (chip && chip.dataset.prompt) {
        elPromptInput.value = chip.dataset.prompt;
        elPromptInput.focus();
        adjustTextareaHeight();
      }
    });

    // Composer textarea auto-grow & keyboard submit
    elPromptInput.addEventListener('input', adjustTextareaHeight);
    elPromptInput.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        elComposerForm.requestSubmit();
      }
    });

    // Form submit
    elComposerForm.addEventListener('submit', handleTaskSubmit);
  }

  function adjustTextareaHeight() {
    elPromptInput.style.height = 'auto';
    elPromptInput.style.height = Math.min(elPromptInput.scrollHeight, 160) + 'px';
  }

  // --- WEBSOCKET CLIENT ---
  function connectWebSocket(taskId) {
    if (ws) {
      ws.close();
      ws = null;
    }
    clearTimeout(reconnectTimer);

    updateConnectionStatus('connecting', 'Connecting...');
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws?task_id=${encodeURIComponent(taskId)}`;

    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        updateConnectionStatus('connected', 'Live Connected');
        appendTerminalLine('system', `Connected to WebSocket stream for task ${taskId.slice(0, 8)}...`);
      };

      ws.onmessage = (event) => {
        const raw = event.data;
        const time = new Date().toLocaleTimeString();
        rawLogs.push({ time, raw });
        appendTerminalLine('event', raw);

        try {
          const parsed = JSON.parse(raw);
          handleAgentEvent(parsed);
        } catch (err) {
          console.warn('Non-JSON WebSocket message:', raw);
        }
      };

      ws.onclose = (event) => {
        updateConnectionStatus('disconnected', 'Disconnected');
        appendTerminalLine('system', `WebSocket connection closed (code: ${event.code})`);
        // Refresh detail on finish
        setTimeout(() => {
          if (activeTaskId === taskId) {
            fetchTaskDetail(taskId);
            fetchTasks();
          }
        }, 1000);
      };

      ws.onerror = (err) => {
        updateConnectionStatus('disconnected', 'Connection Error');
        console.error('WebSocket error:', err);
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      updateConnectionStatus('disconnected', 'Failed');
    }
  }

  function updateConnectionStatus(state, label) {
    elConnectionStatus.innerHTML = `
      <span class="status-dot ${state}"></span>
      <span class="status-text">${label}</span>
    `;
  }

  // --- AGENT EVENT DISPATCHER ---
  function handleAgentEvent(evt) {
    events.push(evt);
    updateMetricsFromEvent(evt);
    renderEvents();
    if (elEmptyState) {
      elEmptyState.style.display = 'none';
    }
  }

  function updateMetricsFromEvent(evt) {
    const steps = events.filter((e) => 'step' in e).length;
    elMetricSteps.textContent = steps;

    if (evt.event_type === 'task_complete') {
      elTaskIdBadge.textContent = 'COMPLETED';
      elTaskIdBadge.className = 'task-id-badge status-tag COMPLETED';
      if (evt.total_tokens) elMetricTokens.textContent = evt.total_tokens.toLocaleString();
      if (evt.total_cost_usd !== undefined) elMetricCost.textContent = '$' + evt.total_cost_usd.toFixed(5);
      if (evt.duration_seconds !== undefined) elMetricDuration.textContent = evt.duration_seconds.toFixed(2) + 's';
    } else if (evt.event_type === 'error') {
      elTaskIdBadge.textContent = 'FAILED';
      elTaskIdBadge.className = 'task-id-badge status-tag FAILED';
    } else if (activeTaskId) {
      elTaskIdBadge.textContent = 'RUNNING';
      elTaskIdBadge.className = 'task-id-badge status-tag RUNNING';
    }
  }

  // --- RENDER EVENTS (AGENT STREAM) ---
  function renderEvents() {
    const query = elStreamSearch.value.trim().toLowerCase();
    const filtered = events.filter((evt) => {
      if (activeFilter !== 'all' && evt.event_type !== activeFilter) {
        return false;
      }
      if (query) {
        return JSON.stringify(evt).toLowerCase().includes(query);
      }
      return true;
    });

    if (events.length === 0) {
      elEmptyState.style.display = 'flex';
      return;
    }
    elEmptyState.style.display = 'none';

    // Remove existing event cards
    const existingCards = elEventsContainer.querySelectorAll('.event-card');
    existingCards.forEach((c) => c.remove());

    filtered.forEach((evt, idx) => {
      const card = createEventCard(evt, idx);
      elEventsContainer.appendChild(card);
    });

    // Auto-scroll bottom
    elEventsContainer.scrollTop = elEventsContainer.scrollHeight;
  }

  function createEventCard(evt, index) {
    const card = document.createElement('div');
    card.className = `event-card event-${evt.event_type}`;
    const time = evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : '';

    if (evt.event_type === 'thought') {
      card.innerHTML = `
        <div class="event-header">
          <div class="event-header-left">
            <span class="step-chip">Step ${evt.step}</span>
            <span class="event-type-badge badge-thought">Reasoning Thought</span>
          </div>
          <span class="event-time">${time}</span>
        </div>
        <div class="event-body thought-content">${escapeHtml(evt.thought)}</div>
      `;
    } else if (evt.event_type === 'tool_call') {
      const argsJson = JSON.stringify(evt.arguments, null, 2);
      card.innerHTML = `
        <div class="event-header">
          <div class="event-header-left">
            <span class="step-chip">Step ${evt.step}</span>
            <span class="event-type-badge badge-tool_call">Action</span>
            <span class="event-tool-name">${escapeHtml(evt.tool_name)}</span>
          </div>
          <span class="event-time">${time}</span>
        </div>
        <div class="event-body">
          <pre class="code-container"><code>${escapeHtml(argsJson)}</code></pre>
        </div>
      `;
    } else if (evt.event_type === 'tool_output') {
      card.innerHTML = `
        <div class="event-header">
          <div class="event-header-left">
            <span class="step-chip">Step ${evt.step}</span>
            <span class="event-type-badge badge-tool_output">Observation</span>
            <span class="event-tool-name">${escapeHtml(evt.tool_name)}</span>
          </div>
          <span class="event-time">${time}</span>
        </div>
        <div class="event-body">
          <pre class="code-container"><code>${escapeHtml(evt.output)}</code></pre>
        </div>
      `;
    } else if (evt.event_type === 'task_complete') {
      card.className += ' complete-card';
      card.innerHTML = `
        <div class="event-header">
          <div class="event-header-left">
            <span class="event-type-badge badge-task_complete">Task Accomplished</span>
          </div>
          <span class="event-time">${time}</span>
        </div>
        <div class="event-body">
          <div class="final-answer-box">${escapeHtml(evt.final_answer || 'Task completed successfully.')}</div>
        </div>
      `;
    } else if (evt.event_type === 'error') {
      card.innerHTML = `
        <div class="event-header">
          <div class="event-header-left">
            <span class="event-type-badge badge-error">Error</span>
          </div>
          <span class="event-time">${time}</span>
        </div>
        <div class="event-body">
          <div class="code-container" style="color: var(--color-error);">${escapeHtml(evt.error)}</div>
        </div>
      `;
    }

    return card;
  }

  // --- TERMINAL LOGGING ---
  function appendTerminalLine(type, text) {
    const line = document.createElement('div');
    line.className = `terminal-line ${type}`;
    const time = new Date().toLocaleTimeString();

    if (type === 'system') {
      line.innerHTML = `<span class="terminal-time">[${time}]</span> <span class="terminal-prompt">$</span> ${escapeHtml(text)}`;
    } else {
      try {
        const parsed = JSON.parse(text);
        const tag = parsed.event_type || 'event';
        line.innerHTML = `<span class="terminal-time">[${time}]</span> <span class="badge-${tag}">[${tag}]</span> ${escapeHtml(text)}`;
      } catch {
        line.innerHTML = `<span class="terminal-time">[${time}]</span> ${escapeHtml(text)}`;
      }
    }

    elTerminalOutput.appendChild(line);
    elTerminalOutput.scrollTop = elTerminalOutput.scrollHeight;
  }

  // --- TASK SUBMISSION ---
  async function handleTaskSubmit(e) {
    e.preventDefault();
    const prompt = elPromptInput.value.trim();
    if (!prompt) return;

    elBtnSubmit.disabled = true;
    elBtnSubmit.innerHTML = '<span>Dispatching...</span>';

    try {
      const res = await fetch('/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: prompt, metadata: { source: 'lovable-vanilla-client' } }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(`Failed to submit task: ${err.detail || res.statusText}`);
        return;
      }

      const data = await res.json();
      selectTask(data.task_id, prompt);
      elPromptInput.value = '';
      adjustTextareaHeight();
      fetchTasks();
    } catch (err) {
      console.error('Task submission error:', err);
      alert(`Submission error: ${err}`);
    } finally {
      elBtnSubmit.disabled = false;
      elBtnSubmit.innerHTML = `
        <span class="btn-text">Run Agent</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      `;
    }
  }

  // --- TASK HISTORY & STATE MANAGEMENT ---
  async function fetchTasks() {
    try {
      const res = await fetch('/tasks?limit=50');
      if (res.ok) {
        tasks = await res.json();
        renderHistoryList();
      }
    } catch (err) {
      console.error('Failed to fetch tasks:', err);
    }
  }

  function renderHistoryList() {
    const q = elHistorySearch.value.trim().toLowerCase();
    const filtered = tasks.filter((t) => !q || t.task.toLowerCase().includes(q) || t.task_id.includes(q));

    if (filtered.length === 0) {
      elHistoryList.innerHTML = `<div class="history-empty">${q ? 'No matching tasks.' : 'No tasks in history.'}</div>`;
      return;
    }

    elHistoryList.innerHTML = '';
    filtered.forEach((t) => {
      const item = document.createElement('div');
      item.className = `history-item ${t.task_id === activeTaskId ? 'active' : ''}`;
      item.onclick = () => selectTask(t.task_id, t.task);

      const costText = t.total_cost_usd > 0 ? `$${t.total_cost_usd.toFixed(5)}` : '';
      const tokensText = t.total_tokens > 0 ? `${t.total_tokens.toLocaleString()} tok` : '';

      item.innerHTML = `
        <div class="history-item-header">
          <span class="status-tag ${t.status}">${t.status}</span>
          <button type="button" class="delete-task-btn" title="Delete Task" onclick="event.stopPropagation(); window.deleteTask('${t.task_id}')">✕</button>
        </div>
        <div class="history-title" title="${escapeHtml(t.task)}">${escapeHtml(t.task)}</div>
        <div class="history-meta">
          <span class="history-cost">${costText}</span>
          <span>${tokensText}</span>
        </div>
      `;
      elHistoryList.appendChild(item);
    });
  }

  window.deleteTask = async function (taskId) {
    try {
      const res = await fetch(`/tasks/${encodeURIComponent(taskId)}`, { method: 'DELETE' });
      if (res.ok) {
        tasks = tasks.filter((t) => t.task_id !== taskId);
        if (activeTaskId === taskId) {
          resetActiveSession();
        }
        renderHistoryList();
      }
    } catch (err) {
      console.error('Failed to delete task:', err);
    }
  };

  async function handleClearAllHistory() {
    if (!confirm('Are you sure you want to clear all tasks from the database?')) return;
    try {
      const res = await fetch('/tasks', { method: 'DELETE' });
      if (res.ok) {
        tasks = [];
        resetActiveSession();
        renderHistoryList();
      }
    } catch (err) {
      console.error('Error clearing tasks:', err);
    }
  }

  async function fetchTaskDetail(taskId) {
    try {
      const res = await fetch(`/tasks/${encodeURIComponent(taskId)}`);
      if (res.ok) {
        activeTaskDetail = await res.json();
        elTaskTitleDisplay.textContent = activeTaskDetail.task;
        elTaskIdBadge.textContent = activeTaskDetail.status;
        elTaskIdBadge.className = `task-id-badge status-tag ${activeTaskDetail.status}`;
        elMetricTokens.textContent = (activeTaskDetail.total_tokens || 0).toLocaleString();
        elMetricCost.textContent = '$' + (activeTaskDetail.total_cost_usd || 0).toFixed(5);
        elMetricDuration.textContent = (activeTaskDetail.duration_seconds || 0).toFixed(2) + 's';
      }
    } catch (err) {
      console.error('Failed to fetch detail:', err);
    }
  }

  function selectTask(taskId, taskText) {
    activeTaskId = taskId;
    events = [];
    rawLogs = [];
    elTerminalOutput.innerHTML = '';
    elTaskTitleDisplay.textContent = taskText || `Task ${taskId.slice(0, 8)}`;
    elTaskIdBadge.textContent = 'RUNNING';
    elTaskIdBadge.className = 'task-id-badge status-tag RUNNING';
    
    elMetricSteps.textContent = '0';
    elMetricTokens.textContent = '0';
    elMetricCost.textContent = '$0.00000';
    elMetricDuration.textContent = '0.0s';

    renderHistoryList();
    renderEvents();
    connectWebSocket(taskId);
    fetchTaskDetail(taskId);
  }

  function resetActiveSession() {
    activeTaskId = null;
    activeTaskDetail = null;
    events = [];
    rawLogs = [];
    if (ws) {
      ws.close();
      ws = null;
    }
    updateConnectionStatus('disconnected', 'Idle');
    elTaskTitleDisplay.textContent = 'New Autonomous Session';
    elTaskIdBadge.textContent = 'IDLE';
    elTaskIdBadge.className = 'task-id-badge';
    elMetricSteps.textContent = '0';
    elMetricTokens.textContent = '0';
    elMetricCost.textContent = '$0.00000';
    elMetricDuration.textContent = '0.0s';
    
    renderEvents();
    renderHistoryList();
  }

  function handleExportTrajectory() {
    const data = {
      taskId: activeTaskId,
      taskDetail: activeTaskDetail,
      events,
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `talos-trajectory-${activeTaskId || 'session'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Run on load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
