import { useState, useEffect, useCallback } from 'react'
import { useTaskWebSocket } from './hooks/useTaskWebSocket'
import { TaskSubmitForm } from './components/TaskSubmitForm'
import { TaskHistory } from './components/TaskHistory'
import { EventStream } from './components/EventStream'
import { TerminalPane } from './components/TerminalPane'
import { MetricsHud } from './components/MetricsHud'
import { TaskDetailsPane } from './components/TaskDetailsPane'
import { StatusBadge } from './components/StatusBadge'
import type { TaskDetailResponse, TaskSubmitResponse, TaskStatus } from './types/events'
import './styles/index.css'

export function App() {
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null)
  const [activeTaskDetail, setActiveTaskDetail] = useState<TaskDetailResponse | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [refreshHistoryTrigger, setRefreshHistoryTrigger] = useState(0)
  const [activeView, setActiveView] = useState<'split' | 'stream' | 'terminal' | 'details'>('split')
  const [activeActivityTab, setActiveActivityTab] = useState<'explorer' | 'run' | 'history' | 'terminal'>('explorer')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [panelCollapsed, setPanelCollapsed] = useState(false)

  // Fetch task detail when activeTaskId changes
  const fetchTaskDetail = useCallback(async (taskId: string) => {
    try {
      const res = await fetch(`/tasks/${encodeURIComponent(taskId)}`)
      if (res.ok) {
        const data = (await res.json()) as TaskDetailResponse
        setActiveTaskDetail(data)
      }
    } catch (err) {
      console.error('Failed to fetch task detail:', err)
    }
  }, [])

  const handleTaskFinished = useCallback(() => {
    if (activeTaskId) {
      setTimeout(() => {
        fetchTaskDetail(activeTaskId)
        setRefreshHistoryTrigger((prev) => prev + 1)
      }, 500)
    }
  }, [activeTaskId, fetchTaskDetail])

  const { events, rawLogs, status: connStatus, clearEvents, reconnect } = useTaskWebSocket({
    taskId: activeTaskId,
    onTaskFinished: handleTaskFinished,
  })

  useEffect(() => {
    if (activeTaskId) {
      fetchTaskDetail(activeTaskId)
    } else {
      setActiveTaskDetail(null)
    }
  }, [activeTaskId, fetchTaskDetail])

  const handleTaskSubmit = async (prompt: string, metadata: Record<string, unknown> = {}) => {
    setSubmitting(true)
    try {
      const res = await fetch('/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task: prompt,
          metadata: { ...metadata, source: 'react-frontend' },
        }),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        alert(`Failed to submit task: ${errData.detail || res.statusText}`)
        return
      }

      const data = (await res.json()) as TaskSubmitResponse
      setActiveTaskId(data.task_id)
      setActiveView('split')
      setRefreshHistoryTrigger((prev) => prev + 1)
    } catch (err) {
      console.error('Error submitting task:', err)
      alert(`Network error: ${err}`)
    } finally {
      setSubmitting(false)
    }
  }

  const handleSelectTask = (taskId: string) => {
    setActiveTaskId(taskId)
  }

  const handleClearActiveTask = () => {
    setActiveTaskId(null)
    setActiveTaskDetail(null)
  }

  // Calculate metrics for status bar
  const totalTokens = activeTaskDetail?.total_tokens ?? 0
  const totalCost = activeTaskDetail?.total_cost_usd ?? 0
  const taskStatus: TaskStatus = activeTaskDetail?.status ?? (activeTaskId ? 'RUNNING' : 'PENDING')

  return (
    <div className="vscode-window">
      {/* 1. VS Code Clean Title Bar */}
      <header className="vscode-titlebar">
        <div className="titlebar-left">
          <div className="vscode-app-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0078d4" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
          </div>
          <div className="app-title-text">
            <span className="app-main-title">Talos Agent</span>
            <span className="app-version-badge">v0.1.0</span>
          </div>
        </div>

        {/* Command Center */}
        <div className="titlebar-center">
          <div className="command-center-box" onClick={() => setActiveView('split')}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8"/>
              <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <span className="command-title">
              talos-workspace {activeTaskId ? `— task: ${activeTaskId.slice(0, 8)}...` : '— (No active task)'}
            </span>
            <span className="command-shortcut">Ctrl+P</span>
          </div>
        </div>

        <div className="titlebar-right">
          <div className="titlebar-actions">
            <button
              type="button"
              className={`titlebar-icon-btn ${!sidebarCollapsed ? 'active' : ''}`}
              onClick={() => setSidebarCollapsed((prev) => !prev)}
              title="Toggle Primary Side Bar (Ctrl+B)"
            >
              <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                <path d="M2 3h12a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zm4 1H2v8h4V4zm1 8h7V4H7v8z"/>
              </svg>
            </button>
            <button
              type="button"
              className={`titlebar-icon-btn ${!panelCollapsed ? 'active' : ''}`}
              onClick={() => setPanelCollapsed((prev) => !prev)}
              title="Toggle Bottom Panel (Ctrl+J)"
            >
              <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                <path d="M2 3h12a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zm0 1v5h12V4H2zm12 6H2v2h12v-2z"/>
              </svg>
            </button>
          </div>

          <div className="titlebar-conn-badge">
            <span className={`conn-indicator-dot ${connStatus}`} />
            <span className="conn-status-text">WS: {connStatus}</span>
            {connStatus === 'disconnected' && activeTaskId && (
              <button
                type="button"
                className="titlebar-reconnect-btn"
                onClick={reconnect}
              >
                Reconnect
              </button>
            )}
          </div>
        </div>
      </header>

      {/* 2. Main Workbench Shell */}
      <div className="vscode-workbench">
        {/* Activity Bar (Far Left 48px) */}
        <aside className="vscode-activity-bar">
          <div className="activity-top-icons">
            <button
              type="button"
              className={`activity-icon-btn ${activeActivityTab === 'explorer' && !sidebarCollapsed ? 'active' : ''}`}
              onClick={() => {
                if (activeActivityTab === 'explorer' && !sidebarCollapsed) {
                  setSidebarCollapsed(true)
                } else {
                  setActiveActivityTab('explorer')
                  setSidebarCollapsed(false)
                }
              }}
              title="Explorer: All Panels"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
                <polyline points="13 2 13 9 20 9"/>
              </svg>
            </button>

            <button
              type="button"
              className={`activity-icon-btn ${activeActivityTab === 'run' && !sidebarCollapsed ? 'active' : ''}`}
              onClick={() => {
                if (activeActivityTab === 'run' && !sidebarCollapsed) {
                  setSidebarCollapsed(true)
                } else {
                  setActiveActivityTab('run')
                  setSidebarCollapsed(false)
                }
              }}
              title="Run & Dispatch Agent"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="5 3 19 12 5 21 5 3"/>
              </svg>
            </button>

            <button
              type="button"
              className={`activity-icon-btn ${activeActivityTab === 'history' && !sidebarCollapsed ? 'active' : ''}`}
              onClick={() => {
                if (activeActivityTab === 'history' && !sidebarCollapsed) {
                  setSidebarCollapsed(true)
                } else {
                  setActiveActivityTab('history')
                  setSidebarCollapsed(false)
                }
              }}
              title="Task History & Timeline"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <polyline points="12 6 12 12 16 14"/>
              </svg>
            </button>

            <button
              type="button"
              className={`activity-icon-btn ${activeView === 'terminal' ? 'active' : ''}`}
              onClick={() => {
                setActiveView('terminal')
                setPanelCollapsed(false)
              }}
              title="Terminal Output"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="4 17 10 11 4 5"/>
                <line x1="12" y1="19" x2="20" y2="19"/>
              </svg>
            </button>
          </div>
        </aside>

        {/* Primary Sidebar (Collapsible) */}
        {!sidebarCollapsed && (
          <aside className="vscode-sidebar">
            <div className="sidebar-header">
              <span className="sidebar-title">
                {activeActivityTab === 'run'
                  ? 'RUN & DISPATCH'
                  : activeActivityTab === 'history'
                  ? 'TIMELINE & HISTORY'
                  : 'EXPLORER: TALOS AGENT'}
              </span>
              <div className="sidebar-header-actions">
                <button
                  type="button"
                  className="sidebar-action-btn"
                  onClick={() => setSidebarCollapsed(true)}
                  title="Collapse Side Bar"
                >
                  <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M11.354 1.646a.5.5 0 0 1 0 .708L5.707 8l5.647 5.646a.5.5 0 0 1-.708.708l-6-6a.5.5 0 0 1 0-.708l6-6a.5.5 0 0 1 .708 0z"/>
                  </svg>
                </button>
              </div>
            </div>

            <div className="sidebar-content">
              {(activeActivityTab === 'explorer' || activeActivityTab === 'run') && (
                <TaskSubmitForm
                  onSubmit={handleTaskSubmit}
                  submitting={submitting}
                  activeTaskId={activeTaskId}
                  onClearActiveTask={handleClearActiveTask}
                />
              )}

              {(activeActivityTab === 'explorer' || activeActivityTab === 'history') && (
                <TaskHistory
                  activeTaskId={activeTaskId}
                  onSelectTask={handleSelectTask}
                  onClearActiveTask={handleClearActiveTask}
                  refreshTrigger={refreshHistoryTrigger}
                />
              )}
            </div>
          </aside>
        )}

        {/* Editor & Main Working Area */}
        <section className="vscode-editor-group">
          {/* Editor Tab Bar */}
          <div className="editor-tab-bar">
            <div className="editor-tabs-scroll">
              <div
                className={`editor-tab ${activeView === 'split' ? 'active' : ''}`}
                onClick={() => setActiveView('split')}
              >
                <span className="file-icon ts-icon">TS</span>
                <span className="tab-label">reasoning_stream.ts</span>
                <span className="tab-badge">{events.length}</span>
              </div>

              <div
                className={`editor-tab ${activeView === 'stream' ? 'active' : ''}`}
                onClick={() => setActiveView('stream')}
              >
                <span className="file-icon ts-icon">TS</span>
                <span className="tab-label">reasoning_only.ts</span>
              </div>

              <div
                className={`editor-tab ${activeView === 'details' ? 'active' : ''}`}
                onClick={() => setActiveView('details')}
              >
                <span className="file-icon json-icon">&#123;&#125;</span>
                <span className="tab-label">task_details.json</span>
              </div>

              <div
                className={`editor-tab ${activeView === 'terminal' ? 'active' : ''}`}
                onClick={() => setActiveView('terminal')}
              >
                <span className="file-icon sh-icon">&gt;_</span>
                <span className="tab-label">agent_terminal.sh</span>
                <span className="tab-badge">{rawLogs.length}</span>
              </div>
            </div>

            <div className="editor-actions">
              <button
                type="button"
                className={`editor-action-btn ${activeView === 'split' ? 'active' : ''}`}
                onClick={() => setActiveView('split')}
                title="Split Editor (Event Stream + Terminal)"
              >
                <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                  <path d="M14 2H2a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V3a1 1 0 0 0-1-1zM2 3h5v10H2V3zm12 10H8V3h6v10z"/>
                </svg>
              </button>
            </div>
          </div>

          {/* Breadcrumb Navigation */}
          <div className="editor-breadcrumbs">
            <span className="breadcrumb-item">talos</span>
            <span className="breadcrumb-sep">&gt;</span>
            <span className="breadcrumb-item">src</span>
            <span className="breadcrumb-sep">&gt;</span>
            <span className="breadcrumb-item">agent</span>
            <span className="breadcrumb-sep">&gt;</span>
            <span className="breadcrumb-item active">
              {activeTaskId ? `task-${activeTaskId.slice(0, 8)}` : 'reasoning_loop.ts'}
            </span>
          </div>

          {/* Metrics HUD Subheader */}
          <MetricsHud
            taskId={activeTaskId}
            taskDetail={activeTaskDetail}
            events={events}
            activeView={activeView}
            onChangeView={setActiveView}
          />

          {/* Editor Body */}
          <div className="editor-view-container">
            {activeView === 'details' ? (
              <TaskDetailsPane
                taskId={activeTaskId}
                taskDetail={activeTaskDetail}
              />
            ) : activeView === 'stream' ? (
              <div className="single-editor-pane">
                <EventStream events={events} taskId={activeTaskId} />
              </div>
            ) : activeView === 'terminal' ? (
              <div className="single-editor-pane">
                <TerminalPane logs={rawLogs} onClear={clearEvents} />
              </div>
            ) : (
              /* Split View: Editor Top/Left + Integrated Panel Bottom/Right */
              <div className={`vscode-split-layout ${panelCollapsed ? 'panel-hidden' : ''}`}>
                <div className="vscode-editor-pane">
                  <EventStream events={events} taskId={activeTaskId} />
                </div>
                {!panelCollapsed && (
                  <div className="vscode-panel-pane">
                    <TerminalPane logs={rawLogs} onClear={clearEvents} />
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      </div>

      {/* 3. VS Code Bottom Status Bar */}
      <footer className="vscode-statusbar">
        <div className="statusbar-left">
          <div className="statusbar-item branch-item" title="Git Branch: main">
            <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
              <path d="M11.75 2a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5zm-7.5 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5zm0 8.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5zm5.75-5.25a.75.75 0 0 0-.75.75v3.69a2.5 2.5 0 0 1-1.5 2.29V6.25a2.25 2.25 0 1 0-1.5 0v5.69a2.5 2.5 0 0 1-1.5-2.29V5.75a.75.75 0 0 0-1.5 0v3.69a4 4 0 0 0 3 3.87v.44a.75.75 0 0 0 1.5 0v-.44a4 4 0 0 0 3-3.87V6.5a.75.75 0 0 0-.75-.75z"/>
            </svg>
            <span>main*</span>
          </div>

          <div className="statusbar-item sync-item" title="Synchronized with origin/main">
            <span>0&#8595; 0&#8593;</span>
          </div>

          <div className="statusbar-item status-chip">
            <span className="statusbar-label">Agent:</span>
            <StatusBadge type="task" status={taskStatus} />
          </div>
        </div>

        <div className="statusbar-right">
          <div className="statusbar-item" title="Total tokens consumed">
            <span className="stat-label">Tokens:</span>
            <span className="stat-val">{totalTokens.toLocaleString()}</span>
          </div>

          <div className="statusbar-item" title="Estimated cost in USD">
            <span className="stat-label">Cost:</span>
            <span className="stat-val">${totalCost.toFixed(5)}</span>
          </div>

          <div className="statusbar-item" title="FastAPI Backend Port">
            <span>Port: 8000</span>
          </div>

          <div className="statusbar-item" title="Encoding">
            <span>UTF-8</span>
          </div>

          <div className="statusbar-item" title="Language Mode">
            <span>TypeScript React</span>
          </div>

          <div className="statusbar-item" title="Formatter">
            <span>Prettier</span>
          </div>

          <div className="statusbar-item conn-item" title={`WebSocket Status: ${connStatus}`}>
            <span className={`status-dot ${connStatus}`} />
            <span>{connStatus}</span>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App

