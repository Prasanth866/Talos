import { useState, useEffect, useCallback } from 'react'
import { useTaskWebSocket } from './hooks/useTaskWebSocket'
import { TaskSubmitForm } from './components/TaskSubmitForm'
import { TaskHistory } from './components/TaskHistory'
import { EventStream } from './components/EventStream'
import { TerminalPane } from './components/TerminalPane'
import { MetricsHud } from './components/MetricsHud'
import { TaskDetailsPane } from './components/TaskDetailsPane'
import { StatusBadge } from './components/StatusBadge'
import type { TaskDetailResponse, TaskSubmitResponse } from './types/events'
import './styles/index.css'

export function App() {
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null)
  const [activeTaskDetail, setActiveTaskDetail] = useState<TaskDetailResponse | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [refreshHistoryTrigger, setRefreshHistoryTrigger] = useState(0)
  const [activeView, setActiveView] = useState<'split' | 'stream' | 'terminal' | 'details'>('split')

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

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <div>
            <div className="brand-title-row">
              <h1 className="brand-title">Talos Agent Observer</h1>
              <span className="brand-badge">v0.1.0</span>
            </div>
            <p className="brand-subtitle">
              Live Reasoning Loop, Tool Dispatch &amp; Token Persistence Monitor
            </p>
          </div>
        </div>

        <div className="header-status">
          <div className="conn-status-wrapper">
            <span className="conn-label">WebSocket:</span>
            <StatusBadge type="connection" status={connStatus} />
          </div>
          {connStatus === 'disconnected' && activeTaskId && (
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={reconnect}
            >
              Reconnect
            </button>
          )}
        </div>

      </header>

      {/* Main Workspace */}
      <main className="app-main">
        {/* Left Sidebar */}
        <aside className="left-sidebar">
          <TaskSubmitForm
            onSubmit={handleTaskSubmit}
            submitting={submitting}
            activeTaskId={activeTaskId}
            onClearActiveTask={handleClearActiveTask}
          />
          <TaskHistory
            activeTaskId={activeTaskId}
            onSelectTask={handleSelectTask}
            onClearActiveTask={handleClearActiveTask}
            refreshTrigger={refreshHistoryTrigger}
          />

        </aside>

        {/* Right Workspace */}
        <section className="right-workspace">
          {/* Metrics HUD Header */}
          <MetricsHud
            taskId={activeTaskId}
            taskDetail={activeTaskDetail}
            events={events}
            activeView={activeView}
            onChangeView={setActiveView}
          />

          {/* Dynamic Views */}
          {activeView === 'details' ? (
            <TaskDetailsPane
              taskId={activeTaskId}
              taskDetail={activeTaskDetail}
            />
          ) : activeView === 'stream' ? (
            <div className="single-pane-view">
              <EventStream events={events} taskId={activeTaskId} />
            </div>
          ) : activeView === 'terminal' ? (
            <div className="single-pane-view">
              <TerminalPane logs={rawLogs} onClear={clearEvents} />
            </div>
          ) : (
            /* Split View */
            <div className="workspace-panes">
              <EventStream events={events} taskId={activeTaskId} />
              <TerminalPane logs={rawLogs} onClear={clearEvents} />
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
