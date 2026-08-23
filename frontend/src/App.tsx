import { useState, useEffect, useCallback } from 'react'
import { useTaskWebSocket } from './hooks/useTaskWebSocket'
import { TaskSubmitForm } from './components/TaskSubmitForm'
import { TaskHistory } from './components/TaskHistory'
import { EventStream } from './components/EventStream'
import { TerminalPane } from './components/TerminalPane'
import { MetricsHud } from './components/MetricsHud'
import { StatusBadge } from './components/StatusBadge'
import type { TaskDetailResponse, TaskSubmitResponse } from './types/events'
import './styles/index.css'

export function App() {
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null)
  const [activeTaskDetail, setActiveTaskDetail] = useState<TaskDetailResponse | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [refreshHistoryTrigger, setRefreshHistoryTrigger] = useState(0)

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
      // Refresh DB record to get updated token costs and final duration
      setTimeout(() => {
        fetchTaskDetail(activeTaskId)
        setRefreshHistoryTrigger((prev) => prev + 1)
      }, 500)
    }
  }, [activeTaskId, fetchTaskDetail])

  const { events, rawLogs, status: connStatus, clearEvents } = useTaskWebSocket({
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

  const handleTaskSubmit = async (prompt: string) => {
    setSubmitting(true)
    try {
      const res = await fetch('/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: prompt, metadata: { source: 'react-frontend' } }),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        alert(`Failed to submit task: ${errData.detail || res.statusText}`)
        return
      }

      const data = (await res.json()) as TaskSubmitResponse
      setActiveTaskId(data.task_id)
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

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-section">
          <span className="brand-logo">🛡️</span>
          <div>
            <h1 className="brand-title">Talos Agent Observer</h1>
            <p className="brand-subtitle">
              Live Reasoning Loop &amp; Tool Dispatch Monitor
            </p>
          </div>
        </div>

        <div className="header-status">
          <span className="conn-label">WebSocket:</span>
          <StatusBadge type="connection" status={connStatus} />
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
          />
          <TaskHistory
            activeTaskId={activeTaskId}
            onSelectTask={handleSelectTask}
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
          />

          {/* Split Panes: Event Stream & Terminal */}
          <div className="workspace-panes">
            <EventStream events={events} taskId={activeTaskId} />
            <TerminalPane logs={rawLogs} onClear={clearEvents} />
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
