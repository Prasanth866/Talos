import React, { useState } from 'react'
import type { AgentEvent, TaskDetailResponse, TaskStatus } from '../types/events'
import { StatusBadge } from './StatusBadge'

interface MetricsHudProps {
  taskId: string | null
  taskDetail: TaskDetailResponse | null
  events: AgentEvent[]
  activeView: 'split' | 'stream' | 'terminal' | 'details'
  onChangeView: (view: 'split' | 'stream' | 'terminal' | 'details') => void
}

export const MetricsHud: React.FC<MetricsHudProps> = ({
  taskId,
  taskDetail,
  events,
  activeView,
  onChangeView,
}) => {
  const [downloaded, setDownloaded] = useState(false)

  // Extract completed event if present
  const completeEvent = events.find(
    (e) => e.event_type === 'task_complete'
  ) as Extract<AgentEvent, { event_type: 'task_complete' }> | undefined

  const errorEvent = events.find(
    (e) => e.event_type === 'error'
  ) as Extract<AgentEvent, { event_type: 'error' }> | undefined

  // Derive status
  let taskStatus: TaskStatus = 'PENDING'
  if (taskDetail) {
    taskStatus = taskDetail.status
  }
  if (completeEvent) {
    taskStatus = 'COMPLETED'
  } else if (errorEvent) {
    taskStatus = 'FAILED'
  } else if (events.length > 0 && taskStatus === 'PENDING') {
    taskStatus = 'RUNNING'
  }

  // Derive steps
  const stepsCount = completeEvent?.total_steps || events.filter((e) => 'step' in e).length

  // Derive tokens & cost
  const totalTokens = completeEvent?.total_tokens ?? taskDetail?.total_tokens ?? 0
  const promptTokens = taskDetail?.prompt_tokens ?? 0
  const completionTokens = taskDetail?.completion_tokens ?? 0
  const totalCost = completeEvent?.total_cost_usd ?? taskDetail?.total_cost_usd ?? 0
  const duration = completeEvent?.duration_seconds ?? taskDetail?.duration_seconds ?? 0

  const handleExportJson = () => {
    const data = {
      taskId,
      status: taskStatus,
      detail: taskDetail,
      events,
      exportedAt: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `talos-trajectory-${taskId || 'export'}.json`
    a.click()
    URL.revokeObjectURL(url)
    setDownloaded(true)
    setTimeout(() => setDownloaded(false), 2000)
  }

  return (
    <div className="metrics-hud-card">
      <div className="metrics-left">
        <div className="metric-chip">
          <span className="metric-label">Status</span>
          <StatusBadge type="task" status={taskStatus} />
        </div>

        <div className="metric-divider" />

        <div className="metric-chip">
          <span className="metric-label">Steps</span>
          <span className="metric-val">{stepsCount}</span>
        </div>

        <div className="metric-divider" />

        <div className="metric-chip">
          <span className="metric-label">Tokens</span>
          <div className="metric-tokens-group">
            <span className="metric-val">{totalTokens.toLocaleString()}</span>
            {promptTokens > 0 && (
              <span className="metric-sub-val">
                ({promptTokens.toLocaleString()} in / {completionTokens.toLocaleString()} out)
              </span>
            )}
          </div>
        </div>

        <div className="metric-divider" />

        <div className="metric-chip">
          <span className="metric-label">Estimated Cost</span>
          <span className="metric-val highlight-cost">
            ${totalCost.toFixed(5)}
          </span>
        </div>

        <div className="metric-divider" />

        <div className="metric-chip">
          <span className="metric-label">Duration</span>
          <span className="metric-val">{duration.toFixed(2)}s</span>
        </div>
      </div>

      <div className="metrics-right">
        {/* View Mode Switcher */}
        <div className="view-mode-group">
          <button
            type="button"
            className={`view-btn ${activeView === 'split' ? 'active' : ''}`}
            onClick={() => onChangeView('split')}
            title="Split view (Event stream + Terminal)"
          >
            Split
          </button>
          <button
            type="button"
            className={`view-btn ${activeView === 'stream' ? 'active' : ''}`}
            onClick={() => onChangeView('stream')}
            title="Event stream only"
          >
            Stream
          </button>
          <button
            type="button"
            className={`view-btn ${activeView === 'terminal' ? 'active' : ''}`}
            onClick={() => onChangeView('terminal')}
            title="Terminal logs only"
          >
            Terminal
          </button>
          <button
            type="button"
            className={`view-btn ${activeView === 'details' ? 'active' : ''}`}
            onClick={() => onChangeView('details')}
            title="Inspect Task Database Record & Metadata"
          >
            Details
          </button>
        </div>

        {taskId && (
          <button
            type="button"
            className="btn btn-secondary btn-sm export-btn"
            onClick={handleExportJson}
            title="Export trajectory to JSON"
          >
            {downloaded ? 'Saved' : 'Export JSON'}
          </button>
        )}

      </div>
    </div>
  )
}
