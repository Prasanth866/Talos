import React from 'react'
import type { AgentEvent, TaskDetailResponse, TaskStatus } from '../types/events'
import { StatusBadge } from './StatusBadge'

interface MetricsHudProps {
  taskId: string | null
  taskDetail: TaskDetailResponse | null
  events: AgentEvent[]
}

export const MetricsHud: React.FC<MetricsHudProps> = ({ taskId, taskDetail, events }) => {
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

  if (!taskId) {
    return (
      <div className="metrics-hud metrics-hud-empty">
        <span className="metrics-placeholder">No active task selected</span>
      </div>
    )
  }

  return (
    <div className="metrics-hud">
      <div className="metric-item">
        <span className="metric-label">Status</span>
        <StatusBadge type="task" status={taskStatus} />
      </div>

      <div className="metric-divider" />

      <div className="metric-item">
        <span className="metric-label">Steps</span>
        <span className="metric-value">{stepsCount}</span>
      </div>

      <div className="metric-divider" />

      <div className="metric-item">
        <span className="metric-label">Tokens</span>
        <div className="metric-value-group">
          <span className="metric-value">{totalTokens.toLocaleString()}</span>
          {totalTokens > 0 && promptTokens > 0 && (
            <span className="metric-sub">
              ({promptTokens} in / {completionTokens} out)
            </span>
          )}
        </div>
      </div>

      <div className="metric-divider" />

      <div className="metric-item">
        <span className="metric-label">Cost</span>
        <span className="metric-value highlight-cost">
          ${totalCost.toFixed(5)}
        </span>
      </div>

      <div className="metric-divider" />

      <div className="metric-item">
        <span className="metric-label">Duration</span>
        <span className="metric-value">{duration.toFixed(2)}s</span>
      </div>
    </div>
  )
}
