import React from 'react'
import type { TaskDetailResponse } from '../types/events'
import { StatusBadge } from './StatusBadge'

interface TaskDetailsPaneProps {
  taskDetail: TaskDetailResponse | null
  taskId: string | null
}

export const TaskDetailsPane: React.FC<TaskDetailsPaneProps> = ({ taskDetail, taskId }) => {
  if (!taskId) {
    return (
      <div className="task-details-pane card">
        <div className="pane-empty-notice">
          Select or submit a task to inspect database persistence details.
        </div>
      </div>
    )
  }

  if (!taskDetail) {
    return (
      <div className="task-details-pane card">
        <div className="pane-empty-notice">
          <span className="spinner" /> Loading task details from database...
        </div>
      </div>
    )
  }

  return (
    <div className="task-details-pane card">
      <div className="pane-header">
        <div className="pane-title">
          <span>Task Database Record &amp; Metrics</span>
        </div>

        <StatusBadge type="task" status={taskDetail.status} />
      </div>

      <div className="details-grid-body">
        <div className="details-section">
          <h3>Task Overview</h3>
          <div className="details-table">
            <div className="details-row">
              <span className="dt-key">Task ID</span>
              <span className="dt-val font-mono">{taskDetail.task_id}</span>
            </div>
            <div className="details-row">
              <span className="dt-key">Prompt</span>
              <span className="dt-val">{taskDetail.task}</span>
            </div>
            <div className="details-row">
              <span className="dt-key">Created At</span>
              <span className="dt-val">{new Date(taskDetail.created_at).toLocaleString()}</span>
            </div>
            {taskDetail.started_at && (
              <div className="details-row">
                <span className="dt-key">Started At</span>
                <span className="dt-val">{new Date(taskDetail.started_at).toLocaleString()}</span>
              </div>
            )}
            {taskDetail.completed_at && (
              <div className="details-row">
                <span className="dt-key">Completed At</span>
                <span className="dt-val">{new Date(taskDetail.completed_at).toLocaleString()}</span>
              </div>
            )}
          </div>
        </div>

        <div className="details-section">
          <h3>Token Usage &amp; Cost Persistence</h3>
          <div className="details-table">
            <div className="details-row">
              <span className="dt-key">Prompt Tokens</span>
              <span className="dt-val font-mono">{taskDetail.prompt_tokens.toLocaleString()}</span>
            </div>
            <div className="details-row">
              <span className="dt-key">Completion Tokens</span>
              <span className="dt-val font-mono">{taskDetail.completion_tokens.toLocaleString()}</span>
            </div>
            <div className="details-row">
              <span className="dt-key">Total Tokens</span>
              <span className="dt-val font-mono font-bold">{taskDetail.total_tokens.toLocaleString()}</span>
            </div>
            <div className="details-row">
              <span className="dt-key">Total Cost (USD)</span>
              <span className="dt-val font-mono font-bold text-success">
                ${taskDetail.total_cost_usd.toFixed(6)}
              </span>
            </div>
            <div className="details-row">
              <span className="dt-key">Duration (seconds)</span>
              <span className="dt-val font-mono">{taskDetail.duration_seconds.toFixed(3)}s</span>
            </div>
          </div>
        </div>

        {taskDetail.result && (
          <div className="details-section full-width">
            <h3>Final Result</h3>
            <pre className="code-block result-block">{taskDetail.result}</pre>
          </div>
        )}

        {taskDetail.error && (
          <div className="details-section full-width">
            <h3>Error Details</h3>
            <pre className="code-block error-details">{taskDetail.error}</pre>
          </div>
        )}

        <div className="details-section full-width">
          <h3>Metadata (JSON)</h3>
          <pre className="code-block json-block">
            {JSON.stringify(taskDetail.metadata, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  )
}
