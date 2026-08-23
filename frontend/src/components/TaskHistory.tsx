import React, { useEffect, useState, useCallback } from 'react'
import type { TaskDetailResponse } from '../types/events'
import { StatusBadge } from './StatusBadge'

interface TaskHistoryProps {
  activeTaskId: string | null
  onSelectTask: (taskId: string) => void
  refreshTrigger: number
}

export const TaskHistory: React.FC<TaskHistoryProps> = ({
  activeTaskId,
  onSelectTask,
  refreshTrigger,
}) => {
  const [tasks, setTasks] = useState<TaskDetailResponse[]>([])
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState<string>('ALL')

  const fetchTasks = useCallback(async () => {
    setLoading(true)
    try {
      const url = filter === 'ALL' ? '/tasks?limit=20' : `/tasks?status=${filter}&limit=20`
      const res = await fetch(url)
      if (res.ok) {
        const data = (await res.json()) as TaskDetailResponse[]
        setTasks(data)
      }
    } catch (err) {
      console.error('Failed to fetch task history:', err)
    } finally {
      setLoading(false)
    }
  }, [filter])

  useEffect(() => {
    fetchTasks()
  }, [fetchTasks, refreshTrigger])

  return (
    <div className="task-history-container card">
      <div className="card-header history-header">
        <span className="card-title">📜 Task History</span>
        <div className="history-controls">
          <select
            className="filter-select"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="ALL">All Statuses</option>
            <option value="COMPLETED">Completed</option>
            <option value="RUNNING">Running</option>
            <option value="PENDING">Pending</option>
            <option value="FAILED">Failed</option>
          </select>
          <button
            type="button"
            className="btn btn-secondary btn-icon"
            onClick={fetchTasks}
            disabled={loading}
            title="Refresh Tasks"
          >
            🔄
          </button>
        </div>
      </div>

      <div className="history-list">
        {loading && tasks.length === 0 ? (
          <div className="history-loading">Loading tasks...</div>
        ) : tasks.length === 0 ? (
          <div className="history-empty">No tasks found.</div>
        ) : (
          tasks.map((t) => {
            const isSelected = t.task_id === activeTaskId
            const time = new Date(t.created_at).toLocaleTimeString()
            return (
              <div
                key={t.task_id}
                className={`history-item ${isSelected ? 'selected' : ''}`}
                onClick={() => onSelectTask(t.task_id)}
              >
                <div className="history-item-top">
                  <StatusBadge type="task" status={t.status} />
                  <span className="history-time">{time}</span>
                </div>
                <div className="history-prompt" title={t.task}>
                  {t.task}
                </div>
                <div className="history-item-footer">
                  <span className="history-cost">
                    {t.total_cost_usd > 0 ? `$${t.total_cost_usd.toFixed(5)}` : '—'}
                  </span>
                  <span className="history-tokens">
                    {t.total_tokens > 0 ? `${t.total_tokens.toLocaleString()} tokens` : ''}
                  </span>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
