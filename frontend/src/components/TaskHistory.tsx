import React, { useEffect, useState, useCallback, useMemo } from 'react'
import type { TaskDetailResponse } from '../types/events'
import { StatusBadge } from './StatusBadge'

interface TaskHistoryProps {
  activeTaskId: string | null
  onSelectTask: (taskId: string) => void
  onClearActiveTask?: () => void
  refreshTrigger: number
}

function formatRelativeTime(dateStr: string): string {
  const diffMs = Date.now() - new Date(dateStr).getTime()
  const diffSec = Math.floor(diffMs / 1000)
  if (diffSec < 60) return `${Math.max(1, diffSec)}s ago`
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHour = Math.floor(diffMin / 60)
  if (diffHour < 24) return `${diffHour}h ago`
  return `${Math.floor(diffHour / 24)}d ago`
}

export const TaskHistory: React.FC<TaskHistoryProps> = ({
  activeTaskId,
  onSelectTask,
  onClearActiveTask,
  refreshTrigger,
}) => {
  const [tasks, setTasks] = useState<TaskDetailResponse[]>([])
  const [loading, setLoading] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [filter, setFilter] = useState<string>('ALL')
  const [search, setSearch] = useState('')

  const fetchTasks = useCallback(async () => {
    setLoading(true)
    try {
      const url = filter === 'ALL' ? '/tasks?limit=50' : `/tasks?status=${filter}&limit=50`
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

  const handleClearAllHistory = async () => {
    if (!window.confirm('Are you sure you want to clear all task history from the database?')) {
      return
    }

    setClearing(true)
    try {
      const res = await fetch('/tasks', {
        method: 'DELETE',
        headers: { 'Accept': 'application/json' },
      })
      if (res.ok) {
        setTasks([])
        if (onClearActiveTask) {
          onClearActiveTask()
        }
      } else {
        const err = await res.json().catch(() => ({}))
        alert(`Failed to clear history: ${err.detail || res.statusText}`)
      }
    } catch (err) {
      console.error('Error clearing history:', err)
      alert(`Error clearing history: ${err}`)
    } finally {
      setClearing(false)
      fetchTasks()
    }
  }

  const handleDeleteTask = async (taskId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      const res = await fetch(`/tasks/${encodeURIComponent(taskId)}`, {
        method: 'DELETE',
      })
      if (res.ok) {
        setTasks((prev) => prev.filter((t) => t.task_id !== taskId))
        if (taskId === activeTaskId && onClearActiveTask) {
          onClearActiveTask()
        }
      }
    } catch (err) {
      console.error(`Failed to delete task ${taskId}:`, err)
    }
  }

  const filteredTasks = useMemo(() => {
    if (!search.trim()) return tasks
    const q = search.toLowerCase()
    return tasks.filter((t) => t.task.toLowerCase().includes(q) || t.task_id.includes(q))
  }, [tasks, search])

  return (
    <div className="task-history-container card">
      <div className="card-header history-header">
        <div className="history-header-title">
          <span className="card-title">Task History</span>
          <span className="history-count-badge">{filteredTasks.length}</span>
        </div>
        <div className="history-controls">
          {tasks.length > 0 && (
            <button
              type="button"
              className="btn btn-ghost btn-sm clear-history-btn"
              onClick={handleClearAllHistory}
              disabled={clearing}
              title="Delete all task history from database"
            >
              {clearing ? 'Clearing...' : 'Clear All'}
            </button>
          )}
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={fetchTasks}
            disabled={loading}
            title="Refresh History"
          >
            {loading ? '…' : 'Refresh'}
          </button>
        </div>

      </div>

      <div className="history-filters-bar">
        <select
          className="filter-select"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="ALL">All Status</option>
          <option value="COMPLETED">Completed</option>
          <option value="RUNNING">Running</option>
          <option value="PENDING">Pending</option>
          <option value="FAILED">Failed</option>
        </select>

        <input
          type="text"
          className="history-search-input"
          placeholder="Filter history..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="history-list">
        {loading && tasks.length === 0 ? (
          <div className="history-loading">
            <span className="spinner" /> Loading past tasks...
          </div>
        ) : filteredTasks.length === 0 ? (
          <div className="history-empty">
            {search ? 'No matching tasks found.' : 'No task operations in history.'}
          </div>
        ) : (
          filteredTasks.map((t) => {
            const isSelected = t.task_id === activeTaskId
            const relativeTime = formatRelativeTime(t.created_at)
            return (
              <div
                key={t.task_id}
                className={`history-item ${isSelected ? 'selected' : ''}`}
                onClick={() => onSelectTask(t.task_id)}
              >
                <div className="history-item-top">
                  <StatusBadge type="task" status={t.status} />
                  <div className="history-item-top-right">
                    <span className="history-time">{relativeTime}</span>
                    <button
                      type="button"
                      className="delete-item-btn"
                      onClick={(e) => handleDeleteTask(t.task_id, e)}
                      title="Delete this task"
                    >
                      ×
                    </button>
                  </div>
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
