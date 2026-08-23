import React, { useState } from 'react'

interface TaskSubmitFormProps {
  onSubmit: (prompt: string, metadata?: Record<string, unknown>) => Promise<void>
  submitting: boolean
  activeTaskId: string | null
  onClearActiveTask: () => void
}

interface Template {
  tag: string
  title: string
  prompt: string
}

const TEMPLATES: Template[] = [
  {
    tag: 'DIR',
    title: 'Inspect Repository',
    prompt: 'List the project repository directory and summarize the project structure.',
  },
  {
    tag: 'CODE',
    title: 'Analyze Codebase',
    prompt: 'Analyze the architecture, dependencies, and core modules of this project.',
  },
  {
    tag: 'SYS',
    title: 'System Health Check',
    prompt: 'Check system tool availability, test environment files, and report status.',
  },
  {
    tag: 'TEST',
    title: 'Test Verification',
    prompt: 'Execute tests, verify assertions, and summarize results.',
  },
]

export const TaskSubmitForm: React.FC<TaskSubmitFormProps> = ({
  onSubmit,
  submitting,
  activeTaskId,
  onClearActiveTask,
}) => {
  const [prompt, setPrompt] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [tagKey, setTagKey] = useState('')
  const [tagValue, setTagValue] = useState('')
  const [tags, setTags] = useState<Record<string, string>>({ env: 'development' })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!prompt.trim() || submitting) return
    await onSubmit(prompt.trim(), tags)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      if (prompt.trim() && !submitting) {
        onSubmit(prompt.trim(), tags)
      }
    }
  }

  const handleAddTag = () => {
    if (!tagKey.trim() || !tagValue.trim()) return
    setTags((prev) => ({ ...prev, [tagKey.trim()]: tagValue.trim() }))
    setTagKey('')
    setTagValue('')
  }

  const handleRemoveTag = (key: string) => {
    setTags((prev) => {
      const next = { ...prev }
      delete next[key]
      return next
    })
  }


  return (
    <div className="task-form-container card">
      <div className="card-header">
        <div className="card-header-left">
          <span className="card-title">Dispatch Agent</span>
        </div>
        {activeTaskId && (
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={onClearActiveTask}
            title="Deselect active task and start fresh"
          >
            + New Task
          </button>
        )}
      </div>

      <form onSubmit={handleSubmit} className="task-form">
        <div className="form-group">
          <div className="form-label-row">
            <label htmlFor="task-prompt" className="form-label">
              Task Objective
            </label>
            <span className="char-count">{prompt.length} chars</span>
          </div>
          <textarea
            id="task-prompt"
            className="task-textarea"
            placeholder="Describe what the agent should accomplish (e.g. Inspect the codebase, run tools, fix issues)..."
            rows={4}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={submitting}
          />
          <div className="input-hint">
            Press <kbd>Cmd</kbd>+<kbd>Enter</kbd> or <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to submit
          </div>
        </div>

        {/* Quick Templates */}
        <div className="templates-section">
          <div className="templates-label">Quick Presets:</div>
          <div className="templates-grid">
            {TEMPLATES.map((tmpl, idx) => (
              <button
                key={idx}
                type="button"
                className="template-card"
                onClick={() => setPrompt(tmpl.prompt)}
                disabled={submitting}
              >
                <span className="template-tag-badge">{tmpl.tag}</span>
                <span className="template-title">{tmpl.title}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Advanced Options Accordion */}
        <div className="advanced-toggle-row">
          <button
            type="button"
            className="advanced-toggle-btn"
            onClick={() => setShowAdvanced((prev) => !prev)}
          >
            <span>{showAdvanced ? '▼' : '▶'} Metadata &amp; Tags</span>
            <span className="tag-count-badge">{Object.keys(tags).length} tags</span>
          </button>
        </div>

        {showAdvanced && (
          <div className="advanced-panel">
            <div className="tags-list">
              {Object.entries(tags).map(([k, v]) => (
                <span key={k} className="tag-pill">
                  <strong>{k}:</strong> {v}
                  <button
                    type="button"
                    className="tag-remove-btn"
                    onClick={() => handleRemoveTag(k)}
                  >
                    ✕
                  </button>
                </span>
              ))}
            </div>
            <div className="tag-input-row">
              <input
                type="text"
                placeholder="Key"
                className="tag-input"
                value={tagKey}
                onChange={(e) => setTagKey(e.target.value)}
              />
              <input
                type="text"
                placeholder="Value"
                className="tag-input"
                value={tagValue}
                onChange={(e) => setTagValue(e.target.value)}
              />
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={handleAddTag}
              >
                Add
              </button>
            </div>
          </div>
        )}

        <button
          type="submit"
          className="btn btn-primary submit-btn"
          disabled={!prompt.trim() || submitting}
        >
          {submitting ? (
            <>
              <span className="spinner" />
              <span>Dispatching Agent...</span>
            </>
          ) : (
            <span>Run Autonomous Agent</span>
          )}
        </button>

        {activeTaskId && (
          <div className="active-task-bar">
            <span className="active-dot" />
            <div className="active-task-meta">
              <span className="active-task-label">Observing Task:</span>
              <span className="active-task-id" title={activeTaskId}>
                {activeTaskId}
              </span>
            </div>
            <button
              type="button"
              className="copy-id-btn"
              onClick={() => navigator.clipboard.writeText(activeTaskId)}
              title="Copy Task UUID"
            >
              Copy
            </button>
          </div>
        )}
      </form>
    </div>
  )
}
