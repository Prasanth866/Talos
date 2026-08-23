import React, { useState } from 'react'

interface TaskSubmitFormProps {
  onSubmit: (prompt: string) => Promise<void>
  submitting: boolean
  activeTaskId: string | null
}

const EXAMPLE_PROMPTS = [
  'Inspect repository structure and report all files',
  'Analyze project architecture and key modules',
  'Check system health, readiness, and tool status',
]

export const TaskSubmitForm: React.FC<TaskSubmitFormProps> = ({
  onSubmit,
  submitting,
  activeTaskId,
}) => {
  const [prompt, setPrompt] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!prompt.trim() || submitting) return
    await onSubmit(prompt.trim())
  }

  const handleSelectExample = (text: string) => {
    setPrompt(text)
  }

  return (
    <div className="task-form-container card">
      <div className="card-header">
        <span className="card-title">🚀 Dispatch Agent Task</span>
      </div>

      <form onSubmit={handleSubmit} className="task-form">
        <div className="form-group">
          <label htmlFor="task-prompt" className="form-label">
            Task Prompt:
          </label>
          <textarea
            id="task-prompt"
            className="task-textarea"
            placeholder="Describe the objective for the autonomous agent to solve..."
            rows={4}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={submitting}
          />
        </div>

        <div className="example-prompts">
          <span className="example-label">Quick Templates:</span>
          <div className="example-chips">
            {EXAMPLE_PROMPTS.map((ex, idx) => (
              <button
                key={idx}
                type="button"
                className="example-chip"
                onClick={() => handleSelectExample(ex)}
                disabled={submitting}
              >
                {ex}
              </button>
            ))}
          </div>
        </div>

        <div className="form-actions">
          <button
            type="submit"
            className="btn btn-primary submit-btn"
            disabled={!prompt.trim() || submitting}
          >
            {submitting ? (
              <>
                <span className="spinner" />
                <span>Submitting Task...</span>
              </>
            ) : (
              <span>⚡ Execute Agent Task</span>
            )}
          </button>
        </div>

        {activeTaskId && (
          <div className="active-task-footer">
            <span className="active-task-label">Active Task:</span>
            <span className="active-task-id" title={activeTaskId}>
              {activeTaskId}
            </span>
          </div>
        )}
      </form>
    </div>
  )
}
