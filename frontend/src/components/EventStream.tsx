import React, { useEffect, useRef, useState } from 'react'
import type { AgentEvent } from '../types/events'

interface EventStreamProps {
  events: AgentEvent[]
  taskId: string | null
}

export const EventStream: React.FC<EventStreamProps> = ({ events, taskId }) => {
  const bottomRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [events, autoScroll])

  const renderEventCard = (event: AgentEvent, index: number) => {
    const timestamp = new Date(event.timestamp).toLocaleTimeString()

    switch (event.event_type) {
      case 'thought':
        return (
          <div key={`event-${index}`} className="event-card event-thought">
            <div className="event-header">
              <div className="event-type-badge">
                <span className="event-icon">💭</span>
                <span className="event-name">Thought</span>
                <span className="step-tag">Step {event.step}</span>
              </div>
              <span className="event-time">{timestamp}</span>
            </div>
            <div className="event-body thought-text">{event.thought}</div>
          </div>
        )

      case 'tool_call':
        return (
          <div key={`event-${index}`} className="event-card event-tool-call">
            <div className="event-header">
              <div className="event-type-badge">
                <span className="event-icon">⚡</span>
                <span className="event-name">Tool Call:</span>
                <span className="tool-tag">{event.tool_name}</span>
                <span className="step-tag">Step {event.step}</span>
              </div>
              <span className="event-time">{timestamp}</span>
            </div>
            <div className="event-body">
              <pre className="code-block json-block">
                {JSON.stringify(event.arguments, null, 2)}
              </pre>
            </div>
          </div>
        )

      case 'tool_output':
        return (
          <div key={`event-${index}`} className="event-card event-tool-output">
            <div className="event-header">
              <div className="event-type-badge">
                <span className="event-icon">📋</span>
                <span className="event-name">Tool Output:</span>
                <span className="tool-tag">{event.tool_name}</span>
                <span
                  className={`status-pill ${
                    event.success ? 'status-pill-success' : 'status-pill-error'
                  }`}
                >
                  {event.success ? 'Success' : 'Failed'}
                </span>
                <span className="duration-tag">
                  {event.duration_seconds.toFixed(2)}s
                </span>
              </div>
              <span className="event-time">{timestamp}</span>
            </div>
            <div className="event-body">
              <pre className="code-block output-block">{event.output}</pre>
            </div>
          </div>
        )

      case 'task_complete':
        return (
          <div key={`event-${index}`} className="event-card event-task-complete">
            <div className="event-header">
              <div className="event-type-badge">
                <span className="event-icon">✅</span>
                <span className="event-name">Task Completed</span>
              </div>
              <span className="event-time">{timestamp}</span>
            </div>
            <div className="event-body">
              <div className="final-answer-label">Final Answer:</div>
              <div className="final-answer-text">{event.final_answer}</div>
              <div className="complete-stats">
                <span className="stat-pill">
                  <strong>Steps:</strong> {event.total_steps}
                </span>
                <span className="stat-pill">
                  <strong>Tokens:</strong> {event.total_tokens.toLocaleString()}
                </span>
                <span className="stat-pill highlight-pill">
                  <strong>Cost:</strong> ${event.total_cost_usd.toFixed(5)}
                </span>
                <span className="stat-pill">
                  <strong>Duration:</strong> {event.duration_seconds.toFixed(2)}s
                </span>
              </div>
            </div>
          </div>
        )

      case 'error':
        return (
          <div key={`event-${index}`} className="event-card event-error">
            <div className="event-header">
              <div className="event-type-badge">
                <span className="event-icon">⚠️</span>
                <span className="event-name">Execution Error</span>
                {event.step != null && (
                  <span className="step-tag">Step {event.step}</span>
                )}
              </div>
              <span className="event-time">{timestamp}</span>
            </div>
            <div className="event-body">
              <div className="error-text">{event.error}</div>
              {event.details && (
                <pre className="code-block error-details">
                  {JSON.stringify(event.details, null, 2)}
                </pre>
              )}
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="event-stream-container">
      <div className="pane-header">
        <div className="pane-title">
          <span className="pane-icon">🌊</span>
          <span>Reasoning Event Stream</span>
          <span className="event-count-badge">{events.length} events</span>
        </div>
        <div className="pane-controls">
          <button
            type="button"
            className={`control-btn ${autoScroll ? 'active' : ''}`}
            onClick={() => setAutoScroll((prev) => !prev)}
            title="Toggle Auto-Scroll"
          >
            ⬇ Auto-Scroll: {autoScroll ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>

      <div className="events-list">
        {events.length === 0 ? (
          <div className="events-empty">
            {taskId ? (
              <div className="waiting-pulse">
                <span className="pulse-dot" />
                <span>Waiting for agent reasoning events...</span>
              </div>
            ) : (
              <div className="empty-guide">
                <p>Submit a task to observe autonomous reasoning and tool execution in real time.</p>
              </div>
            )}
          </div>
        ) : (
          events.map(renderEventCard)
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
