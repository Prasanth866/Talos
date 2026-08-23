import React, { useEffect, useRef, useState, useMemo } from 'react'
import type { AgentEvent } from '../types/events'

interface EventStreamProps {
  events: AgentEvent[]
  taskId: string | null
}

interface ModalContent {
  title: string
  subtitle: string
  content: string
  isJson?: boolean
}

export const EventStream: React.FC<EventStreamProps> = ({ events, taskId }) => {
  const bottomRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [filterType, setFilterType] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [collapsedCards, setCollapsedCards] = useState<Record<number, boolean>>({})
  const [expandedFullHeight, setExpandedFullHeight] = useState<Record<number, boolean>>({})
  const [modalData, setModalData] = useState<ModalContent | null>(null)

  // Auto-scroll when new events arrive
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [events, autoScroll])

  const copyText = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 1500)
  }

  const toggleCard = (index: number) => {
    setCollapsedCards((prev) => ({ ...prev, [index]: !prev[index] }))
  }

  const toggleFullHeight = (index: number, e: React.MouseEvent) => {
    e.stopPropagation()
    setExpandedFullHeight((prev) => ({ ...prev, [index]: !prev[index] }))
  }

  // Filter and search
  const filteredEvents = useMemo(() => {
    return events.filter((evt) => {
      if (filterType !== 'all' && evt.event_type !== filterType) {
        return false
      }
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase()
        const textToSearch = JSON.stringify(evt).toLowerCase()
        return textToSearch.includes(query)
      }
      return true
    })
  }, [events, filterType, searchQuery])

  // Count by type
  const counts = useMemo(() => {
    const res = { all: events.length, thought: 0, tool_call: 0, tool_output: 0, task_complete: 0, error: 0 }
    for (const e of events) {
      if (e.event_type in res) {
        res[e.event_type as keyof typeof res] += 1
      }
    }
    return res
  }, [events])

  const renderEventCard = (event: AgentEvent, index: number) => {
    const timestamp = new Date(event.timestamp).toLocaleTimeString()
    const isCollapsed = collapsedCards[index] || false
    const isFullHeight = expandedFullHeight[index] || false
    const copyKey = `evt-${index}`

    switch (event.event_type) {
      case 'thought':
        return (
          <div key={`event-${index}`} className="event-card event-thought">
            <div className="event-header" onClick={() => toggleCard(index)}>
              <div className="event-type-badge">
                <span className="event-name">Reasoning Thought</span>
                <span className="step-tag">Step {event.step}</span>
              </div>
              <div className="event-header-actions" onClick={(e) => e.stopPropagation()}>
                <span className="event-time">{timestamp}</span>
                <button
                  type="button"
                  className="card-action-btn"
                  onClick={() =>
                    setModalData({
                      title: `Thought (Step ${event.step})`,
                      subtitle: timestamp,
                      content: event.thought,
                    })
                  }
                  title="View full text in modal"
                >
                  View Full
                </button>
                <button
                  type="button"
                  className="card-action-btn"
                  onClick={() => copyText(event.thought, copyKey)}
                  title="Copy thought text"
                >
                  {copiedId === copyKey ? 'Copied' : 'Copy'}
                </button>
              </div>
            </div>
            {!isCollapsed && (
              <div className={`event-body thought-text ${isFullHeight ? 'full-height' : ''}`}>
                {event.thought}
              </div>
            )}
          </div>
        )

      case 'tool_call': {
        const jsonStr = JSON.stringify(event.arguments, null, 2)
        return (
          <div key={`event-${index}`} className="event-card event-tool-call">
            <div className="event-header" onClick={() => toggleCard(index)}>
              <div className="event-type-badge">
                <span className="event-name">Action:</span>
                <span className="tool-tag">{event.tool_name}</span>
                <span className="step-tag">Step {event.step}</span>
              </div>
              <div className="event-header-actions" onClick={(e) => e.stopPropagation()}>
                <span className="event-time">{timestamp}</span>
                <button
                  type="button"
                  className="card-action-btn"
                  onClick={() =>
                    setModalData({
                      title: `Tool Arguments: ${event.tool_name} (Step ${event.step})`,
                      subtitle: timestamp,
                      content: jsonStr,
                      isJson: true,
                    })
                  }
                  title="View full arguments in modal"
                >
                  View Full
                </button>
                <button
                  type="button"
                  className="card-action-btn"
                  onClick={(e) => toggleFullHeight(index, e)}
                  title="Toggle full height"
                >
                  {isFullHeight ? 'Collapse Height' : 'Expand Height'}
                </button>
                <button
                  type="button"
                  className="card-action-btn"
                  onClick={() => copyText(jsonStr, copyKey)}
                  title="Copy arguments JSON"
                >
                  {copiedId === copyKey ? 'Copied' : 'Copy'}
                </button>
              </div>
            </div>
            {!isCollapsed && (
              <div className="event-body">
                <div className="code-label">Arguments:</div>
                <pre className={`code-block json-block ${isFullHeight ? 'full-height' : ''}`}>
                  {jsonStr}
                </pre>
              </div>
            )}
          </div>
        )
      }

      case 'tool_output':
        return (
          <div key={`event-${index}`} className="event-card event-tool-output">
            <div className="event-header" onClick={() => toggleCard(index)}>
              <div className="event-type-badge">
                <span className="event-name">Observation:</span>
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
              <div className="event-header-actions" onClick={(e) => e.stopPropagation()}>
                <span className="event-time">{timestamp}</span>
                <button
                  type="button"
                  className="card-action-btn"
                  onClick={() =>
                    setModalData({
                      title: `Tool Output: ${event.tool_name} (Step ${event.step})`,
                      subtitle: `${event.duration_seconds.toFixed(2)}s • ${event.success ? 'Success' : 'Failed'}`,
                      content: event.output,
                    })
                  }
                  title="View full output in modal"
                >
                  View Full
                </button>
                <button
                  type="button"
                  className="card-action-btn"
                  onClick={(e) => toggleFullHeight(index, e)}
                  title="Toggle full height"
                >
                  {isFullHeight ? 'Collapse Height' : 'Expand Height'}
                </button>
                <button
                  type="button"
                  className="card-action-btn"
                  onClick={() => copyText(event.output, copyKey)}
                  title="Copy output text"
                >
                  {copiedId === copyKey ? 'Copied' : 'Copy'}
                </button>
              </div>
            </div>
            {!isCollapsed && (
              <div className="event-body">
                <pre className={`code-block output-block ${isFullHeight ? 'full-height' : ''}`}>
                  {event.output}
                </pre>
              </div>
            )}
          </div>
        )

      case 'task_complete':
        return (
          <div key={`event-${index}`} className="event-card event-task-complete">
            <div className="event-header">
              <div className="event-type-badge">
                <span className="event-name">Task Successfully Completed</span>
              </div>
              <div className="event-header-actions">
                <span className="event-time">{timestamp}</span>
                <button
                  type="button"
                  className="card-action-btn"
                  onClick={() =>
                    setModalData({
                      title: 'Final Answer & Resolution Summary',
                      subtitle: `${event.total_steps} steps • ${event.total_tokens.toLocaleString()} tokens • $${event.total_cost_usd.toFixed(5)}`,
                      content: event.final_answer,
                    })
                  }
                  title="View full answer in modal"
                >
                  View Full
                </button>
                <button
                  type="button"
                  className="card-action-btn"
                  onClick={() => copyText(event.final_answer, copyKey)}
                  title="Copy final answer"
                >
                  {copiedId === copyKey ? 'Copied' : 'Copy'}
                </button>
              </div>
            </div>
            <div className="event-body">
              <div className="final-answer-label">Final Answer:</div>
              <div className="final-answer-text">{event.final_answer}</div>
              <div className="complete-stats">
                <span className="stat-pill">
                  <strong>Steps:</strong> {event.total_steps}
                </span>
                <span className="stat-pill">
                  <strong>Total Tokens:</strong> {event.total_tokens.toLocaleString()}
                </span>
                <span className="stat-pill highlight-pill">
                  <strong>Estimated Cost:</strong> ${event.total_cost_usd.toFixed(5)}
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
                <span className="event-name">Execution Failure</span>
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
      {/* Pane Header */}
      <div className="pane-header">
        <div className="pane-title">
          <span>Reasoning Stream</span>
          <span className="event-count-badge">{filteredEvents.length} events</span>
        </div>

        <div className="pane-controls">
          <button
            type="button"
            className={`control-btn ${autoScroll ? 'active' : ''}`}
            onClick={() => setAutoScroll((prev) => !prev)}
            title="Toggle automatic scrolling as new events arrive"
          >
            {autoScroll ? 'Auto-Scroll: ON' : 'Auto-Scroll: OFF'}
          </button>
        </div>
      </div>

      {/* Filter and Search Toolbar */}
      <div className="stream-toolbar">
        <div className="filter-chips">
          <button
            type="button"
            className={`filter-chip ${filterType === 'all' ? 'active' : ''}`}
            onClick={() => setFilterType('all')}
          >
            All ({counts.all})
          </button>
          <button
            type="button"
            className={`filter-chip ${filterType === 'thought' ? 'active' : ''}`}
            onClick={() => setFilterType('thought')}
          >
            Thoughts ({counts.thought})
          </button>
          <button
            type="button"
            className={`filter-chip ${filterType === 'tool_call' ? 'active' : ''}`}
            onClick={() => setFilterType('tool_call')}
          >
            Tools ({counts.tool_call})
          </button>
          <button
            type="button"
            className={`filter-chip ${filterType === 'tool_output' ? 'active' : ''}`}
            onClick={() => setFilterType('tool_output')}
          >
            Outputs ({counts.tool_output})
          </button>
          {counts.error > 0 && (
            <button
              type="button"
              className={`filter-chip filter-chip-error ${filterType === 'error' ? 'active' : ''}`}
              onClick={() => setFilterType('error')}
            >
              Errors ({counts.error})
            </button>
          )}
        </div>

        <div className="search-input-wrapper">
          <input
            type="text"
            className="stream-search-input"
            placeholder="Search in events..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button
              type="button"
              className="clear-search-btn"
              onClick={() => setSearchQuery('')}
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Events List */}
      <div className="events-list">
        {filteredEvents.length === 0 ? (
          <div className="events-empty">
            {taskId ? (
              <div className="waiting-pulse">
                <span className="pulse-dot" />
                <div className="waiting-text-group">
                  <strong>Streaming agent thoughts and tool actions...</strong>
                  <span>Events will appear here in real time as the agent reasons.</span>
                </div>
              </div>
            ) : (
              <div className="empty-guide">
                <h3>Talos Agent Observer</h3>
                <p>Submit a task from the left panel to watch step-by-step reasoning, tool dispatching, token tracking, and execution metrics.</p>
              </div>
            )}
          </div>
        ) : (
          filteredEvents.map(renderEventCard)
        )}
        <div ref={bottomRef} />
      </div>

      {/* Full Content Inspection Modal */}
      {modalData && (
        <div className="modal-backdrop" onClick={() => setModalData(null)}>
          <div className="modal-container card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3 className="modal-title">{modalData.title}</h3>
                <span className="modal-subtitle">{modalData.subtitle}</span>
              </div>
              <div className="modal-actions">
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => copyText(modalData.content, 'modal-copy')}
                >
                  {copiedId === 'modal-copy' ? 'Copied' : 'Copy All'}
                </button>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm modal-close-btn"
                  onClick={() => setModalData(null)}
                >
                  Close
                </button>
              </div>
            </div>
            <div className="modal-body">
              <pre className="modal-code-block">{modalData.content}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
