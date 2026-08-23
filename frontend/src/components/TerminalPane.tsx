import React, { useEffect, useRef, useState, useMemo } from 'react'
import type { RawLogItem } from '../hooks/useTaskWebSocket'

interface TerminalPaneProps {
  logs: RawLogItem[]
  onClear: () => void
}

export const TerminalPane: React.FC<TerminalPaneProps> = ({ logs, onClear }) => {
  const terminalEndRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [copied, setCopied] = useState(false)
  const [search, setSearch] = useState('')
  const [wrapText, setWrapText] = useState(true)

  useEffect(() => {
    if (autoScroll && terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  const filteredLogs = useMemo(() => {
    if (!search.trim()) return logs
    const q = search.toLowerCase()
    return logs.filter((l) => l.payload.toLowerCase().includes(q))
  }, [logs, search])

  const handleCopy = () => {
    const text = filteredLogs.map((l) => `[${l.timestamp}] ${l.payload}`).join('\n')
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const formatPayload = (raw: string) => {
    try {
      const parsed = JSON.parse(raw)
      const eventType = parsed.event_type || 'event'
      return (
        <span>
          <span className={`terminal-event-tag tag-${eventType}`}>[{eventType}]</span>{' '}
          <span className="terminal-json">{JSON.stringify(parsed)}</span>
        </span>
      )
    } catch {
      return <span>{raw}</span>
    }
  }

  return (
    <div className="terminal-container">
      <div className="pane-header terminal-header">
        <div className="pane-title">
          <span className="terminal-dots">
            <span className="dot dot-red" />
            <span className="dot dot-yellow" />
            <span className="dot dot-green" />
          </span>
          <span className="terminal-title-text">WebSocket Raw Terminal</span>
          <span className="log-count-badge">{filteredLogs.length} entries</span>
        </div>

        <div className="terminal-toolbar-center">
          <input
            type="text"
            className="terminal-search-input"
            placeholder="Search terminal logs..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div className="pane-controls">
          <button
            type="button"
            className={`control-btn ${wrapText ? 'active' : ''}`}
            onClick={() => setWrapText((prev) => !prev)}
            title="Toggle line wrapping"
          >
            Wrap: {wrapText ? 'ON' : 'OFF'}
          </button>
          <button
            type="button"
            className={`control-btn ${autoScroll ? 'active' : ''}`}
            onClick={() => setAutoScroll((prev) => !prev)}
          >
            {autoScroll ? 'Auto: ON' : 'Auto: OFF'}
          </button>
          <button
            type="button"
            className="control-btn"
            onClick={handleCopy}
            disabled={logs.length === 0}
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
          <button
            type="button"
            className="control-btn"
            onClick={onClear}
            disabled={logs.length === 0}
          >
            Clear
          </button>

        </div>
      </div>

      <div className={`terminal-body ${wrapText ? 'wrap-enabled' : 'wrap-disabled'}`}>
        {filteredLogs.length === 0 ? (
          <div className="terminal-empty">
            <span className="terminal-prompt">$</span> Listening on WebSocket stream...
          </div>
        ) : (
          filteredLogs.map((log) => (
            <div key={log.id} className="terminal-line">
              <span className="line-prefix">[{log.timestamp}]</span>
              <span className="line-content">{formatPayload(log.payload)}</span>
            </div>
          ))
        )}
        <div ref={terminalEndRef} />
      </div>
    </div>
  )
}
