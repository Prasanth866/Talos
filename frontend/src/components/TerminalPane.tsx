import React, { useEffect, useRef, useState } from 'react'
import type { RawLogItem } from '../hooks/useTaskWebSocket'

interface TerminalPaneProps {
  logs: RawLogItem[]
  onClear: () => void
}

export const TerminalPane: React.FC<TerminalPaneProps> = ({ logs, onClear }) => {
  const terminalEndRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (autoScroll && terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  const handleCopy = () => {
    const text = logs.map((l) => `[${l.timestamp}] ${l.payload}`).join('\n')
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
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
          <span className="terminal-title-text">WebSocket Event Log & Terminal</span>
          <span className="log-count-badge">{logs.length} entries</span>
        </div>

        <div className="pane-controls">
          <button
            type="button"
            className={`control-btn ${autoScroll ? 'active' : ''}`}
            onClick={() => setAutoScroll((prev) => !prev)}
          >
            ⬇ Auto-Scroll: {autoScroll ? 'ON' : 'OFF'}
          </button>
          <button
            type="button"
            className="control-btn"
            onClick={handleCopy}
            disabled={logs.length === 0}
          >
            {copied ? '✓ Copied' : '📋 Copy All'}
          </button>
          <button
            type="button"
            className="control-btn"
            onClick={onClear}
            disabled={logs.length === 0}
          >
            🗑 Clear
          </button>
        </div>
      </div>

      <div className="terminal-body">
        {logs.length === 0 ? (
          <div className="terminal-empty">
            <span className="terminal-prompt">$</span> Listening on WebSocket stream...
          </div>
        ) : (
          logs.map((log) => (
            <div key={log.id} className="terminal-line">
              <span className="line-prefix">[{log.timestamp}]</span>
              <span className="line-content">{log.payload}</span>
            </div>
          ))
        )}
        <div ref={terminalEndRef} />
      </div>
    </div>
  )
}
