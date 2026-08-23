import { useState, useEffect, useRef, useCallback } from 'react'
import type { AgentEvent, ConnectionStatus } from '../types/events'

interface UseTaskWebSocketOptions {
  taskId: string | null
  onTaskFinished?: () => void
}

export interface RawLogItem {
  id: string
  timestamp: string
  payload: string
}

export function useTaskWebSocket({ taskId, onTaskFinished }: UseTaskWebSocketOptions) {
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [rawLogs, setRawLogs] = useState<RawLogItem[]>([])
  const [status, setStatus] = useState<ConnectionStatus>('disconnected')
  const [error, setError] = useState<string | null>(null)

  const socketRef = useRef<WebSocket | null>(null)
  const reconnectAttemptRef = useRef(0)
  const maxReconnectAttempts = 3
  const isExplicitCloseRef = useRef(false)
  const onTaskFinishedRef = useRef(onTaskFinished)
  onTaskFinishedRef.current = onTaskFinished

  const clearEvents = useCallback(() => {
    setEvents([])
    setRawLogs([])
    setError(null)
  }, [])

  const connect = useCallback(() => {
    if (!taskId) {
      setStatus('disconnected')
      return
    }

    // Determine correct WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const wsUrl = `${protocol}//${host}/ws?task_id=${encodeURIComponent(taskId)}`

    setStatus(reconnectAttemptRef.current > 0 ? 'reconnecting' : 'connecting')
    setError(null)
    isExplicitCloseRef.current = false

    const ws = new WebSocket(wsUrl)
    socketRef.current = ws

    ws.onopen = () => {
      setStatus('connected')
      reconnectAttemptRef.current = 0
    }

    ws.onmessage = (messageEvent) => {
      try {
        const rawText = messageEvent.data as string
        const parsed: AgentEvent = JSON.parse(rawText)

        setEvents((prev) => [...prev, parsed])
        setRawLogs((prev) => [
          ...prev,
          {
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
            timestamp: new Date().toLocaleTimeString(),
            payload: rawText,
          },
        ])

        if (parsed.event_type === 'task_complete' || parsed.event_type === 'error') {
          if (onTaskFinishedRef.current) {
            onTaskFinishedRef.current()
          }
        }
      } catch (err) {
        console.error('Failed to parse WebSocket event payload:', err)
      }
    }

    ws.onerror = (evt) => {
      console.warn('WebSocket encountered an error:', evt)
      setError('WebSocket connection error')
    }

    ws.onclose = (closeEvent) => {
      socketRef.current = null
      if (isExplicitCloseRef.current) {
        setStatus('disconnected')
        return
      }

      // Check if closed normally or due to task finish
      if (closeEvent.code === 1000 || closeEvent.code === 1001) {
        setStatus('disconnected')
        if (onTaskFinishedRef.current) {
          onTaskFinishedRef.current()
        }
        return
      }

      // Attempt reconnection if attempts remaining
      if (reconnectAttemptRef.current < maxReconnectAttempts) {
        const delay = Math.pow(2, reconnectAttemptRef.current) * 1000
        reconnectAttemptRef.current += 1
        setStatus('reconnecting')
        setTimeout(() => {
          if (!isExplicitCloseRef.current && taskId) {
            connect()
          }
        }, delay)
      } else {
        setStatus('disconnected')
        setError('Connection closed. Reconnect attempts exhausted.')
      }
    }
  }, [taskId])

  useEffect(() => {
    if (!taskId) {
      if (socketRef.current) {
        isExplicitCloseRef.current = true
        socketRef.current.close()
        socketRef.current = null
      }
      setStatus('disconnected')
      return
    }

    reconnectAttemptRef.current = 0
    clearEvents()
    connect()

    return () => {
      isExplicitCloseRef.current = true
      if (socketRef.current) {
        socketRef.current.close()
        socketRef.current = null
      }
    }
  }, [taskId, connect, clearEvents])

  return {
    events,
    rawLogs,
    status,
    error,
    clearEvents,
    reconnect: () => {
      reconnectAttemptRef.current = 0
      connect()
    },
  }
}
