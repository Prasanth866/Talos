import React from 'react'
import type { ConnectionStatus, TaskStatus } from '../types/events'

interface StatusBadgeProps {
  type: 'connection' | 'task'
  status: ConnectionStatus | TaskStatus
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ type, status }) => {
  const getBadgeClass = () => {
    if (type === 'connection') {
      switch (status) {
        case 'connected':
          return 'badge-success'
        case 'connecting':
        case 'reconnecting':
          return 'badge-warning'
        case 'disconnected':
        default:
          return 'badge-neutral'
      }
    } else {
      switch (status) {
        case 'COMPLETED':
          return 'badge-success'
        case 'RUNNING':
          return 'badge-info'
        case 'PENDING':
          return 'badge-warning'
        case 'FAILED':
          return 'badge-error'
        default:
          return 'badge-neutral'
      }
    }
  }

  const getLabel = () => {
    if (type === 'connection') {
      switch (status) {
        case 'connected':
          return 'Connected'
        case 'connecting':
          return 'Connecting...'
        case 'reconnecting':
          return 'Reconnecting...'
        case 'disconnected':
        default:
          return 'Disconnected'
      }
    }
    return status
  }

  return (
    <span className={`status-badge ${getBadgeClass()}`}>
      <span className="badge-dot" />
      {getLabel()}
    </span>
  )
}
