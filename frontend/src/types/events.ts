export type EventType =
  | 'thought'
  | 'tool_call'
  | 'tool_output'
  | 'task_complete'
  | 'error'

export interface BaseEvent {
  version: string
  timestamp: string
  task_id?: string | null
  event_type: EventType
}

export interface ThoughtEvent extends BaseEvent {
  event_type: 'thought'
  thought: string
  step: number
}

export interface ToolCallEvent extends BaseEvent {
  event_type: 'tool_call'
  tool_name: string
  tool_call_id: string
  arguments: Record<string, unknown>
  step: number
}

export interface ToolOutputEvent extends BaseEvent {
  event_type: 'tool_output'
  tool_name: string
  tool_call_id: string
  output: string
  success: boolean
  duration_seconds: number
  step: number
}

export interface TaskCompleteEvent extends BaseEvent {
  event_type: 'task_complete'
  task: string
  final_answer: string
  total_steps: number
  total_tokens: number
  total_cost_usd: number
  duration_seconds: number
}

export interface ErrorEvent extends BaseEvent {
  event_type: 'error'
  error: string
  details?: Record<string, unknown> | null
  step?: number | null
}

export type AgentEvent =
  | ThoughtEvent
  | ToolCallEvent
  | ToolOutputEvent
  | TaskCompleteEvent
  | ErrorEvent

export type TaskStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED'

export interface TaskSubmitResponse {
  task_id: string
  status: TaskStatus
  ws_url: string
}

export interface TaskDetailResponse {
  task_id: string
  task: string
  status: TaskStatus
  result?: string | null
  error?: string | null
  metadata: Record<string, unknown>
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  total_cost_usd: number
  duration_seconds: number
  created_at: string
  updated_at: string
  started_at?: string | null
  completed_at?: string | null
}

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting'
