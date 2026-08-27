"""Agent package.

Reasoning loop, tool dispatch, retry, LangGraph state machine, and token tracking.
"""

from src.agent.context import ContextManager
from src.agent.dispatcher import (
    ToolDefinition,
    ToolDispatcher,
    create_default_dispatcher,
)
from src.agent.graph import LangGraphAgent
from src.agent.llm_client import (
    BaseLLMClient,
    HTTPLLMClient,
    MockLLMClient,
    extract_json_payload,
    parse_llm_response_content,
)
from src.agent.loop import ReasoningLoop
from src.agent.models import (
    AgentStatus,
    AgentStep,
    CostRates,
    LLMResponse,
    Message,
    MessageRole,
    Plan,
    PlanStep,
    ReasoningTrajectory,
    TokenUsage,
    ToolCall,
    ToolExecutionRecord,
    ToolResult,
    TrajectoryStatus,
)
from src.agent.pipeline import (
    create_workspace_dispatcher,
    execute_workspace_task,
)
from src.agent.prompts import (
    DEFAULT_AGENT_SYSTEM_PROMPT,
    build_system_prompt,
    format_tool_doc,
)
from src.agent.retry import (
    NonRetryableError,
    compute_backoff_delay,
    is_transient_error,
    retry_async,
    retry_sync,
)
from src.agent.state import AgentState, create_initial_agent_state
from src.agent.token_tracker import MODEL_PRICING, TokenTracker

__all__ = [
    "DEFAULT_AGENT_SYSTEM_PROMPT",
    "MODEL_PRICING",
    "AgentState",
    "AgentStatus",
    "AgentStep",
    "BaseLLMClient",
    "ContextManager",
    "CostRates",
    "HTTPLLMClient",
    "LLMResponse",
    "LangGraphAgent",
    "Message",
    "MessageRole",
    "MockLLMClient",
    "NonRetryableError",
    "Plan",
    "PlanStep",
    "ReasoningLoop",
    "ReasoningTrajectory",
    "TokenTracker",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "ToolDispatcher",
    "ToolExecutionRecord",
    "ToolResult",
    "TrajectoryStatus",
    "build_system_prompt",
    "compute_backoff_delay",
    "create_default_dispatcher",
    "create_initial_agent_state",
    "create_workspace_dispatcher",
    "execute_workspace_task",
    "extract_json_payload",
    "format_tool_doc",
    "is_transient_error",
    "parse_llm_response_content",
    "retry_async",
    "retry_sync",
]
