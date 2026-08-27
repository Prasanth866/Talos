from __future__ import annotations

from src.agent.models import (
    Message,
    MessageRole,
    Plan,
    ToolExecutionRecord,
)


class ContextManager:
    """Manages context window sliding and history compaction for LLM prompts."""

    def __init__(
        self,
        max_recent_records: int = 5,
        max_history_tokens: int = 4000,
        max_compact_chars: int = 150,
    ) -> None:
        self.max_recent_records = max_recent_records
        self.max_history_tokens = max_history_tokens
        self.max_compact_chars = max_compact_chars

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimates token count (~4 characters per token heuristic)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def estimate_tool_history_tokens(self, records: list[ToolExecutionRecord]) -> int:
        """Calculates approximate total tokens across a list of tool records."""
        total_chars = sum(
            len(r.tool_name) + len(str(r.arguments)) + len(r.output) + len(str(r.step))
            for r in records
        )
        return self.estimate_tokens("a" * total_chars)

    def trim_tool_history(
        self,
        tool_history: list[ToolExecutionRecord],
        max_recent: int | None = None,
        max_tokens: int | None = None,
    ) -> list[ToolExecutionRecord]:
        """Applies sliding-window compaction to tool history.

        Preserves the most recent N records in full detail, while compacting
        older records into concise 1-line summaries to stay within the token budget.
        """
        recent_limit = max_recent if max_recent is not None else self.max_recent_records
        token_limit = max_tokens if max_tokens is not None else self.max_history_tokens

        if not tool_history:
            return []

        if (
            len(tool_history) <= recent_limit
            and self.estimate_tool_history_tokens(tool_history) <= token_limit
        ):
            return list(tool_history)

        split_idx = max(0, len(tool_history) - recent_limit)
        older_records = tool_history[:split_idx]
        recent_records = tool_history[split_idx:]

        compacted_older: list[ToolExecutionRecord] = []
        for record in older_records:
            compact_summary = record.to_compact_summary(
                max_output_chars=self.max_compact_chars
            )
            compacted_older.append(
                ToolExecutionRecord(
                    step=record.step,
                    tool_name=record.tool_name,
                    arguments=record.arguments,
                    output=compact_summary,
                    success=record.success,
                    duration_seconds=record.duration_seconds,
                    timestamp=record.timestamp,
                )
            )

        trimmed = compacted_older + list(recent_records)

        current_tokens = self.estimate_tool_history_tokens(trimmed)
        if current_tokens > token_limit and compacted_older:
            reduced_older: list[ToolExecutionRecord] = []
            for record in compacted_older:
                reduced_older.append(
                    ToolExecutionRecord(
                        step=record.step,
                        tool_name=record.tool_name,
                        arguments={},
                        output=f"[Step {record.step}] {record.tool_name} completed.",
                        success=record.success,
                        duration_seconds=record.duration_seconds,
                        timestamp=record.timestamp,
                    )
                )
            trimmed = reduced_older + list(recent_records)

        return trimmed

    def format_plan_section(self, plan: Plan | None, current_step_index: int) -> str:
        """Renders plan steps with status markers for LLM prompt context."""
        if not plan or not plan.steps:
            return "No plan established yet."

        lines = [f"Plan Rationale: {plan.rationale}", "Steps:"]
        for idx, step in enumerate(plan.steps):
            marker = "[ ]"
            if idx < current_step_index or step.status == "completed":
                marker = "[x]"
            elif idx == current_step_index:
                marker = "[->]"
            lines.append(
                f"  {marker} Step {step.step_id}: {step.description} "
                f"(Expected: {step.expected_output})"
            )
        return "\n".join(lines)

    def format_history_section(self, tool_history: list[ToolExecutionRecord]) -> str:
        """Formats trimmed tool history into a clear trajectory block.

        Wraps each tool output in untrusted_observation tags for prompt isolation.
        """
        if not tool_history:
            return "No previous actions taken."

        trimmed = self.trim_tool_history(tool_history)
        lines = ["Execution History:"]
        for r in trimmed:
            status_str = "SUCCESS" if r.success else "FAILED"
            lines.append(f"\n[Step {r.step}] Action: {r.tool_name}({r.arguments})")
            lines.append(
                f"  Result ({status_str}):\n"
                f'<untrusted_observation source="{r.tool_name}" step="{r.step}">\n'
                f"{r.output}\n"
                f"</untrusted_observation>"
            )
        return "\n".join(lines)

    def build_context_messages(
        self,
        system_prompt: str,
        task: str,
        plan: Plan | None = None,
        current_step_index: int = 0,
        tool_history: list[ToolExecutionRecord] | None = None,
        reflection_history: list[str] | None = None,
    ) -> list[Message]:
        """Assembles a token-efficient context message list for the LLM."""
        history_list = tool_history or []
        reflections = reflection_history or []

        plan_text = self.format_plan_section(plan, current_step_index)
        history_text = self.format_history_section(history_list)

        user_content_parts = [
            f"=== TASK GOAL ===\n{task}\n",
            f"=== CURRENT PLAN ===\n{plan_text}\n",
            f"=== TOOL & OBSERVATION HISTORY ===\n{history_text}\n",
        ]

        if reflections:
            reflection_text = "\n".join(f"- {ref}" for ref in reflections[-3:])
            user_content_parts.append(
                f"=== RECENT REFLECTIONS ===\n{reflection_text}\n"
            )

        user_content_parts.append(
            "Determine the next action to take to progress the plan or provide "
            "the final answer if complete."
        )

        return [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content="\n".join(user_content_parts)),
        ]
