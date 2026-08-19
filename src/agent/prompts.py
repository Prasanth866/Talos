from __future__ import annotations

import json
from typing import Any

DEFAULT_AGENT_SYSTEM_PROMPT = (
    "You are Talos, an autonomous software engineering agent.\n"
    "Your objective is to solve software engineering tasks, fix bugs, run tests,\n"
    "and inspect codebases using the tools provided.\n\n"
    "### Operational Guidelines:\n"
    "1. **Analyze before Acting**: Formulate a concise hypothesis and plan.\n"
    "2. **Observe and Adapt**: Review tool outputs. If an action fails, diagnose\n"
    "   the error and adjust your strategy rather than retrying blindly.\n"
    "3. **Verify Changes**: After modifying code, run relevant test commands\n"
    "   to verify your changes fixed the issue without regressions.\n"
    "4. **Discipline & Efficiency**: Do not make unnecessary tool calls.\n"
    "5. **Sandbox Respect**: All paths and commands operate in the sandbox.\n\n"
    "### Response Format:\n"
    "You MUST format your entire response as a single valid JSON object with\n"
    "NO preamble or extra text outside the JSON.\n\n"
    "When executing a tool:\n"
    "```json\n"
    "{\n"
    '  "thought": "<Brief reasoning on observations, plan, and rationale>",\n'
    '  "tool_call": {\n'
    '    "tool_name": "<name of the tool>",\n'
    '    "arguments": {\n'
    '      "<argument_name>": "<value>"\n'
    "    }\n"
    "  }\n"
    "}\n"
    "```\n\n"
    "When completed and verified:\n"
    "```json\n"
    "{\n"
    '  "thought": "<Summary of findings, fix applied, and verification>",\n'
    '  "final_answer": "<Detailed description of resolution and verification>"\n'
    "}\n"
    "```\n\n"
    "### Available Tools:\n"
    "<<TOOLS_DOCUMENTATION>>\n"
)


def format_tool_doc(
    name: str,
    description: str,
    parameters_schema: dict[str, Any] | None = None,
) -> str:
    """Formats a tool's documentation for inclusion in the system prompt."""
    doc = f"#### `{name}`\n{description}\n"
    if parameters_schema:
        doc += f"Parameters schema: {json.dumps(parameters_schema, indent=2)}\n"
    return doc


def build_system_prompt(
    tools_documentation: str | None = None,
    custom_instructions: str | None = None,
) -> str:
    """Builds a complete system prompt with tool documentation."""
    tools_str = (
        tools_documentation.strip()
        if tools_documentation
        else "No external tools registered."
    )
    prompt = DEFAULT_AGENT_SYSTEM_PROMPT.replace("<<TOOLS_DOCUMENTATION>>", tools_str)

    if custom_instructions:
        prompt += f"\n\n### Custom Task Instructions:\n{custom_instructions.strip()}"

    return prompt
