from src.agent.prompts import build_system_prompt, format_tool_doc


def test_format_tool_doc() -> None:
    doc = format_tool_doc(
        name="read_file",
        description="Reads a file safely.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    )
    assert "#### `read_file`" in doc
    assert "Reads a file safely." in doc
    assert '"path"' in doc


def test_build_system_prompt() -> None:
    tools_doc = "#### `tool1`\nTool 1 description."
    custom_inst = "Always explain your thoughts before calling tools."
    prompt = build_system_prompt(
        tools_documentation=tools_doc, custom_instructions=custom_inst
    )

    assert "You are Talos, an autonomous software engineering agent." in prompt
    assert "Tool 1 description." in prompt
    assert "Custom Task Instructions:" in prompt
    assert "Always explain your thoughts before calling tools." in prompt
