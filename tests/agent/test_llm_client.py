from unittest.mock import patch

import httpx
import pytest

from src.agent.llm_client import (
    HTTPLLMClient,
    extract_json_payload,
    parse_llm_response_content,
)
from src.agent.models import Message, MessageRole, TokenUsage


def test_extract_json_payload_direct_and_markdown() -> None:

    direct = '{"thought": "test", "final_answer": "ok"}'
    assert extract_json_payload(direct) == {"thought": "test", "final_answer": "ok"}

    markdown = (
        "```json\n"
        '{"thought": "test", "tool_call": {"tool_name": "foo", "arguments": {}}}\n'
        "```"
    )
    assert extract_json_payload(markdown)["thought"] == "test"

    embedded = (
        "Here is the action:\n"
        '{"thought": "running", "final_answer": "done"}\n'
        "Hope this helps!"
    )
    assert extract_json_payload(embedded)["thought"] == "running"

    with pytest.raises(ValueError, match="Failed to parse valid JSON"):
        extract_json_payload("No json here whatsoever")


def test_parse_llm_response_content() -> None:
    token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    raw = (
        '{"thought": "searching", "tool_call": {"tool_name": "list_dir", '
        '"arguments": {"path": "."}}}'
    )
    resp = parse_llm_response_content(raw, token_usage, latency=0.12)
    assert resp.thought == "searching"
    assert resp.tool_call is not None
    assert resp.tool_call.tool_name == "list_dir"
    assert resp.tool_call.arguments == {"path": "."}
    assert resp.final_answer is None
    assert resp.token_usage.total_tokens == 15
    assert resp.latency_seconds == 0.12

    raw_final = '{"thought": "all done", "final_answer": "Success!"}'
    resp_final = parse_llm_response_content(raw_final, token_usage)
    assert resp_final.thought == "all done"
    assert resp_final.final_answer == "Success!"
    assert resp_final.tool_call is None

    raw_invalid = "Just casual text without JSON format"
    resp_invalid = parse_llm_response_content(raw_invalid, token_usage)
    assert resp_invalid.final_answer is None
    assert resp_invalid.raw_content == raw_invalid
    assert "Failed to parse structured JSON" in resp_invalid.thought


@pytest.mark.asyncio
async def test_http_llm_client_generate_response() -> None:
    client = HTTPLLMClient(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
    )

    mock_response_json = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"thought": "checking files", "tool_call": '
                        '{"tool_name": "read_file", "arguments": {"path": "main.py"}}}'
                    ),
                }
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 45,
            "total_tokens": 195,
        },
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = httpx.Response(
            status_code=200,
            json=mock_response_json,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content="System prompt"),
            Message(role=MessageRole.USER, content="User task"),
        ]

        response = await client.generate_response(messages)

        assert response.thought == "checking files"
        assert response.tool_call is not None
        assert response.tool_call.tool_name == "read_file"
        assert response.token_usage.prompt_tokens == 150
        assert response.token_usage.completion_tokens == 45
        assert client.token_tracker.cumulative_usage.total_tokens == 195
