import pytest

from src.agent.dispatcher import ToolDispatcher
from src.agent.llm_client import MockLLMClient
from src.agent.loop import ReasoningLoop
from src.agent.models import TrajectoryStatus


@pytest.mark.asyncio
async def test_tool_dispatch_with_mocked_llm_returns_correct_tool_call() -> None:
    dispatcher = ToolDispatcher()
    executed_args = []

    def mock_calc(x: int) -> int:
        executed_args.append(x)
        return x * 2

    dispatcher.register_tool(
        name="calc",
        description="Multiplies by 2",
        handler=mock_calc,
        parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
    )

    mock_llm = MockLLMClient(
        responses=[
            {
                "thought": "I need to calculate double of 21.",
                "tool_call": {"tool_name": "calc", "arguments": {"x": 21}},
            },
            {
                "thought": "The result is 42. Task complete.",
                "final_answer": "42",
            },
        ]
    )

    runner = ReasoningLoop(llm_client=mock_llm, dispatcher=dispatcher, max_steps=5)
    trajectory = await runner.run("Calculate double of 21")

    assert trajectory.status == TrajectoryStatus.COMPLETED
    assert trajectory.final_answer == "42"
    assert len(trajectory.steps) == 2
    assert executed_args == [21]

    step1 = trajectory.steps[0]
    assert step1.step_number == 1
    assert step1.thought == "I need to calculate double of 21."
    assert step1.tool_call is not None
    assert step1.tool_call.tool_name == "calc"
    assert step1.tool_result is not None
    assert step1.tool_result.output == "42"
    assert step1.tool_result.success is True

    step2 = trajectory.steps[1]
    assert step2.step_number == 2
    assert step2.thought == "The result is 42. Task complete."
    assert step2.tool_call is None


@pytest.mark.asyncio
async def test_reasoning_loop_max_steps_exceeded() -> None:
    dispatcher = ToolDispatcher()
    dispatcher.register_tool(name="ping", description="Ping", handler=lambda: "pong")

    responses = [
        {
            "thought": f"Calling ping step {i}",
            "tool_call": {"tool_name": "ping", "arguments": {}},
        }
        for i in range(10)
    ]
    mock_llm = MockLLMClient(responses=responses)

    runner = ReasoningLoop(llm_client=mock_llm, dispatcher=dispatcher, max_steps=3)
    trajectory = await runner.run("Ping repeatedly")

    assert trajectory.status == TrajectoryStatus.MAX_STEPS_EXCEEDED
    assert len(trajectory.steps) == 3
    assert "maximum allowed steps" in (trajectory.error or "")


@pytest.mark.asyncio
async def test_reasoning_loop_llm_failure_handling() -> None:
    dispatcher = ToolDispatcher()
    mock_llm = MockLLMClient(responses=[RuntimeError("Fatal LLM crash")])

    runner = ReasoningLoop(llm_client=mock_llm, dispatcher=dispatcher, max_steps=5)
    trajectory = await runner.run("Crash task")

    assert trajectory.status == TrajectoryStatus.FAILED
    assert "LLM generation failed" in (trajectory.error or "")


@pytest.mark.asyncio
async def test_reasoning_loop_fallback_on_unformatted_response() -> None:
    dispatcher = ToolDispatcher()
    mock_llm = MockLLMClient(
        responses=[
            "I'm not sure what to do yet.",
            {"thought": "Now I will finish.", "final_answer": "Done."},
        ]
    )

    runner = ReasoningLoop(llm_client=mock_llm, dispatcher=dispatcher, max_steps=5)
    trajectory = await runner.run("Ambiguous task")

    assert trajectory.status == TrajectoryStatus.COMPLETED
    assert trajectory.final_answer == "Done."
    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].tool_call is None
    assert trajectory.steps[0].tool_result is None


@pytest.mark.asyncio
async def test_reasoning_loop_observation_truncation() -> None:
    dispatcher = ToolDispatcher()
    large_output = "X" * 1000
    dispatcher.register_tool(
        name="large_tool",
        description="Generates large output",
        handler=lambda: large_output,
    )

    mock_llm = MockLLMClient(
        responses=[
            {
                "thought": "Calling large tool.",
                "tool_call": {"tool_name": "large_tool", "arguments": {}},
            },
            {
                "thought": "Received output.",
                "final_answer": "Done.",
            },
        ]
    )

    runner = ReasoningLoop(
        llm_client=mock_llm,
        dispatcher=dispatcher,
        max_steps=5,
        max_observation_chars=200,
    )
    trajectory = await runner.run("Large output task")

    assert trajectory.status == TrajectoryStatus.COMPLETED
    assert len(mock_llm.call_history) == 2

    second_call_messages = mock_llm.call_history[1]
    obs_message = second_call_messages[-1]
    assert "[truncated" in obs_message.content
    assert len(obs_message.content) < 500
