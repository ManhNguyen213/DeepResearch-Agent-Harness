"""Layer 1: Chainlit Interactive Web UI with Streaming Thought Process and Memory Inspector."""

import chainlit as cl

from src.harness.graph import build_harness_graph
from src.schemas.state import HarnessState


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize a new Deep Research Agent Harness session."""
    await cl.Message(
        content=(
            "**Deep Research Agent Harness (GR2)** initialized.\n"
            "- **Context Isolation:** Enabled (`StructuredEvidenceCard`)\n"
            "- **3-Tier Cognitive Memory:** Ready (`ACTIVE` / `SUPERSEDED` tracking)"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Execute the LangGraph Harness workflow and stream intermediate steps."""
    graph = build_harness_graph().compile()
    initial_state = HarnessState(user_query=message.content)
    result = graph.invoke(initial_state)
    await cl.Message(content=result.get("final_report", "No report generated.")).send()
