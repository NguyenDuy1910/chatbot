import asyncio
from typing import AsyncIterator, Dict, Union
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

from backend.app.stream.main import astream_state, to_sse


class MockRunnable(Runnable):
    """Mock implementation of a Runnable for testing."""

    def __init__(self, message: BaseMessage = None):
        """Initialize the MockRunnable class."""
        self.message = message

    async def mock_llm_stream(query: str) -> AsyncIterator[str]:
        """Mô phỏng luồng token từ LLM."""
        fake_tokens = [
            "Streaming ", "là ", "kỹ thuật ", "giúp ", "tối ưu ", "trả lời ", "thời gian thực."
        ]
        for token in fake_tokens:
            await asyncio.sleep(0.5)  # Giả lập thời gian trễ
            yield token

    async def invoke(self, input: Union[list[str], Dict[str, str]], config: RunnableConfig) -> str:
        """Mock implementation of the required 'invoke' method."""
        return "Mock result"

    async def astream_events(
        self,
        input: Union[list[str], Dict[str, str]],
        config: RunnableConfig,
        version: str = "v1",
        stream_mode: str = "values",
        exclude_tags: list[str] = None,
    ) -> AsyncIterator[Dict]:
        """Simulate a sequence of events for testing."""
        run_id = "mock-run-id-1234"
        yield {"event": "on_chain_start", "run_id": run_id}

        # Simulate streaming chunks
        for i in range(5):
            await asyncio.sleep(1)  # Simulate delay
            yield {
                "event": "on_chain_stream",
                "run_id": run_id,
                "data": {"chunk": [{"id": f"msg-{i}", "text": f"Chunk {i + 1}"}]},
            }

        # Simulate additional chat model stream messages
        for i in range(2):
            await asyncio.sleep(1)  # Simulate delay
            yield {
                "event": "on_chat_model_stream",
                "run_id": run_id,
                "data": {"chunk": {"id": f"chat-{i}", "text": f"Chat Message {i}"}},
            }

        # End of stream
        yield {"event": "end"}


async def demo_astream_state():
    """Test the astream_state function."""
    mock_runnable = MockRunnable()
    input_data = [{"id": "test-1", "text": "Hello, world!"}]
    config = RunnableConfig()

    print("Testing astream_state...")
    async for event in astream_state(mock_runnable, input_data, config):
        print("Event:", event)


async def demo_to_sse():
    """Test the to_sse function."""
    mock_runnable = MockRunnable()
    input_data = [{"id": "test-1", "text": "Hello, world!"}]
    config = RunnableConfig()

    # Get the message stream from astream_state
    messages_stream = astream_state(mock_runnable, input_data, config)

    print("Testing to_sse...")
    async for sse_event in to_sse(messages_stream):
        print("SSE Event:", sse_event)


# Run the demo
if __name__ == "__main__":
    asyncio.run(demo_astream_state())
    asyncio.run(demo_to_sse())
