import logging
import os
import asyncio
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from queue import Queue

from langchain_core.outputs.llm_result import LLMResult
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from starlette.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect, WebSocket

load_dotenv()
logger = logging.getLogger(__name__)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# initialize the agent (we need to do this for the callbacks)
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPEN_API_KEY"),
    temperature=0.0,
    model_name="gpt-3.5-turbo",
    streaming=True,  # ! important
    callbacks=[]  # ! important (but we will add them later)
)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    output_key="output"
)
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[],
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    return_intermediate_steps=False
)


class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False

    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    print(token)
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""


async def run_call(query: str, stream_it: AsyncCallbackHandler):
    # assign callback handler
    agent.agent.llm_chain.llm.callbacks = [stream_it]
    # now query
    await agent.acall(inputs={"input": query})


# request input format
class Query(BaseModel):
    text: str


async def create_gen(query: str, stream_it: AsyncCallbackHandler):
    task = asyncio.create_task(run_call(query, stream_it))
    async for token in stream_it.aiter():
        yield token
    await task
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}  # {session_id: websocket}
        self.timers: Dict[str, asyncio.Task] = {}  # {session_id: timer_task}
    async def connect(self, websocket: WebSocket, session_id: str, timeout: int):
        """Add a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.timers[session_id] = asyncio.create_task(self.close_after_timeout(session_id, timeout))

    async def close_after_timeout(self, session_id: str, timeout: int):
        """Close the connection after a timeout."""
        await asyncio.sleep(timeout)
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_text("Session timed out. Connection will be closed.")
            await websocket.close()
            self.disconnect(session_id)
            print(f"Session {session_id} timed out and connection closed.")
    def disconnect(self, session_id: str):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_to_client(self, session_id: str, message: str):
        """Send a message to a specific client."""
        websocket = self.active_connections.get(session_id)
        if websocket:
            await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Send a message to all connected clients."""
        for websocket in self.active_connections.values():
            await websocket.send_text(message)


manager = ConnectionManager()

@app.post("/chat")
async def chat(
        query: Query = Body(...),
):
    stream_it = AsyncCallbackHandler()
    gen = create_gen(query.text, stream_it)
    return StreamingResponse(gen, media_type="text/event-stream")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time interaction"""
    timeout = 30000
    await manager.connect(websocket, session_id, timeout)
    print(f"Client {session_id} connected.")

    try:
        while True:
            data = await websocket.receive_text()
            # Create the AsyncCallbackHandler to stream tokens
            stream_it = AsyncCallbackHandler()
            gen = create_gen(data, stream_it)  # Pass the query and handler
            # Stream response from AI token-by-token
            async for token in gen:
                print(f"Received token demo: {token}")
                await manager.send_to_client(session_id, token)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "🤙"}


if __name__ == "__main__":
    uvicorn.run(
        "streaming:app",
        host="localhost",
        port=8000,
        reload=True
    )


