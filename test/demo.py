from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()


async def fake_llm_stream(query: str):
    """Simulate streaming tokens from a fake LLM."""
    fake_tokens = [
        "This ", "is ", "a ", "streaming ", "response ", "from ", "a ", "fake ", "LLM.",
        " Hope ", "this ", "helps!"
    ]
    for token in fake_tokens:
        await asyncio.sleep(0.5)  # Simulate delay between tokens
        yield f"data: {token}\n\n"  # SSE format


@app.post("/fake-stream")
async def fake_chat_stream(request: Request):
    """Fake streaming endpoint for LLM responses."""
    body = await request.json()
    query = body.get("query", "")
    if not query:
        return {"error": "Query cannot be empty"}

    return StreamingResponse(
        fake_llm_stream(query),
        media_type="text/event-stream"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "OK"}
