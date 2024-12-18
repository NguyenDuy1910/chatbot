from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
import json
import uvicorn  # Import Uvicorn to run the application programmatically


app = FastAPI()

# Mock dependencies
async def get_verified_user():
    """Mock user verification function."""
    return {
        "id": 1,
        "name": "Test User",
        "email": "test@example.com",
        "role": "user"
    }

async def get_all_models():
    """Mock function to retrieve all models."""
    return [
        {"id": "model_1", "owned_by": "arena", "info": {"meta": {"filter_mode": "exclude"}}},
        {"id": "model_2", "owned_by": "ollama"}
    ]


from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import logging

# Thiết lập logger
logger = logging.getLogger("websocket")
logger.setLevel(logging.INFO)


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to stream messages to the client.
    Streams 10 chunks of messages to the connected client with a 1-second delay between each.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    try:
        for i in range(10):  # Simulate streaming 10 messages
            data = {
                "type": "response",
                "chunk_number": i + 1,
                "message": f"This is chunk {i + 1} of the streamed response."
            }
            await websocket.send_json(data)  # Send data as JSON
            logger.info(f"Sent chunk {i + 1} to client.")
            await asyncio.sleep(1)  # Simulate delay between chunks

    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected by the client.")
    except asyncio.CancelledError:
        logger.warning("WebSocket task was cancelled.")
    except Exception as e:
        logger.error(f"Unexpected error during WebSocket communication: {e}")
    finally:
        logger.info("WebSocket connection closed.")


@app.post("/api/chat/completions")
async def generate_chat_completions(
    form_data: dict, user=Depends(get_verified_user)
):
    """Simulate streaming chat completions."""
    if form_data.get("stream"):
        async def stream_response():
            for i in range(10):  # Simulate 10 chunks of streamed data
                data = {
                    "chunk_number": i + 1,
                    "message": f"This is chunk {i + 1} of the streamed response."
                }
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(1)  # Simulate delay between chunks

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    # For non-streaming requests, return a simple response
    return {
        "message": "This is a non-streaming response.",
        "form_data": form_data
    }

def main():
    """Run the FastAPI application."""
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
