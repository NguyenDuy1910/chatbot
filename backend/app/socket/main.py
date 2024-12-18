from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from typing import List
import uvicorn
from redis.asyncio import Redis
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Redis connection URL
REDIS_URL = "redis://localhost:6379"

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origins for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Connection manager to handle WebSocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and track new WebSocket connections."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """Send a message to all active WebSocket clients."""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error broadcasting to connection: {e}")


# Redis connection manager
class RedisConnectionManager:
    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url)

    async def publish(self, channel: str, message: str):
        """Publish a message to a Redis channel."""
        await self.redis.publish(channel, message)

    async def close(self):
        """Close the Redis connection."""
        await self.redis.close()


# Initialize managers
manager = ConnectionManager()
redis_manager = RedisConnectionManager(REDIS_URL)


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    print("Application startup: Redis connection established!")
    asyncio.create_task(redis_listener())  # Start Redis listener


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    await redis_manager.close()
    print("Application shutdown: Redis connection closed!")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle WebSocket connections."""
    await manager.connect(websocket)
    try:
        # Notify other clients about the new connection
        await redis_manager.publish("chat_channel", f"Client {client_id} joined the chat")
        async for message in websocket.iter_text():
            # Publish messages to Redis
            await redis_manager.publish("chat_channel", f"Client {client_id}: {message}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # Notify about the disconnection
        await redis_manager.publish("chat_channel", f"Client {client_id} left the chat")
    except Exception as e:
        print(f"Error in WebSocket endpoint: {e}")


# Redis subscription listener
async def redis_listener():
    """Listen for messages on Redis and broadcast to WebSocket clients."""
    try:
        pubsub = redis_manager.redis.pubsub()
        await pubsub.subscribe("chat_channel")

        async for message in pubsub.listen():
            if message["type"] == "message":
                # Broadcast received messages to all WebSocket clients
                await manager.broadcast(message["data"].decode("utf-8"))
    except Exception as e:
        print(f"Error in Redis listener: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Replace "main" with your filename if different
        host="localhost",
        port=8000,
        reload=True
    )
