import asyncio
import websockets

async def websocket_client():
    uri = "ws://localhost:8000/ws/client3"
    async with websockets.connect(uri) as websocket:
        # Send a message to the WebSocket server
        await websocket.send("Hello, world!")
        print("Message sent: Hello, world!")

        # Wait for a response
        response = await websocket.recv()
        print(f"Received: {response}")

# Run the WebSocket client
if __name__ == "__main__":
    asyncio.run(websocket_client())
