import asyncio
import websockets

async def test():
    uri = "ws://127.0.0.1:8001/ws/alerts"
    print("🔌 Connexion...")
    async with websockets.connect(uri) as websocket:
        print("✅ Connecté. En attente de notifications...")
        while True:
            message = await websocket.recv()
            print("📨 Notification reçue :", message)

asyncio.run(test())
