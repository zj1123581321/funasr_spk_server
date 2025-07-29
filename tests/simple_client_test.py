"""
简单的WebSocket客户端测试
"""
import asyncio
import websockets
import json

async def test_connection():
    uri = "ws://localhost:8766"
    try:
        print(f"连接到: {uri}")
        websocket = await websockets.connect(uri)
        
        print("连接成功!")
        
        # 接收欢迎消息
        message = await websocket.recv()
        data = json.loads(message)
        print(f"收到消息: {data}")
        
        # 发送ping消息
        ping_message = {
            "type": "ping",
            "data": {}
        }
        await websocket.send(json.dumps(ping_message))
        print("发送ping消息")
        
        # 等待pong响应
        response = await websocket.recv()
        response_data = json.loads(response)
        print(f"收到响应: {response_data}")
        
        await websocket.close()
        print("连接已关闭")
        
    except Exception as e:
        print(f"连接失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())