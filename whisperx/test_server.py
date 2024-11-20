
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from whisperx.async_pipeline import WhisperXRealtimePipeline

app = FastAPI(title="WhisperX Realtime API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create pipeline instance
pipeline = WhisperXRealtimePipeline()

@app.websocket("/ws/transcribe")
async def transcribe(websocket: WebSocket):
    await websocket.accept()
    try:
        await pipeline.handle_websocket_stream(websocket)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in processing audio stream: {e}")
        await websocket.close(code=1001)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "async_pipeline:app",
        host="0.0.0.0",
        port=8765,
        reload=True,
        workers=1
    )