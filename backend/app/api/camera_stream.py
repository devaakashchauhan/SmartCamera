from typing import Optional
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.camera_worker import CameraWorker

router = APIRouter(prefix="/camera", tags=["Camera Stream & Control"])


class StartStreamRequest(BaseModel):
    source: Optional[str] = None
    youtube_url: Optional[str] = None


@router.get("/stream")
def get_camera_stream():
    """
    Returns an ultra-low latency multipart MJPEG video stream with AI overlays.
    Renderable directly in HTML with `<img src="/api/camera/stream" />`.
    """
    worker = CameraWorker.get_instance()
    return StreamingResponse(
        worker.stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/upload-frame")
async def upload_frame(request: Request):
    """
    Allows the standalone AI engine (main.py) to stream pre-annotated JPEG frames into the web player
    without causing webcam locking conflicts.
    """
    body = await request.body()
    if body:
        CameraWorker.get_instance().set_external_frame(body)
        return {"status": "ok"}
    return {"status": "empty"}


@router.post("/upload-raw-frame")
async def upload_raw_frame(request: Request):
    """
    Receives raw unannotated JPEG frames from client webcam (browser or laptop Python script).
    Buffers asynchronously (<1ms) so the client is never bottlenecked by network Stop-and-Wait latency.
    """
    body = await request.body()
    if body:
        worker = CameraWorker.get_instance()
        worker.push_raw_frame(body)
        return {"status": "ok"}
    return {"status": "empty"}


@router.post("/start")
def start_camera(source: int | str = 0):
    """Starts the real-time AI camera worker on the specified source."""
    worker = CameraWorker.get_instance()
    success = worker.start(source=source)
    return {"status": "started" if success else "failed", "source": source}


@router.post("/start-youtube")
def start_youtube_stream(req: StartStreamRequest):
    """
    Starts real-time AI computer vision on a live YouTube stream.
    Resolves the stream directly within Google Colab GPU for ultra-high FPS (20-30 FPS).
    """
    url = req.youtube_url or req.source
    if not url:
        return {"status": "error", "message": "Missing youtube_url"}
    worker = CameraWorker.get_instance()
    success = worker.start(source=url)
    return {"status": "started" if success else "failed", "source": url}


@router.post("/stop")
def stop_camera():
    """Stops the camera worker and releases hardware resources."""
    worker = CameraWorker.get_instance()
    worker.stop()
    return {"status": "stopped"}


@router.get("/status")
def get_camera_status():
    """Returns current camera running status and hardware telemetry."""
    worker = CameraWorker.get_instance()
    return {
        "running": worker.is_running,
        "fps": round(worker.fps, 1),
        "device": worker.device,
        "show_zones": worker.show_zones,
        "show_objects": worker.show_objects
    }


@router.post("/toggle-zones")
def toggle_zones():
    """Toggles spatial zone overlay."""
    worker = CameraWorker.get_instance()
    worker.show_zones = not worker.show_zones
    return {"show_zones": worker.show_zones}


@router.post("/toggle-objects")
def toggle_objects():
    """Toggles object detection overlay."""
    worker = CameraWorker.get_instance()
    worker.show_objects = not worker.show_objects
    return {"show_objects": worker.show_objects}
