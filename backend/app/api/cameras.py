from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.camera import Camera
from app.schemas.camera import CameraCreate, CameraResponse

router = APIRouter(prefix="/cameras", tags=["Cameras"])


@router.get("", response_model=List[CameraResponse])
def list_cameras(db: Session = Depends(get_db)):
    """List all registered cameras."""
    cameras = db.query(Camera).all()
    if not cameras:
        # Create default camera if none exists
        default_cam = Camera(name="Main Camera (Webcam 0)", source="0", location="Main Area", status="ONLINE")
        db.add(default_cam)
        db.commit()
        db.refresh(default_cam)
        cameras = [default_cam]
    return cameras


@router.post("", response_model=CameraResponse)
def create_camera(camera_in: CameraCreate, db: Session = Depends(get_db)):
    """Register a new camera source."""
    cam = Camera(
        name=camera_in.name,
        source=camera_in.source,
        location=camera_in.location,
        status=camera_in.status
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: str, db: Session = Depends(get_db)):
    """Get details for a specific camera."""
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam
