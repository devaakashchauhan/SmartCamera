import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, String

from app.database.session import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    source = Column(String(255), nullable=False, default="0")
    location = Column(String(100), default="Main Area")
    status = Column(String(20), default="ONLINE")
    created_at = Column(DateTime, default=datetime.utcnow)
