import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
root_dir = backend_dir.parent
frontend_dir = root_dir / "frontend"

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.core.config import settings
from app.database.session import Base, engine

# Create SQLite tables if they do not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Intelligent Computer Vision & Person Activity Monitoring Backend API"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints
app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/health", tags=["Health"])
def health_check():
    """System health verification."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard():
    """Serves the Smart Camera web dashboard."""
    html_path = frontend_dir / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Dashboard index.html not found</h1>", status_code=404)


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard_alias():
    """Alias for accessing the dashboard."""
    return get_dashboard()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
