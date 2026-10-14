"""FastAPI Application entry point for AgentPulse."""

from __future__ import annotations

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router

app = FastAPI(
    title="AgentPulse API",
    version="1.0.0",
    description="Runtime Security and Behavioral Trust Framework for AI Agents",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration restricted to local dev frontend origins
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/api", tags=["Health"])
def api_root():
    """Return API metadata and endpoint index."""
    return JSONResponse(
        {
            "service": "AgentPulse API",
            "status": "online",
            "docs": "/docs",
            "api_endpoints": {
                "chat": "POST /api/chat",
                "events": "GET /api/events",
                "sessions": "GET /api/sessions",
                "metrics": "GET /api/metrics",
                "status": "GET /api/status",
            },
        }
    )

# Mount frontend build directory if it exists
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/", tags=["Health"])
    def root():
        return api_root()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.server:app", host="127.0.0.1", port=port, reload=True)
