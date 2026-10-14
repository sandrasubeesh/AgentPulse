"""Convenience launcher for the AgentPulse FastAPI server."""

from __future__ import annotations

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"Starting AgentPulse API on http://{host}:{port}")
    uvicorn.run("api.server:app", host=host, port=port, reload=True)
