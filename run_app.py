"""Convenience launcher for AgentPulse.

Starts the FastAPI backend server on http://127.0.0.1:8000 which automatically
serves the production frontend at http://127.0.0.1:8000/ and API endpoints at /api.

Optionally can also launch Vite dev server on port 5173.
"""

from __future__ import annotations

import os
import sys
import subprocess
import uvicorn

def main():
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")

    print("=" * 60)
    print("  AgentPulse: Runtime Security & Behavioral Trust Framework")
    print("=" * 60)
    print(f"Backend API & Web App: http://{host}:{port}/")
    print(f"Security Dashboard:    http://{host}:{port}/#/dashboard")
    print(f"AI Agent Chat:         http://{host}:{port}/#/chat")
    print(f"Interactive API Docs:  http://{host}:{port}/docs")
    print("=" * 60)

    uvicorn.run("api.server:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    main()
