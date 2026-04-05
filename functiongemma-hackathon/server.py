"""
CactusRoute — REST API Server
==============================

Exposes the 7-layer adaptive hybrid router as an HTTP service so that any
frontend or external tool can call it over the network.

Endpoints:
  POST /api/generate           — Run inference and return routing result
  GET  /api/health             — Service + model readiness check
  GET  /api/metrics            — Session-level aggregated statistics
  GET  /api/schema/examples    — Pre-built tool schemas for quick demos
  GET  /                       — Serve the web UI (static/index.html)

Usage:
    python server.py                        # default: http://localhost:8000
    python server.py --port 9000
    python server.py --mock                 # mock mode (no Cactus SDK needed)
    uvicorn server:app --reload             # development mode

Environment:
    GEMINI_API_KEY   — Required for cloud fallback (as in main.py)
    SERVER_MOCK      — Set to "1" to enable mock mode without Cactus
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

# ── FastAPI ────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Determine operating mode ───────────────────────────────────────────────────
MOCK_MODE: bool = os.environ.get("SERVER_MOCK", "").strip() == "1"

_generate_hybrid = None   # set in lifespan


# ══════════════════════════════════════════════════════════════════════════════
# Session Metrics — thread-safe rolling statistics
# ══════════════════════════════════════════════════════════════════════════════

class SessionMetrics:
    """Accumulates per-request statistics for the lifetime of the server."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._lock = threading.Lock()
        self._history: deque[dict] = deque(maxlen=maxlen)
        self._total_calls = 0
        self._on_device_calls = 0
        self._total_latency_ms = 0.0

    def record(self, result: dict) -> None:
        with self._lock:
            self._total_calls += 1
            latency = result.get("total_time_ms", 0)
            self._total_latency_ms += latency
            is_on_device = not result.get("source", "").startswith("cloud")
            if is_on_device:
                self._on_device_calls += 1
            self._history.append({
                "source": result.get("_detail", result.get("source", "unknown")),
                "difficulty": result.get("difficulty", "?"),
                "confidence": round(result.get("confidence", 0), 4),
                "latency_ms": round(latency, 1),
                "on_device": is_on_device,
                "num_calls": len(result.get("function_calls", [])),
            })

    def snapshot(self) -> dict:
        with self._lock:
            n = self._total_calls
            return {
                "total_requests": n,
                "on_device_requests": self._on_device_calls,
                "on_device_ratio": round(self._on_device_calls / n, 4) if n else 0.0,
                "avg_latency_ms": round(self._total_latency_ms / n, 1) if n else 0.0,
                "recent": list(self._history)[-20:],  # last 20 requests
            }


_metrics = SessionMetrics()


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan — import router on startup
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _generate_hybrid
    if MOCK_MODE:
        log.info("Starting in MOCK mode — Cactus SDK not required")
        _generate_hybrid = _mock_generate
    else:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cactus", "python", "src"))
            from main import generate_hybrid  # noqa: PLC0415
            _generate_hybrid = generate_hybrid
            log.info("CactusRoute main.py loaded successfully")
        except Exception as exc:
            log.warning("Could not import main.py (%s) — falling back to mock mode", exc)
            _generate_hybrid = _mock_generate
    yield
    # Cleanup happens via atexit in main.py


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="CactusRoute API",
    description=(
        "Adaptive edge/cloud hybrid router for function calling. "
        "Routes between FunctionGemma (on-device, 270M) and Gemini 2.5 Flash (cloud) "
        "using a 7-layer schema-driven framework."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ══════════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str = Field(..., examples=["user"])
    content: str = Field(..., examples=["What is the weather in Paris?"])


class ToolParameter(BaseModel):
    type: str = Field(..., examples=["string"])
    description: str = Field("", examples=["City name"])
    enum: list[str] | None = None


class ToolParameters(BaseModel):
    type: str = "object"
    properties: dict[str, ToolParameter] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class Tool(BaseModel):
    name: str = Field(..., examples=["get_weather"])
    description: str = Field(..., examples=["Get the current weather for a city"])
    parameters: ToolParameters


class GenerateRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    tools: list[Tool] = Field(..., min_length=1)


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class GenerateResponse(BaseModel):
    function_calls: list[FunctionCall]
    source: str
    detail: str
    difficulty: str
    confidence: float
    local_confidence: float | None
    total_time_ms: float
    cloud_handoff: bool
    spike_handoff: bool
    mock: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# Mock inference — used when Cactus SDK is unavailable
# ══════════════════════════════════════════════════════════════════════════════

def _mock_generate(messages: list[dict], tools: list[dict]) -> dict:
    """Return a deterministic mock result so the UI works without the SDK."""
    time.sleep(0.05)  # simulate a bit of latency
    tool = tools[0]
    args = {}
    for pname, pmeta in tool.get("parameters", {}).get("properties", {}).items():
        ptype = pmeta.get("type", "string")
        if ptype == "integer":
            args[pname] = 42
        elif ptype == "number":
            args[pname] = 3.14
        elif ptype == "boolean":
            args[pname] = True
        else:
            content = messages[-1]["content"] if messages else "demo"
            args[pname] = content[:40]
    return {
        "function_calls": [{"name": tool["name"], "arguments": args}],
        "source": "on-device",
        "_detail": "on-device (mock)",
        "difficulty": "easy",
        "confidence": 0.91,
        "local_confidence": None,
        "total_time_ms": 52.0,
        "cloud_handoff": False,
        "spike_handoff": False,
        "_mock": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Route a function-calling query through the 7-layer adaptive framework.

    Returns the selected function call(s) together with routing metadata such
    as the source (on-device / cloud), confidence score, difficulty rating, and
    end-to-end latency.
    """
    if _generate_hybrid is None:
        raise HTTPException(status_code=503, detail="Router not yet initialized")

    messages_raw = [m.model_dump() for m in req.messages]
    tools_raw = [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters.model_dump(exclude_none=True),
        }
        for t in req.tools
    ]

    try:
        result: dict = _generate_hybrid(messages_raw, tools_raw)
    except Exception as exc:
        log.exception("generate_hybrid raised an error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _metrics.record(result)

    return GenerateResponse(
        function_calls=[
            FunctionCall(name=c["name"], arguments=c.get("arguments", {}))
            for c in result.get("function_calls", [])
        ],
        source=result.get("source", "unknown"),
        detail=result.get("_detail", result.get("source", "unknown")),
        difficulty=result.get("difficulty", "unknown"),
        confidence=result.get("confidence", 0.0),
        local_confidence=result.get("local_confidence"),
        total_time_ms=result.get("total_time_ms", 0.0),
        cloud_handoff=result.get("cloud_handoff", False),
        spike_handoff=result.get("spike_handoff", False),
        mock=result.get("_mock", False),
    )


@app.get("/api/health")
async def health() -> JSONResponse:
    """Return service readiness and model information."""
    return JSONResponse({
        "status": "ok",
        "mock_mode": MOCK_MODE or (_generate_hybrid is _mock_generate),
        "router": "CactusRoute 7-layer adaptive",
        "models": {
            "on_device": "FunctionGemma-270M-IT (via Cactus SDK)",
            "cloud": os.environ.get("CLOUD_MODEL", "gemini-2.5-flash"),
        },
        "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY")),
    })


@app.get("/api/metrics")
async def metrics() -> JSONResponse:
    """Return aggregated session statistics."""
    return JSONResponse(_metrics.snapshot())


@app.get("/api/schema/examples")
async def schema_examples() -> JSONResponse:
    """
    Return a collection of pre-built tool schemas that can be loaded into the
    web UI for quick demos without having to type JSON by hand.
    """
    examples = [
        {
            "label": "Weather + Alarm (hard)",
            "query": "What's the weather in Tokyo and set my alarm for 7:30 AM?",
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                },
                {
                    "name": "set_alarm",
                    "description": "Set an alarm for a given time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hour":   {"type": "integer", "description": "Hour (0–23)"},
                            "minute": {"type": "integer", "description": "Minute (0–59)"},
                        },
                        "required": ["hour", "minute"],
                    },
                },
            ],
        },
        {
            "label": "Send message (medium)",
            "query": "Send Alice a message saying 'see you at 5!'",
            "tools": [
                {
                    "name": "send_message",
                    "description": "Send a message to a contact",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipient": {"type": "string", "description": "Contact name"},
                            "message":   {"type": "string", "description": "Message text"},
                        },
                        "required": ["recipient", "message"],
                    },
                },
            ],
        },
        {
            "label": "Play music (easy)",
            "query": "Play Bohemian Rhapsody",
            "tools": [
                {
                    "name": "play_song",
                    "description": "Play a song by title",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Song title"},
                        },
                        "required": ["title"],
                    },
                },
            ],
        },
        {
            "label": "Set timer (easy)",
            "query": "Set a timer for 10 minutes",
            "tools": [
                {
                    "name": "set_timer",
                    "description": "Set a countdown timer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "duration": {
                                "type": "integer",
                                "description": "Duration in seconds",
                            },
                        },
                        "required": ["duration"],
                    },
                },
            ],
        },
        {
            "label": "Navigation + reminder (hard)",
            "query": "Navigate to the nearest coffee shop and remind me to pick up milk",
            "tools": [
                {
                    "name": "navigate_to",
                    "description": "Start navigation to a destination",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "destination": {
                                "type": "string",
                                "description": "Place or address to navigate to",
                            },
                        },
                        "required": ["destination"],
                    },
                },
                {
                    "name": "create_reminder",
                    "description": "Create a reminder with a title",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Reminder text",
                            },
                        },
                        "required": ["title"],
                    },
                },
            ],
        },
    ]
    return JSONResponse(examples)


# ── Static files & SPA fallback ───────────────────────────────────────────────

_static_dir = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(os.path.join(_static_dir, "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def root_no_ui() -> JSONResponse:
        return JSONResponse({
            "message": "CactusRoute API is running. See /docs for the OpenAPI UI.",
            "docs": "/docs",
        })


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CactusRoute REST API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no Cactus SDK required)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = _parse_args()
    if args.mock:
        os.environ["SERVER_MOCK"] = "1"

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
