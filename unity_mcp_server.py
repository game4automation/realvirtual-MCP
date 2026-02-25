#!/usr/bin/env python3
"""
realvirtual Unity MCP Server
=============================
WebSocket bridge between realvirtual.io Unity Digital Twin and MCP clients.

Unity defines MCP tools via C# [McpTool] attributes. This Python server
discovers them automatically via WebSocket and registers them as FastMCP tools.

Modes:
  stdio  - For Claude Desktop / Claude Code (default)
  sse    - HTTP/SSE for network clients

Usage:
  python unity_mcp_server.py                          # stdio mode, auto-discover port
  python unity_mcp_server.py --mode sse --http-port 8080
  python unity_mcp_server.py --ws-port 18712          # explicit Unity port
  python unity_mcp_server.py --project-path "E:/.../Assets"  # connect to specific instance

WebSocket Protocol (Port 18711):
  __discover__  -> {tools: [{name, description, inputSchema}], schema_version}
  __call__      -> {result: "..."} or {error: "..."}
  __auth__      -> {status: "ok"} or {error: "invalid token"}
  __heartbeat__ -> {status: "ok", tools_count: N}
"""

import argparse
import asyncio
import base64
import ctypes
import ctypes.wintypes
import enum
import json
import logging
import os
import struct
import sys
import tempfile
import time
import zlib
from collections import deque
from pathlib import Path
from typing import Any

# Optional: websockets may not be installed in all environments
try:
    import websockets
    import websockets.exceptions
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import InitializedNotification

logger = logging.getLogger("unity_mcp")

# Default WebSocket config
DEFAULT_WS_HOST = "127.0.0.1"
DEFAULT_WS_PORT = 18711
DEFAULT_WS_PATH = "/mcp"
DEFAULT_HTTP_PORT = 8080

# Reconnect config
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0
RECONNECT_MULTIPLIER = 2.0

# Tool call timeout (seconds) - default for most tools
TOOL_CALL_TIMEOUT = 15.0

# Extended timeout for heavy tools (screenshots, validation, etc.)
TOOL_CALL_TIMEOUT_LONG = 60.0
LONG_TIMEOUT_TOOLS = {"screenshot_editor", "screenshot_game", "screenshot_scene",
                      "assetstore_validate", "validation_run", "test_run"}

# Watchdog interval (seconds) - heartbeat check when connected
WATCHDOG_INTERVAL = 3.0

# Cache file name (stored next to this script)
CACHE_FILENAME = "tool_schema_cache.json"

# Message buffer config
BUFFER_MAX_SIZE = 100
BUFFER_TTL_SECONDS = 30.0

# Circuit breaker config
CB_FAIL_THRESHOLD = 3
CB_RESET_TIMEOUT = 15.0

# Max reconnect attempts before ERROR state
MAX_RECONNECT_ATTEMPTS = 10

# Discovery directory for multi-instance support
DISCOVERY_DIR = Path.home() / ".unity-mcp"

# Max age in seconds for a status file to be considered valid
DISCOVERY_MAX_AGE = 30.0

# Unity window wake-up: minimum interval between attempts (seconds)
FOCUS_MIN_INTERVAL = 0.5
_last_focus_time: float = 0.0


def _focus_unity_window() -> bool:
    """Wake Unity Editor so it processes MCP calls promptly.

    Unity throttles EditorApplication.update to ~2Hz when not focused, causing
    MCP WebSocket calls to timeout. This function uses two strategies:

    1. PostMessageW(WM_NULL) -- Posts a no-op message directly to Unity's Win32
       message queue. This interrupts WaitMessage() and causes Unity to pump
       its message loop (and thus EditorApplication.update) regardless of focus.
       WM_NULL is the safest possible message -- it does nothing by definition.

    2. SetForegroundWindow -- Attempts to bring Unity to the foreground for full
       speed updates. This is best-effort due to Windows focus-stealing prevention.

    Rate-limited to at most once per FOCUS_MIN_INTERVAL seconds.
    Returns True if the window was found and a wake-up was attempted.
    """
    global _last_focus_time

    if sys.platform != "win32":
        return False

    now = time.monotonic()
    if now - _last_focus_time < FOCUS_MIN_INTERVAL:
        return True  # Skipped (recently woken)

    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        hwnd = user32.FindWindowW("UnityContainerWndClass", None)
        if not hwnd:
            logger.debug("Focus: Unity window not found")
            return False

        # Always post WM_NULL to wake Unity's message loop (works without focus)
        WM_NULL = 0x0000
        user32.PostMessageW(hwnd, WM_NULL, 0, 0)

        # Check if Unity is already the foreground window
        fg_hwnd = user32.GetForegroundWindow()
        if fg_hwnd == hwnd:
            _last_focus_time = now
            return True

        # If window is minimized, restore it first
        SW_RESTORE = 9
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)

        # AttachThreadInput trick: attach our thread to the foreground window's
        # thread so Windows allows us to call SetForegroundWindow
        our_tid = kernel32.GetCurrentThreadId()
        fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None)

        attached = False
        try:
            if our_tid != fg_tid:
                attached = bool(user32.AttachThreadInput(our_tid, fg_tid, True))

            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(our_tid, fg_tid, False)

        _last_focus_time = now
        logger.debug("Focus: Unity window activated")
        return True
    except Exception as e:
        logger.debug(f"Focus: failed - {e}")
        return False


def discover_unity_port(project_path: str | None = None) -> int | None:
    """Discover the WebSocket port for a Unity instance from status files.

    Scans ~/.unity-mcp/unity-mcp-status-*.json for live Unity instances.
    If project_path is given, finds the matching instance by path.
    Otherwise, returns the most recently heartbeated instance.

    Returns the ws_port or None if no live instance found.
    """
    if not DISCOVERY_DIR.exists():
        return None

    candidates = []
    for status_file in DISCOVERY_DIR.glob("unity-mcp-status-*.json"):
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
            ws_port = data.get("ws_port")
            file_project_path = data.get("project_path", "")
            reloading = data.get("reloading", False)
            heartbeat_str = data.get("last_heartbeat", "")
            pid = data.get("pid")

            if not ws_port:
                continue

            # Check if the process is still alive
            if pid:
                try:
                    import os
                    os.kill(pid, 0)  # signal 0 = check existence
                except (OSError, ProcessLookupError):
                    # Process dead, clean up stale file
                    try:
                        status_file.unlink()
                    except Exception:
                        pass
                    continue

            # Check heartbeat freshness
            if heartbeat_str:
                try:
                    from datetime import datetime, timezone
                    heartbeat_time = datetime.fromisoformat(
                        heartbeat_str.replace("Z", "+00:00"))
                    age = (datetime.now(timezone.utc) - heartbeat_time).total_seconds()
                    if age > DISCOVERY_MAX_AGE:
                        continue
                except (ValueError, TypeError):
                    pass

            candidates.append({
                "ws_port": ws_port,
                "project_path": file_project_path,
                "reloading": reloading,
                "heartbeat": heartbeat_str,
                "file": status_file.name,
            })
        except (json.JSONDecodeError, OSError):
            continue

    if not candidates:
        return None

    # If project_path given, find exact match
    if project_path:
        # Normalize for comparison
        norm_target = project_path.replace("\\", "/").rstrip("/").lower()
        for c in candidates:
            norm_path = c["project_path"].replace("\\", "/").rstrip("/").lower()
            if norm_path == norm_target:
                logger.info(
                    f"Discovered Unity instance for project: "
                    f"port {c['ws_port']} ({c['file']})"
                )
                return c["ws_port"]

    # No exact match or no project_path - return first non-reloading, or first
    non_reloading = [c for c in candidates if not c["reloading"]]
    best = non_reloading[0] if non_reloading else candidates[0]
    logger.info(
        f"Discovered Unity instance: port {best['ws_port']} ({best['file']})"
    )
    return best["ws_port"]


class State(enum.Enum):
    """Connection states for the Unity sidecar relay."""
    STARTING = "starting"           # Initial state, loading cache
    READY = "ready"                 # Connected, forwarding tool calls
    RELOADING = "reloading"         # Unity signaled domain_reload, buffering
    RECONNECTING = "reconnecting"   # Unexpected disconnect, buffering
    ERROR = "error"                 # Max retries exceeded, fail fast
    SHUTDOWN = "shutdown"           # Unity requested clean shutdown, exit process


class MessageBuffer:
    """Bounded message queue with TTL for Unity downtime."""

    def __init__(self, max_size: int = BUFFER_MAX_SIZE,
                 ttl_seconds: float = BUFFER_TTL_SECONDS):
        self._queue: deque = deque(maxlen=max_size)
        self._ttl = ttl_seconds

    @property
    def size(self) -> int:
        return len(self._queue)

    def is_full(self) -> bool:
        return len(self._queue) >= (self._queue.maxlen or BUFFER_MAX_SIZE)

    def enqueue(self, tool_name: str, arguments: dict,
                future: asyncio.Future) -> int:
        """Queue a tool call with its response future. Returns queue position."""
        self._queue.append({
            "tool": tool_name,
            "arguments": arguments,
            "future": future,
            "queued_at": time.monotonic(),
        })
        return len(self._queue)

    async def flush(self, call_tool_fn) -> tuple[int, int]:
        """Replay queued messages after reconnect. Returns (replayed, discarded)."""
        replayed = 0
        discarded = 0
        while self._queue:
            item = self._queue.popleft()
            age = time.monotonic() - item["queued_at"]
            if age > self._ttl:
                if not item["future"].done():
                    item["future"].set_result(json.dumps({
                        "error": f"Queued call to '{item['tool']}' expired "
                                 f"after {age:.0f}s (TTL={self._ttl:.0f}s)"
                    }))
                discarded += 1
                continue
            try:
                result = await call_tool_fn(item["tool"], item["arguments"])
                if not item["future"].done():
                    item["future"].set_result(result)
                replayed += 1
            except Exception as e:
                if not item["future"].done():
                    item["future"].set_result(json.dumps({
                        "error": f"Replay failed for '{item['tool']}': {e}"
                    }))
                discarded += 1
        return replayed, discarded

    def reject_all(self, reason: str):
        """Reject all queued messages."""
        while self._queue:
            item = self._queue.popleft()
            if not item["future"].done():
                item["future"].set_result(json.dumps({"error": reason}))


class CircuitBreaker:
    """Simple circuit breaker for Unity WebSocket calls."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, fail_threshold: int = CB_FAIL_THRESHOLD,
                 reset_timeout: float = CB_RESET_TIMEOUT):
        self.state = self.CLOSED
        self.failure_count = 0
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time: float = 0

    def record_success(self):
        self.failure_count = 0
        self.state = self.CLOSED

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.fail_threshold:
            self.state = self.OPEN

    def can_execute(self) -> bool:
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            if time.monotonic() - self.last_failure_time > self.reset_timeout:
                self.state = self.HALF_OPEN
                return True
            return False
        if self.state == self.HALF_OPEN:
            return True
        return False

    def reset(self):
        """Force reset to CLOSED state (e.g. after successful reconnect)."""
        self.state = self.CLOSED
        self.failure_count = 0


class UnityConnection:
    """Manages the WebSocket connection to Unity and tool discovery/execution."""

    def __init__(self, host: str, port: int, path: str = DEFAULT_WS_PATH,
                 auth_token: str | None = None):
        self.host = host
        self.port = port
        self.path = path
        self.auth_token = auth_token
        self.ws: Any = None
        self._connected = False
        self._tools: list[dict] = []
        self._schema_version: str = ""
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._lock = asyncio.Lock()
        self._last_comm_time: float = 0
        # Sidecar relay state machine
        self._state = State.STARTING
        self._reconnect_attempts = 0
        self.buffer = MessageBuffer()
        self.circuit_breaker = CircuitBreaker()

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}{self.path}"

    @property
    def connected(self) -> bool:
        return self._connected and self.ws is not None

    @property
    def tools(self) -> list[dict]:
        return self._tools

    @property
    def state(self) -> State:
        return self._state

    def _set_state(self, new_state: State):
        """Transition to a new state with logging."""
        old = self._state
        if old != new_state:
            self._state = new_state
            logger.info(f"State: {old.value} -> {new_state.value}")

    async def connect(self) -> bool:
        """Connect to Unity WebSocket server."""
        if not HAS_WEBSOCKETS:
            logger.error("websockets package not installed")
            return False

        try:
            logger.info(f"Connecting to Unity at {self.ws_url}")
            self.ws = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    ping_interval=10,
                    ping_timeout=5,
                    close_timeout=3,
                ),
                timeout=5.0,
            )
            self._connected = True
            self._reconnect_delay = RECONNECT_BASE_DELAY
            self._reconnect_attempts = 0
            self._last_comm_time = time.monotonic()
            logger.info("WebSocket connected")

            # Authenticate if token provided
            if self.auth_token:
                if not await self._authenticate():
                    await self.disconnect()
                    return False

            # Discover tools
            if not await self._discover():
                logger.warning("Discovery failed, using cached schemas if available")

            self._set_state(State.READY)
            self.circuit_breaker.reset()
            return True

        except (OSError, ConnectionRefusedError, asyncio.TimeoutError) as e:
            logger.warning(f"Cannot connect to Unity: {e}")
            self._connected = False
            self._reconnect_attempts += 1
            if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                self._set_state(State.ERROR)
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self._connected = False
            self._reconnect_attempts += 1
            if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                self._set_state(State.ERROR)
            return False

    async def disconnect(self):
        """Close WebSocket connection."""
        logger.debug("disconnect() called")
        self._connected = False
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

    async def ensure_connected(self) -> bool:
        """Ensure connection is alive, reconnect if needed.

        Called before each tool invocation for automatic recovery
        after Unity domain reloads or restarts.
        """
        if self.connected:
            return True

        # Try to reconnect
        logger.info("Not connected, attempting reconnect...")
        return await self.connect()

    async def _send_command(self, command: dict) -> dict | None:
        """Send a command to Unity and wait for response."""
        if not self.connected:
            logger.debug(f"_send_command skipped (not connected): {command.get('command', '?')}")
            return None

        cmd_name = command.get('command', '?')
        tool_name = command.get('tool', '')
        label = f"{cmd_name}" + (f"/{tool_name}" if tool_name else "")

        # Focus Unity window before tool calls so it processes promptly
        if cmd_name == "__call__":
            _focus_unity_window()

        try:
            timeout = (TOOL_CALL_TIMEOUT_LONG
                       if tool_name in LONG_TIMEOUT_TOOLS
                       else TOOL_CALL_TIMEOUT)
            async with self._lock:
                payload = json.dumps(command)
                logger.debug(f">> {label} ({len(payload)} bytes, timeout={timeout}s)")
                await self.ws.send(payload)
                response = await asyncio.wait_for(
                    self.ws.recv(),
                    timeout=timeout
                )
                self._last_comm_time = time.monotonic()
                logger.debug(f"<< {label} ({len(response)} bytes)")
                return json.loads(response)
        except asyncio.TimeoutError:
            logger.error(f"Timeout ({timeout}s) waiting for {label}")
            self.circuit_breaker.record_failure()
            return None
        except websockets.exceptions.ConnectionClosed as e:
            self._connected = False
            close_reason = getattr(e, "reason", "") or ""
            if hasattr(e, "rcvd") and e.rcvd:
                close_reason = e.rcvd.reason or close_reason
            if "shutdown" in close_reason:
                self._set_state(State.SHUTDOWN)
                logger.info(f"Unity requested shutdown during {label}")
            elif "domain_reload" in close_reason:
                self._set_state(State.RELOADING)
                logger.info(f"Unity signaled domain reload during {label}")
            else:
                self._set_state(State.RECONNECTING)
                logger.warning(f"Unity disconnected during {label}: {close_reason}")
            return None
        except Exception as e:
            logger.warning(f"WebSocket error during {label}: {e}")
            self._connected = False
            self.circuit_breaker.record_failure()
            if self._state == State.READY:
                self._set_state(State.RECONNECTING)
            return None

    async def _authenticate(self) -> bool:
        response = await self._send_command({
            "command": "__auth__",
            "token": self.auth_token
        })
        if response and response.get("status") == "ok":
            logger.info("Authentication successful")
            return True
        error = response.get("error", "unknown") if response else "no response"
        logger.error(f"Authentication failed: {error}")
        return False

    async def _discover(self) -> bool:
        response = await self._send_command({"command": "__discover__"})
        if response and "tools" in response:
            self._tools = response["tools"]
            self._schema_version = response.get("schema_version", "unknown")
            logger.info(
                f"Discovered {len(self._tools)} tools "
                f"(schema v{self._schema_version})"
            )
            return True
        return False

    async def heartbeat(self) -> dict | None:
        return await self._send_command({"command": "__heartbeat__"})

    async def _execute_tool_call(self, tool_name: str,
                                arguments: dict) -> str:
        """Forward a tool call to Unity (READY state only)."""
        response = await self._send_command({
            "command": "__call__",
            "tool": tool_name,
            "arguments": arguments,
        })

        # If command failed, try one reconnect + retry
        if response is None and not self.connected:
            logger.info(f"Retrying {tool_name} after reconnect...")
            if await self.ensure_connected():
                response = await self._send_command({
                    "command": "__call__",
                    "tool": tool_name,
                    "arguments": arguments,
                })

        if response is None:
            return json.dumps({
                "error": f"No response from Unity for tool '{tool_name}'. "
                         "Unity may be reloading.",
                "state": self._state.value,
            })

        if "error" in response:
            return json.dumps({"error": response["error"]})

        result = response.get("result", response)
        return result if isinstance(result, str) else json.dumps(result)

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool on Unity side with state-aware handling."""
        logger.debug(
            f"call_tool({tool_name}) state={self._state.value} "
            f"connected={self.connected}"
        )

        # STATE: ERROR - fail fast
        if self._state == State.ERROR:
            return json.dumps({
                "error": "Unity Editor not responding",
                "message": "Is Unity running? Check Unity Editor is open. "
                           "Use unity_reconnect to retry.",
                "state": "error",
            })

        # STATE: RELOADING or RECONNECTING - buffer the call
        if self._state in (State.RELOADING, State.RECONNECTING):
            if self.buffer.is_full():
                return json.dumps({
                    "error": "Message queue full",
                    "message": "Unity is recompiling. Too many queued calls.",
                    "state": self._state.value,
                })

            loop = asyncio.get_running_loop()
            future = loop.create_future()
            pos = self.buffer.enqueue(tool_name, arguments, future)

            state_msg = ("recompiling (domain reload)"
                         if self._state == State.RELOADING
                         else "reconnecting")
            logger.info(
                f"Queued '{tool_name}' (position {pos}) - "
                f"Unity is {state_msg}"
            )

            # Wait for the future to resolve (watchdog will flush after reconnect)
            try:
                result = await asyncio.wait_for(future, timeout=BUFFER_TTL_SECONDS + 5)
                return result
            except asyncio.TimeoutError:
                return json.dumps({
                    "error": f"Queued call to '{tool_name}' timed out waiting "
                             f"for Unity to reconnect.",
                    "state": self._state.value,
                })

        # STATE: STARTING - try to connect first
        if self._state == State.STARTING:
            if not self.connected:
                await self.ensure_connected()
            if not self.connected:
                return json.dumps({
                    "error": "Unity not yet connected",
                    "message": "Waiting for Unity Editor connection...",
                    "state": "starting",
                })

        # Python-side screenshot: capture Unity window via Win32 PrintWindow
        # (works even when Unity is backgrounded - no main thread dispatch needed)
        if tool_name == "screenshot_editor" and sys.platform == "win32":
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    None, _capture_screenshot_editor, arguments
                )
                return result
            except Exception as e:
                logger.warning(f"Python screenshot failed ({e}), falling back to Unity")
                # Fall through to Unity-side capture

        # STATE: READY - forward to Unity
        if not self.connected:
            await self.ensure_connected()

        if not self.connected:
            return json.dumps({
                "error": "Unity not connected. Is Unity running with the MCP server?",
                "state": self._state.value,
            })

        if not self.circuit_breaker.can_execute():
            return json.dumps({
                "error": "Circuit breaker open",
                "message": "Unity calls failing repeatedly. Waiting for recovery.",
                "retry_after": self.circuit_breaker.reset_timeout,
            })

        try:
            result = await self._execute_tool_call(tool_name, arguments)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            return json.dumps({"error": str(e)})


def get_cache_path(cache_dir: Path | None = None) -> Path:
    if cache_dir:
        return cache_dir / CACHE_FILENAME
    return Path(__file__).parent / CACHE_FILENAME


def save_schema_cache(tools: list[dict], schema_version: str,
                      cache_dir: Path | None = None):
    cache_path = get_cache_path(cache_dir)
    cache_data = {
        "schema_version": schema_version,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tools": tools,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(f"Schema cache saved: {cache_path} ({len(tools)} tools)")
    except Exception as e:
        logger.warning(f"Failed to save schema cache: {e}")


def load_schema_cache(cache_dir: Path | None = None) -> list[dict] | None:
    cache_path = get_cache_path(cache_dir)
    if not cache_path.exists():
        return None

    try:
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
        tools = cache_data.get("tools", [])
        cached_at = cache_data.get("cached_at", "unknown")
        version = cache_data.get("schema_version", "unknown")
        logger.info(
            f"Loaded {len(tools)} tools from cache "
            f"(v{version}, cached {cached_at})"
        )
        return tools
    except Exception as e:
        logger.warning(f"Failed to load schema cache: {e}")
        return None


def _convert_image_response(result: str):
    """Detect _image key in Unity JSON response and convert to MCP Image.

    Unity screenshot tools return JSON with _image (base64) and _mimeType.
    This converts them into FastMCP Image objects so the MCP protocol
    delivers proper ImageContent blocks that Claude can see.
    """
    try:
        data = json.loads(result)
        if isinstance(data, dict) and "_image" in data:
            img_b64 = data["_image"]
            mime = data.get("_mimeType", "image/png")
            img_bytes = base64.b64decode(img_b64)
            # Metadata without the large image blob
            meta = {k: v for k, v in data.items()
                    if k not in ("_image", "_mimeType")}
            return [
                Image(data=img_bytes, format=mime.split("/")[-1]),
                json.dumps(meta, indent=2),
            ]
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return result


def register_tools(mcp_server: FastMCP, tools: list[dict],
                   unity_conn: UnityConnection,
                   registered_names: set[str] | None = None,
                   ensure_watchdog=None):
    """Register discovered tools with FastMCP server.

    Args:
        mcp_server: FastMCP server instance
        tools: List of tool schema dicts from Unity
        unity_conn: Unity connection for tool execution
        registered_names: Set of already-registered tool names (skip duplicates)
        ensure_watchdog: Optional async callback to ensure watchdog is running
    """
    if registered_names is None:
        registered_names = set()

    count = 0
    for tool_schema in tools:
        name = tool_schema.get("name", "")
        description = tool_schema.get("description", f"Unity tool: {name}")
        input_schema = tool_schema.get("inputSchema", {})

        if not name or name in registered_names:
            continue

        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))

        def make_handler(tool_name: str, tool_props: dict, tool_required: set):
            async def handler(**kwargs):
                if ensure_watchdog:
                    await ensure_watchdog()
                # Claude Code MCP proxy wraps all params into a single "kwargs" string.
                # Unwrap it back into individual arguments for Unity.
                if "kwargs" in kwargs and len(kwargs) == 1 and isinstance(kwargs["kwargs"], str):
                    raw = kwargs["kwargs"]
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            kwargs = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
                result = await unity_conn.call_tool(tool_name, kwargs)
                return _convert_image_response(result)

            handler.__name__ = tool_name
            handler.__qualname__ = tool_name
            handler.__doc__ = description

            annotations = {}
            for param_name, param_info in tool_props.items():
                json_type = param_info.get("type", "string")
                type_map = {
                    "string": str,
                    "number": float,
                    "integer": int,
                    "boolean": bool,
                }
                py_type = type_map.get(json_type, str)

                if param_name not in tool_required:
                    py_type = py_type | None

                annotations[param_name] = py_type

            handler.__annotations__ = annotations
            return handler

        fn = make_handler(name, properties, required)
        try:
            mcp_server.add_tool(fn, name=name, description=description)
            registered_names.add(name)
            count += 1
        except Exception as e:
            logger.error(f"Failed to register tool '{name}': {e}")

    if count > 0:
        logger.info(f"Registered {count} Unity tools with FastMCP")


def create_server(
    ws_host: str = DEFAULT_WS_HOST,
    ws_port: int = DEFAULT_WS_PORT,
    ws_path: str = DEFAULT_WS_PATH,
    auth_token: str | None = None,
    cache_dir: Path | None = None,
    http_host: str = "0.0.0.0",
    http_port: int = 8000,
) -> tuple[FastMCP, "UnityConnection", set[str], list]:
    """Create and configure the FastMCP server.

    Pre-loads cached tools so they are available immediately when
    Claude Code starts, even before Unity is connected.

    Returns (mcp_server, unity, registered_names, watchdog_holder).
    Caller should set watchdog_holder[0] to an async ensure_watchdog callback.
    """
    mcp_server = FastMCP("realvirtual", host=http_host, port=http_port)

    cached_tools = load_schema_cache(cache_dir)
    unity = UnityConnection(ws_host, ws_port, ws_path, auth_token)
    registered_names: set[str] = set()
    # Mutable holder for watchdog callback (set by main after create_server)
    watchdog_holder: list = [None]

    async def _trigger_watchdog():
        if watchdog_holder[0]:
            await watchdog_holder[0]()

    if cached_tools:
        unity._tools = cached_tools
        register_tools(mcp_server, cached_tools, unity, registered_names,
                       ensure_watchdog=_trigger_watchdog)
        logger.info(f"Pre-loaded {len(registered_names)} tools from cache")

    # --- Built-in management tools ---

    async def unity_status() -> str:
        """Get realvirtual Unity connection status and available tools."""
        await _trigger_watchdog()
        # Auto-connect if not connected
        if not unity.connected:
            await unity.ensure_connected()
        status = {
            "connected": unity.connected,
            "ws_url": unity.ws_url,
            "tools_count": len(unity.tools),
            "state": unity.state.value,
            "buffered_messages": unity.buffer.size,
            "circuit_breaker": unity.circuit_breaker.state,
        }
        if unity.connected:
            hb = await unity.heartbeat()
            if hb:
                status["heartbeat"] = hb
        return json.dumps(status, indent=2)

    unity_status.__name__ = "unity_status"
    mcp_server.add_tool(
        unity_status,
        name="unity_status",
        description="Get realvirtual Unity connection status and available tools",
    )

    async def unity_reconnect() -> str:
        """Reconnect to Unity simulation and re-discover tools."""
        await _trigger_watchdog()
        await unity.disconnect()
        # Reset error state so reconnect attempts are allowed
        unity._reconnect_attempts = 0
        if unity.state == State.ERROR:
            unity._set_state(State.RECONNECTING)
        success = await unity.connect()
        if success and unity.tools:
            save_schema_cache(unity.tools, unity._schema_version, cache_dir)
            # Clear and re-register all tools to pick up new ones after recompile
            old_count = len(registered_names)
            registered_names.clear()
            register_tools(mcp_server, unity.tools, unity, registered_names)
            new_tools = len(registered_names) - old_count
            result = {
                "status": "reconnected",
                "tools_count": len(unity.tools),
            }
            if new_tools > 0:
                result["new_tools_registered"] = new_tools
            return json.dumps(result)
        return json.dumps({
            "status": "failed",
            "error": f"Cannot connect to {unity.ws_url}",
        })

    unity_reconnect.__name__ = "unity_reconnect"
    mcp_server.add_tool(
        unity_reconnect,
        name="unity_reconnect",
        description="Reconnect to Unity simulation and re-discover tools",
    )

    async def editor_wait_ready(timeout: float | None = None) -> str:
        """Wait until Unity Editor is ready (done compiling/importing).

        Polls the connection state and returns as soon as Unity is connected
        and responsive. Use after editor_recompile or editor_refresh_assets
        instead of blind sleep. Returns immediately if already ready.
        """
        if timeout is None:
            timeout = 60.0
        await _trigger_watchdog()
        start = time.monotonic()
        poll_interval = 0.5  # fast polling for responsiveness

        while True:
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                return json.dumps({
                    "status": "timeout",
                    "error": f"Unity not ready after {timeout:.0f}s",
                    "state": unity.state.value,
                    "elapsed": round(elapsed, 1),
                })

            # If connected, verify with heartbeat
            if unity.connected:
                hb = await unity.heartbeat()
                if hb and hb.get("status") == "ok":
                    # Also check if Unity is still compiling/importing
                    try:
                        status_result = await unity._execute_tool_call(
                            "editor_get_status", {})
                        status_data = json.loads(status_result)
                        is_playing = status_data.get("isPlaying", False)
                        if is_playing:
                            return json.dumps({
                                "status": "playing",
                                "error": "Unity is in play mode. Stop the simulation first (sim_stop) before recompiling or waiting for ready.",
                                "state": unity.state.value,
                                "isPlaying": True,
                                "waited": round(elapsed, 1),
                            })
                        is_busy = (status_data.get("isCompiling", False)
                                   or status_data.get("isUpdating", False))
                        if not is_busy:
                            return json.dumps({
                                "status": "ready",
                                "state": unity.state.value,
                                "tools_count": len(unity.tools),
                                "waited": round(elapsed, 1),
                            })
                        # Still compiling, keep polling
                        logger.debug(
                            f"editor_wait_ready: Unity busy "
                            f"(compiling={status_data.get('isCompiling')}, "
                            f"updating={status_data.get('isUpdating')}), "
                            f"elapsed={elapsed:.1f}s"
                        )
                    except Exception:
                        pass  # Fall through to retry

            # Not ready yet, try reconnecting if disconnected
            if not unity.connected:
                unity._reconnect_attempts = 0
                if unity.state == State.ERROR:
                    unity._set_state(State.RECONNECTING)
                await unity.ensure_connected()

            await asyncio.sleep(poll_interval)

    editor_wait_ready.__name__ = "editor_wait_ready"
    mcp_server.add_tool(
        editor_wait_ready,
        name="editor_wait_ready",
        description=(
            "Wait until Unity Editor is ready (done compiling/importing). "
            "Use after editor_recompile or editor_refresh_assets instead of sleeping."
        ),
    )

    # screenshot_editor is intercepted Python-side in call_tool() using Win32
    # PrintWindow API for reliable capture even when Unity is backgrounded.
    # Falls back to the C# implementation if Python capture fails.

    return mcp_server, unity, registered_names, watchdog_holder


def _capture_screenshot_editor(arguments: dict) -> str:
    """Python-side screenshot_editor: captures Unity window via Win32 PrintWindow.

    Called from a thread executor to avoid blocking the async event loop.
    Returns JSON string matching the C# ScreenshotEditor response format.
    """
    save_path = arguments.get("save_path", "")

    png_bytes, width, height = _capture_unity_window()
    base64_str = base64.b64encode(png_bytes).decode("ascii")

    # Save to file
    saved_to = None
    if save_path:
        try:
            parent = os.path.dirname(save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(png_bytes)
            saved_to = save_path
        except Exception as e:
            logger.warning(f"Failed to save screenshot to '{save_path}': {e}")

    if saved_to is None:
        # Save to default .screenshots/ directory
        screenshots_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", ".screenshots"
        )
        os.makedirs(screenshots_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"editor_{ts}.png"
        saved_to = os.path.join(screenshots_dir, filename)
        try:
            with open(saved_to, "wb") as f:
                f.write(png_bytes)
        except Exception:
            saved_to = None

    return json.dumps({
        "status": "ok",
        "_image": base64_str,
        "_mimeType": "image/png",
        "width": width,
        "height": height,
        "panel": "editor",
        "format": "png",
        "savedTo": saved_to,
        "source": "python-printwindow",
    })


def _capture_unity_window() -> tuple[bytes, int, int]:
    """Capture Unity Editor window using Windows GDI and return PNG bytes.

    Uses PrintWindow for reliable capture even when window is partially obscured.
    Returns (png_bytes, width, height).
    """
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    # Find Unity Editor window
    hwnd = user32.FindWindowW("UnityContainerWndClass", None)
    if not hwnd:
        raise RuntimeError(
            "Unity Editor window not found. "
            "Is Unity running?"
        )

    # Get window dimensions
    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    width = rect.right - rect.left
    height = rect.bottom - rect.top

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid window dimensions: {width}x{height}")

    # Create device contexts and bitmap
    hdc_window = user32.GetDC(hwnd)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
    hbitmap = gdi32.CreateCompatibleBitmap(hdc_window, width, height)
    old_bitmap = gdi32.SelectObject(hdc_mem, hbitmap)

    # PrintWindow captures even partially obscured windows
    PW_RENDERFULLCONTENT = 0x00000002
    user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)

    # Read pixel data via GetDIBits
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32),
            ("biWidth", ctypes.c_int32),
            ("biHeight", ctypes.c_int32),
            ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16),
            ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_int32),
            ("biYPelsPerMeter", ctypes.c_int32),
            ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = width
    bmi.biHeight = -height  # negative = top-down DIB
    bmi.biPlanes = 1
    bmi.biBitCount = 32  # BGRA
    bmi.biCompression = 0  # BI_RGB

    stride = width * 4
    buf_size = stride * height
    pixel_buf = ctypes.create_string_buffer(buf_size)

    BI_RGB = 0
    DIB_RGB_COLORS = 0
    gdi32.GetDIBits(
        hdc_mem, hbitmap, 0, height,
        pixel_buf, ctypes.byref(bmi), DIB_RGB_COLORS
    )

    # Cleanup GDI resources
    gdi32.SelectObject(hdc_mem, old_bitmap)
    gdi32.DeleteObject(hbitmap)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(hwnd, hdc_window)

    # Convert BGRA pixel data to PNG
    raw_pixels = pixel_buf.raw
    png_bytes = _encode_png(raw_pixels, width, height)

    return png_bytes, width, height


def _encode_png(bgra_data: bytes, width: int, height: int) -> bytes:
    """Minimal PNG encoder: converts BGRA pixel data to PNG format.

    Uses zlib compression from stdlib. Produces valid PNG with RGB channels.
    """
    # Build raw scanlines: filter byte (0=None) + RGB pixels per row
    raw_lines = bytearray()
    stride = width * 4
    for y in range(height):
        raw_lines.append(0)  # filter type: None
        row_offset = y * stride
        for x in range(width):
            px = row_offset + x * 4
            # BGRA -> RGB
            raw_lines.append(bgra_data[px + 2])  # R
            raw_lines.append(bgra_data[px + 1])  # G
            raw_lines.append(bgra_data[px + 0])  # B

    compressed = zlib.compress(bytes(raw_lines), 6)

    # Build PNG file
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    png = b"\x89PNG\r\n\x1a\n"  # PNG signature

    # IHDR: width, height, bit depth 8, color type 2 (RGB)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png += _chunk(b"IHDR", ihdr_data)

    # IDAT: compressed image data
    png += _chunk(b"IDAT", compressed)

    # IEND
    png += _chunk(b"IEND", b"")

    return png


async def run_watchdog(unity: UnityConnection, cache_dir: Path | None,
                       mcp_server: FastMCP, registered_names: set[str],
                       project_path: str | None = None):
    """Background watchdog that keeps the Unity connection alive.

    - Connects on startup if not yet connected
    - Periodically heartbeats to detect disconnections
    - Auto-reconnects with backoff after Unity domain reloads
    - Re-discovers port from status files after domain reload
    - Registers newly discovered tools after reconnect
    - Flushes message buffer after successful reconnect
    - Detects close reasons for state machine transitions
    """
    delay = RECONNECT_BASE_DELAY

    while True:
        try:
            # Exit cleanly when Unity requests shutdown (MCP client will restart us)
            if unity.state == State.SHUTDOWN:
                logger.info("Watchdog: Unity requested shutdown - exiting process")
                unity.buffer.reject_all("Server shutting down")
                sys.exit(0)

            if not unity.connected:
                was_reloading = unity.state == State.RELOADING

                # Re-discover port in case Unity restarted on a different port
                discovered_port = discover_unity_port(project_path)
                if discovered_port and discovered_port != unity.port:
                    logger.info(
                        f"Watchdog: Unity port changed "
                        f"{unity.port} -> {discovered_port}"
                    )
                    unity.port = discovered_port

                success = await unity.connect()
                if success:
                    logger.info(
                        f"Watchdog: connected to Unity "
                        f"(was {'reloading' if was_reloading else 'disconnected'})"
                    )
                    save_schema_cache(unity.tools, unity._schema_version, cache_dir)
                    # Clear registered names after domain reload so new tools get registered
                    if was_reloading:
                        old_count = len(registered_names)
                        registered_names.clear()
                        logger.info(f"Watchdog: cleared {old_count} registered tools for re-registration after reload")
                    register_tools(mcp_server, unity.tools, unity, registered_names)
                    delay = RECONNECT_BASE_DELAY

                    # Flush buffered messages after reconnect
                    if unity.buffer.size > 0:
                        logger.info(
                            f"Watchdog: flushing {unity.buffer.size} "
                            f"buffered messages"
                        )
                        replayed, discarded = await unity.buffer.flush(
                            unity._execute_tool_call
                        )
                        logger.info(
                            f"Watchdog: buffer flush complete "
                            f"(replayed={replayed}, discarded={discarded})"
                        )
                else:
                    delay = min(delay * RECONNECT_MULTIPLIER, RECONNECT_MAX_DELAY)
                    # Use shorter delay during RELOADING (expected quick recovery)
                    if unity.state == State.RELOADING:
                        delay = min(delay, 2.0)
                    logger.debug(
                        f"Watchdog: Unity not available "
                        f"(state={unity.state.value}), retry in {delay:.0f}s"
                    )
            else:
                # Heartbeat to verify connection is alive
                hb = await unity.heartbeat()
                if hb is None:
                    logger.warning("Watchdog: heartbeat failed, marking disconnected")
                    unity._connected = False
                    # Check close reason on the websocket if available
                    if unity.ws is not None:
                        try:
                            close_reason = getattr(unity.ws, "close_reason", "") or ""
                            if "shutdown" in close_reason:
                                unity._set_state(State.SHUTDOWN)
                            elif "domain_reload" in close_reason:
                                unity._set_state(State.RELOADING)
                            elif unity.state == State.READY:
                                unity._set_state(State.RECONNECTING)
                        except Exception:
                            if unity.state == State.READY:
                                unity._set_state(State.RECONNECTING)
                    elif unity.state == State.READY:
                        unity._set_state(State.RECONNECTING)
                    delay = RECONNECT_BASE_DELAY
                    continue
                delay = RECONNECT_BASE_DELAY
        except Exception as e:
            logger.debug(f"Watchdog error: {e}")
            delay = min(delay * RECONNECT_MULTIPLIER, RECONNECT_MAX_DELAY)

        await asyncio.sleep(delay if not unity.connected else WATCHDOG_INTERVAL)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="realvirtual Unity MCP Server - Bridge to Unity Digital Twin"
    )
    parser.add_argument(
        "--mode", choices=["stdio", "sse"], default="stdio",
        help="Transport mode (default: stdio)"
    )
    parser.add_argument(
        "--ws-host", default=DEFAULT_WS_HOST,
        help=f"Unity WebSocket host (default: {DEFAULT_WS_HOST})"
    )
    parser.add_argument(
        "--ws-port", type=int, default=None,
        help=f"Unity WebSocket port (default: auto-discover, fallback {DEFAULT_WS_PORT})"
    )
    parser.add_argument(
        "--http-port", type=int, default=DEFAULT_HTTP_PORT,
        help=f"HTTP/SSE port (default: {DEFAULT_HTTP_PORT}, only for --mode sse)"
    )
    parser.add_argument(
        "--auth-token", default=None,
        help="Authentication token for Unity connection"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory for schema cache file (default: script directory)"
    )
    parser.add_argument(
        "--project-path", default=None,
        help="Unity project Assets path for multi-instance discovery"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    log_fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

    # File-based debug logging (always at DEBUG level, deleted on each start)
    # Use system temp dir - works in Editor and builds, never triggers Unity AssetDatabase
    log_dir = Path(tempfile.gettempdir()) / "realvirtual-mcp"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "mcp_debug.log"
    try:
        if log_file.exists():
            log_file.unlink()
    except Exception:
        pass
    # Clean up stale log from old location (inside Assets/, causes Unity import errors)
    script_dir = Path(__file__).resolve().parent
    stale_log = script_dir / "mcp_debug.log"
    try:
        if stale_log.exists():
            stale_log.unlink()
    except Exception:
        pass

    handlers = [logging.StreamHandler(sys.stderr)]
    handlers[0].setLevel(level)
    handlers[0].setFormatter(logging.Formatter(log_fmt))

    try:
        file_handler = logging.FileHandler(str(log_file), mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_fmt))
        handlers.append(file_handler)
    except Exception:
        pass  # File logging not available

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    # Resolve WebSocket port: explicit arg > discovery > default
    ws_port = args.ws_port
    if ws_port is None:
        discovered = discover_unity_port(args.project_path)
        if discovered:
            ws_port = discovered
            logger.info(f"Auto-discovered Unity WebSocket port: {ws_port}")
        else:
            ws_port = DEFAULT_WS_PORT
            logger.info(
                f"No Unity instance discovered, using default port {ws_port}"
            )

    if not 1024 <= ws_port <= 65535:
        logger.error(f"Invalid WebSocket port: {ws_port}")
        sys.exit(1)

    mcp_server, unity, registered_names, watchdog_holder = create_server(
        ws_host=args.ws_host,
        ws_port=ws_port,
        auth_token=args.auth_token,
        cache_dir=args.cache_dir,
        http_port=args.http_port,
    )

    # Watchdog startup with lazy fallback via tool calls
    _watchdog_started = False
    _project_path = args.project_path

    async def _ensure_watchdog():
        nonlocal _watchdog_started
        if _watchdog_started:
            return
        _watchdog_started = True
        logger.info("Starting connection watchdog")
        asyncio.ensure_future(
            run_watchdog(unity, args.cache_dir, mcp_server, registered_names,
                         project_path=_project_path)
        )

    # Wire up the watchdog callback into the holder from create_server
    watchdog_holder[0] = _ensure_watchdog

    # Hook watchdog into FastMCP lifecycle (primary start mechanism)
    try:
        async def _init_handler(notification):
            await _ensure_watchdog()
        mcp_server._mcp_server.notification_handlers[InitializedNotification] = _init_handler
        logger.debug("Watchdog hooked into MCP initialized notification")
    except Exception as e:
        logger.warning(f"Could not hook watchdog into MCP lifecycle: {e}")

    logger.info(
        f"Starting realvirtual MCP Server (mode={args.mode}, "
        f"ws={args.ws_host}:{ws_port})"
    )

    if args.mode == "sse":
        mcp_server.run(transport="sse")
    else:
        mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
