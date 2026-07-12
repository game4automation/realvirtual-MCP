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

__version__ = "1.1.0"

import argparse
import asyncio
import base64
import ctypes
import ctypes.wintypes
import enum
import json
import logging
import os
import re
import struct
import subprocess
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

# WebViewer bridge config
DEFAULT_WV_PORT = 18712
DEFAULT_WV_PATH = "/webviewer"

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

# Main thread stall threshold (seconds). Above this, the Unity editor main
# thread is reported as blocked. The WebSocket heartbeat alone stays alive
# during a freeze because it is answered on a background thread.
MAIN_THREAD_STALL_THRESHOLD = 5.0

# Timeout for PowerShell process queries (seconds)
PROCESS_QUERY_TIMEOUT = 20.0

# How long unity_restart waits for killed Unity PIDs to disappear (seconds)
KILL_WAIT_TIMEOUT = 15.0

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


# ---------------------------------------------------------------------------
# Unity process control (unity_kill / unity_restart)
#
# These helpers are pure Python/OS-side and MUST work when the Unity editor
# is frozen or dead - no Unity WebSocket roundtrip is allowed here.
# ---------------------------------------------------------------------------

def _norm_path(p: str) -> str:
    """Normalize a filesystem path for case-insensitive, separator-tolerant comparison."""
    return str(p).replace("\\", "/").rstrip("/").lower()


def _get_project_root() -> Path:
    """Unity project root derived from this script's location.

    The server lives at <project>/Assets/StreamingAssets/realvirtual-MCP/,
    so the project root is three directory levels up.
    """
    return Path(__file__).resolve().parents[3]


def _extract_projectpath(cmdline: str) -> str | None:
    """Extract the -projectpath/-projectPath argument value from a command line."""
    if not cmdline:
        return None
    m = re.search(
        r'-projectpath\s+(?:"([^"]+)"|\'([^\']+)\'|([^\s"]+))',
        cmdline, re.IGNORECASE)
    if not m:
        return None
    return m.group(1) or m.group(2) or m.group(3)


def _query_unity_processes() -> list[dict]:
    """List all running Unity.exe processes with PID, exe path and command line.

    Uses PowerShell Get-CimInstance (Win32_Process). Works without Unity.
    Returns [] on non-Windows platforms or query failure.
    """
    if sys.platform != "win32":
        return []
    ps_script = (
        "Get-CimInstance Win32_Process -Filter \"Name='Unity.exe'\" | "
        "Select-Object ProcessId,ExecutablePath,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True, text=True, timeout=PROCESS_QUERY_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"Unity process query failed: {e}")
        return []

    out = (proc.stdout or "").strip()
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):  # single process -> object, not array
        data = [data]

    result = []
    for entry in data:
        pid = entry.get("ProcessId")
        if not pid:
            continue
        result.append({
            "pid": int(pid),
            "exe": entry.get("ExecutablePath") or "",
            "cmdline": entry.get("CommandLine") or "",
        })
    return result


def _find_project_unity_processes(project_root: Path) -> list[dict]:
    """Find Unity.exe processes (editor + asset import workers) whose -projectpath
    matches THIS project. Unity instances of other projects are never returned.
    """
    target = _norm_path(str(project_root))
    matches = []
    for proc in _query_unity_processes():
        proj = _extract_projectpath(proc["cmdline"])
        if proj and _norm_path(proj) == target:
            matches.append(proc)
    return matches


def _pid_alive(pid: int) -> bool:
    """Check whether a process is still running.

    Windows-safe: uses OpenProcess/GetExitCodeProcess. Never use os.kill(pid, 0)
    on Windows - it TERMINATES the target process instead of probing it.
    """
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        exit_code = ctypes.wintypes.DWORD()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return exit_code.value == STILL_ACTIVE
        return True
    finally:
        kernel32.CloseHandle(handle)


def _kill_unity_processes(project_root: Path) -> dict:
    """Force-kill all Unity.exe processes belonging to THIS project (blocking).

    Process-selective: only PIDs whose command line -projectpath matches
    project_root are killed. Never touches this Python process or Unity
    instances of other projects.

    Returns {"killed": [{pid, exe, cmdline}], "exe": <remembered editor exe>}.
    """
    procs = _find_project_unity_processes(project_root)
    if not procs:
        return {"killed": [], "exe": None}

    remembered_exe = None
    for p in procs:
        if p["exe"] and p["exe"].lower().endswith("unity.exe"):
            remembered_exe = p["exe"]
            break

    own_pid = os.getpid()
    pids = [p["pid"] for p in procs if p["pid"] != own_pid]
    if pids:
        pid_list = ",".join(str(p) for p in pids)
        logger.info(f"Killing Unity processes for {project_root}: {pid_list}")
        try:
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                 f"Stop-Process -Id {pid_list} -Force -ErrorAction SilentlyContinue"],
                capture_output=True, text=True, timeout=PROCESS_QUERY_TIMEOUT)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(f"Stop-Process failed: {e}")

    return {
        "killed": [{"pid": p["pid"], "exe": p["exe"]} for p in procs],
        "exe": remembered_exe,
    }


def _find_unity_exe(project_root: Path) -> str | None:
    """Locate Unity.exe for this project via Unity Hub installations.

    Reads ProjectSettings/ProjectVersion.txt (m_EditorVersion) and prefers the
    exact matching Unity Hub installation; falls back to the newest installed
    editor under C:/Program Files/Unity/Hub/Editor.
    """
    editor_version = None
    version_file = project_root / "ProjectSettings" / "ProjectVersion.txt"
    try:
        for line in version_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("m_EditorVersion:"):
                editor_version = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    hub_dir = Path("C:/Program Files/Unity/Hub/Editor")
    if not hub_dir.exists():
        return None

    # Exact version match first
    if editor_version:
        exact = hub_dir / editor_version / "Editor" / "Unity.exe"
        if exact.exists():
            return str(exact)

    # Fallback: newest installed editor (by folder modification time)
    candidates = []
    try:
        for d in hub_dir.iterdir():
            exe = d / "Editor" / "Unity.exe"
            if exe.exists():
                candidates.append((d.stat().st_mtime, str(exe)))
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _start_unity(exe: str, project_root: Path) -> int:
    """Start Unity detached with -projectpath (no waiting). Returns the new PID."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        [exe, "-projectpath", str(project_root)],
        creationflags=flags,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    return proc.pid


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


class WebViewerBridge:
    """WebSocket server for browser MCP tool connections.

    The browser's McpBridgePlugin connects as a WebSocket client and:
    - Sends a 'discover' message with tool schemas and instructions on connect
    - Receives 'call' messages from Python (forwarded MCP tool calls)
    - Returns 'result' messages with tool execution results

    Protocol:
      Browser -> Python: { type: 'discover', tools: [...], instructions: '...', schema_version: '1.0.0' }
      Python -> Browser: { type: 'call', id: N, tool: 'web_drive_list', arguments: {} }
      Browser -> Python: { type: 'result', id: N, result: '...' }
      Browser -> Python: { type: 'result', id: N, error: '...' }
    """

    def __init__(self, host: str = DEFAULT_WS_HOST, port: int = DEFAULT_WV_PORT,
                 mcp_server: Any = None):
        self.host = host
        self.port = port
        self._mcp_server = mcp_server
        self._wv_registered_names: set[str] = set()  # Separate from Unity registered_names
        self._browser_ws: Any = None
        self._connected = False
        self._cmd_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._tools: list[dict] = []
        self._instructions: str = ""
        self._server: Any = None

    @property
    def connected(self) -> bool:
        return self._connected and self._browser_ws is not None

    async def start(self):
        """Start the WebSocket server for browser connections."""
        if not HAS_WEBSOCKETS:
            logger.warning("WebViewer bridge: websockets package not installed")
            return

        try:
            self._server = await websockets.serve(
                self._handle_browser,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=20,
            )
            logger.info(
                f"WebViewer bridge listening on "
                f"ws://{self.host}:{self.port}/webviewer"
            )
        except OSError as e:
            logger.warning(f"WebViewer bridge: cannot bind port {self.port}: {e}")

    async def stop(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._connected = False
        self._browser_ws = None

    async def _handle_browser(self, websocket, path=None):
        """Handle a single browser WebSocket connection."""
        remote = getattr(websocket, 'remote_address', ('?', '?'))
        logger.info(f"WebViewer browser connected from {remote}")

        # Close old browser connection when new one arrives (two-tab scenario)
        if self._browser_ws is not None and self._browser_ws is not websocket:
            logger.info("Closing previous WebViewer browser connection")
            try:
                await self._browser_ws.close(code=1008, reason="Another tab connected")
            except Exception:
                pass
            self._reject_all_pending()

        self._browser_ws = websocket
        self._connected = True

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "")

                    if msg_type == "discover":
                        await self._handle_discover(data)

                    elif msg_type == "result":
                        self._resolve_pending(data)

                except Exception as e:
                    logger.warning(f"WebViewer: error handling message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebViewer browser disconnected")
        finally:
            self._connected = False
            self._browser_ws = None
            self._reject_all_pending()
            # Clear web_* tool names on disconnect (do NOT touch Unity registered_names)
            self._wv_registered_names.clear()
            logger.info(f"WebViewer tools cleared on disconnect")

    async def _handle_discover(self, data: dict):
        """Handle discover message from browser: register web_* tools with FastMCP."""
        tools = data.get("tools", [])
        instructions = data.get("instructions", "")
        schema_version = data.get("schema_version", "unknown")

        self._tools = tools
        self._instructions = instructions

        logger.info(
            f"WebViewer discover: {len(tools)} tools, "
            f"schema_version={schema_version}, "
            f"instructions={len(instructions)} chars"
        )

        # Clear old web_* tools before re-registering
        self._wv_registered_names.clear()

        # Register browser tools with FastMCP
        self._register_web_tools(tools)

        # Combine WV instructions with existing MCP instructions
        if instructions and self._mcp_server:
            try:
                existing = getattr(self._mcp_server._mcp_server, 'instructions', '') or ''
                # Append WV instructions if not already present
                if instructions not in existing:
                    combined = f"{existing}\n\n{instructions}" if existing else instructions
                    self._mcp_server._mcp_server.instructions = combined
                    logger.info(f"Updated MCP instructions with WebViewer context")
            except Exception as e:
                logger.debug(f"Could not update MCP instructions: {e}")

        # Notify MCP client that tool list changed
        if self._mcp_server and hasattr(self._mcp_server, '_notify_tools_changed'):
            try:
                await self._mcp_server._notify_tools_changed()
                logger.info("Sent tools/list_changed after WebViewer discover")
            except Exception as e:
                logger.debug(f"Could not notify tools changed: {e}")

    def _register_web_tools(self, tools: list[dict]):
        """Register browser-defined tools with FastMCP.

        Similar to register_tools() but uses _wv_registered_names and
        routes calls to the browser via call_tool() instead of Unity.
        """
        if not self._mcp_server:
            logger.warning("WebViewer bridge: no MCP server to register tools with")
            return

        count = 0
        for tool_schema in tools:
            name = tool_schema.get("name", "")
            description = tool_schema.get("description", f"WebViewer tool: {name}")
            input_schema = tool_schema.get("inputSchema", {})

            if not name or name in self._wv_registered_names:
                continue

            properties = input_schema.get("properties", {})
            required = set(input_schema.get("required", []))

            def make_handler(tool_name: str, tool_props: dict, tool_required: set):
                async def handler(**kwargs):
                    # Claude Code MCP proxy wraps all params into a single "kwargs" string.
                    # Unwrap it back into individual arguments for the browser.
                    if "kwargs" in kwargs and len(kwargs) == 1 and isinstance(kwargs["kwargs"], str):
                        raw = kwargs["kwargs"]
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict):
                                kwargs = parsed
                        except (json.JSONDecodeError, TypeError):
                            pass
                    result = await self.call_tool(tool_name, kwargs)
                    return result

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
                self._mcp_server.add_tool(fn, name=name, description=description)
                self._wv_registered_names.add(name)
                count += 1
            except Exception as e:
                logger.error(f"Failed to register WebViewer tool '{name}': {e}")

        if count > 0:
            logger.info(f"Registered {count} WebViewer tools with FastMCP")

    async def call_tool(self, tool_name: str, arguments: dict,
                        timeout: float = 5.0) -> str:
        """Forward tool call to browser and wait for result.

        Returns JSON string (result from browser or error message).
        """
        if not self._connected or not self._browser_ws:
            return json.dumps({"error": "WebViewer not connected"})

        self._cmd_id += 1
        cmd_id = self._cmd_id

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[cmd_id] = future

        try:
            await self._browser_ws.send(json.dumps({
                "type": "call",
                "id": cmd_id,
                "tool": tool_name,
                "arguments": arguments,
            }))
        except Exception as e:
            self._pending.pop(cmd_id, None)
            return json.dumps({"error": f"WebSocket send failed: {e}"})

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            # Browser sends { type: 'result', id, result: '...' } or { ..., error: '...' }
            if "error" in result:
                return json.dumps({"error": result["error"]})
            return result.get("result", json.dumps({"error": "No result from browser"}))
        except asyncio.TimeoutError:
            self._pending.pop(cmd_id, None)
            return json.dumps({"error": "Browser tool call timeout"})

    def _resolve_pending(self, data: dict):
        """Resolve a pending future from a browser result message."""
        cmd_id = data.get("id")
        future = self._pending.pop(cmd_id, None)
        if future and not future.done():
            future.set_result(data)

    def _reject_all_pending(self):
        """Reject all pending futures (e.g. on disconnect)."""
        for cmd_id, future in self._pending.items():
            if not future.done():
                future.set_result({"error": "Browser disconnected"})
        self._pending.clear()


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
        self._instructions: str = ""
        self._schema_version: str = ""
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._lock = asyncio.Lock()
        self._last_comm_time: float = 0
        # Sidecar relay state machine
        self._state = State.STARTING
        self._reconnect_attempts = 0
        self.buffer = MessageBuffer()
        self.circuit_breaker = CircuitBreaker()
        # Main-thread liveness: reported by the Unity heartbeat
        # (main_thread_inactive_s field) or derived from dispatch error texts.
        # None = unknown (e.g. old Unity package without the heartbeat field).
        self._main_thread_inactive_s: float | None = None
        self._main_thread_report_time: float = 0.0

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

    @property
    def main_thread_inactive_s(self) -> float | None:
        """Last known Unity main-thread pump inactivity in seconds (None = unknown)."""
        return self._main_thread_inactive_s

    def _note_main_thread(self, inactive_s: float):
        """Record a main-thread liveness observation."""
        self._main_thread_inactive_s = inactive_s
        self._main_thread_report_time = time.monotonic()

    def _track_main_thread_from_error(self, error_text) -> None:
        """Derive main-thread inactivity from Unity dispatch error messages.

        The C# bridge fails fast with 'Main thread inactive for Xs' when the
        pump is stalled, and with 'Main thread dispatch timeout' after 30s.
        """
        if not isinstance(error_text, str):
            return
        m = re.search(r"[Mm]ain thread inactive for (\d+(?:\.\d+)?)s", error_text)
        if m:
            self._note_main_thread(float(m.group(1)))
        elif "Main thread dispatch timeout" in error_text:
            self._note_main_thread(max(self._main_thread_inactive_s or 0.0, 30.0))

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
                # Parse in its own guard: a malformed tool response (e.g. a tool
                # building JSON manually with locale decimal commas) is a TOOL
                # bug, not a dead connection. Never mark the connection as
                # disconnected or trip the circuit breaker for a parse error -
                # that caused fake 'reconnecting' loops.
                try:
                    return json.loads(response)
                except (json.JSONDecodeError, ValueError) as e:
                    preview = response[:200] if isinstance(response, str) else str(response)[:200]
                    logger.warning(
                        f"Malformed JSON from Unity for {label}: {e} "
                        f"(raw: {preview!r})"
                    )
                    return {
                        "error": f"Unity returned malformed JSON for {label}: {e}",
                        "raw_preview": preview,
                    }
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
            self._instructions = response.get("instructions", "")
            self._schema_version = response.get("schema_version", "unknown")
            logger.info(
                f"Discovered {len(self._tools)} tools "
                f"(schema v{self._schema_version})"
            )
            if self._instructions:
                logger.info(f"Received MCP instructions ({len(self._instructions)} chars)")
            return True
        return False

    async def heartbeat(self) -> dict | None:
        hb = await self._send_command({"command": "__heartbeat__"})
        if hb is not None:
            # Unity reports pump inactivity in the heartbeat (answered on a
            # background thread, so this works even while the main thread hangs).
            mt = hb.get("main_thread_inactive_s")
            if isinstance(mt, (int, float)) and mt >= 0:
                self._note_main_thread(float(mt))
        return hb

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
            self._track_main_thread_from_error(response["error"])
            err = {"error": response["error"]}
            if "raw_preview" in response:
                err["raw_preview"] = response["raw_preview"]
            return json.dumps(err)

        # A successful result proves the Unity main thread just executed the call
        self._note_main_thread(0.0)

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
                      cache_dir: Path | None = None,
                      instructions: str = ""):
    cache_path = get_cache_path(cache_dir)

    # Skip the write when nothing changed (ignoring the cached_at timestamp).
    # The cache lives under Assets/StreamingAssets - every write triggers a
    # Unity asset refresh, which cascaded badly during reconnect loops.
    try:
        if cache_path.exists():
            existing = json.loads(cache_path.read_text(encoding="utf-8"))
            if (existing.get("tools") == tools
                    and existing.get("schema_version") == schema_version
                    and existing.get("instructions", "") == (instructions or "")):
                logger.debug("Schema cache unchanged, skipping write")
                return
    except Exception:
        pass  # Unreadable/corrupt cache - rewrite it below

    cache_data = {
        "schema_version": schema_version,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tools": tools,
    }
    if instructions:
        cache_data["instructions"] = instructions
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(f"Schema cache saved: {cache_path} ({len(tools)} tools)")
    except Exception as e:
        logger.warning(f"Failed to save schema cache: {e}")


def load_schema_cache(cache_dir: Path | None = None) -> tuple[list[dict] | None, str]:
    """Load cached tool schemas and instructions.

    Returns (tools, instructions) tuple. tools is None if no cache exists.
    """
    cache_path = get_cache_path(cache_dir)
    if not cache_path.exists():
        return None, ""

    try:
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
        tools = cache_data.get("tools", [])
        cached_instructions = cache_data.get("instructions", "")
        cached_at = cache_data.get("cached_at", "unknown")
        version = cache_data.get("schema_version", "unknown")
        logger.info(
            f"Loaded {len(tools)} tools from cache "
            f"(v{version}, cached {cached_at})"
        )
        return tools, cached_instructions
    except Exception as e:
        logger.warning(f"Failed to load schema cache: {e}")
        return None, ""


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
    web_port: int = DEFAULT_WV_PORT,
    no_webviewer: bool = False,
) -> tuple[FastMCP, "UnityConnection", set[str], list, "WebViewerBridge | None"]:
    """Create and configure the FastMCP server.

    Pre-loads cached tools so they are available immediately when
    Claude Code starts, even before Unity is connected.

    Returns (mcp_server, unity, registered_names, watchdog_holder, wv_bridge).
    Caller should set watchdog_holder[0] to an async ensure_watchdog callback.
    """
    mcp_server = FastMCP("realvirtual", host=http_host, port=http_port)

    cached_tools, cached_instructions = load_schema_cache(cache_dir)
    unity = UnityConnection(ws_host, ws_port, ws_path, auth_token)
    registered_names: set[str] = set()
    # Mutable holder for watchdog callback (set by main after create_server)
    watchdog_holder: list = [None]

    async def _trigger_watchdog():
        if watchdog_holder[0]:
            await watchdog_holder[0]()

    if cached_tools:
        unity._tools = cached_tools
        if cached_instructions:
            unity._instructions = cached_instructions
        register_tools(mcp_server, cached_tools, unity, registered_names,
                       ensure_watchdog=_trigger_watchdog)
        logger.info(f"Pre-loaded {len(registered_names)} tools from cache")

    # Apply instructions (from cache or live discovery)
    if cached_instructions:
        mcp_server._mcp_server.instructions = cached_instructions
        logger.info(f"Applied cached MCP instructions ({len(cached_instructions)} chars)")

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
            # Honest main-thread status: the heartbeat is answered on a Unity
            # background thread and stays green even when the main thread is
            # frozen. main_thread_inactive_s comes from the heartbeat (new
            # Unity package) or from dispatch error texts (fallback).
            mt = unity.main_thread_inactive_s
            if mt is not None:
                status["main_thread_inactive_s"] = round(mt, 1)
                status["main_thread_alive"] = mt < MAIN_THREAD_STALL_THRESHOLD
                if mt >= MAIN_THREAD_STALL_THRESHOLD:
                    status["hint"] = (
                        "Unity main thread appears blocked - if this persists, "
                        "use unity_kill or unity_restart"
                    )
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
            save_schema_cache(unity.tools, unity._schema_version, cache_dir,
                              instructions=unity._instructions)
            # Apply fresh instructions from Unity
            if unity._instructions:
                mcp_server._mcp_server.instructions = unity._instructions
            # Clear and re-register all tools to pick up new ones after recompile
            old_count = len(registered_names)
            registered_names.clear()
            register_tools(mcp_server, unity.tools, unity, registered_names)
            new_tools = len(registered_names) - old_count
            # Notify MCP client that tool list changed
            if hasattr(mcp_server, '_notify_tools_changed'):
                await mcp_server._notify_tools_changed()
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
        instead of sleeping. Returns immediately if already ready.

        The response includes a 'phases' list showing what Unity went through
        (e.g. disconnected → domain_reload → compiling → ready) with timestamps.
        """
        if timeout is None:
            timeout = 60.0
        await _trigger_watchdog()
        start = time.monotonic()
        poll_interval = 0.5  # fast polling for responsiveness

        # Track phases for clear feedback
        phases: list[dict] = []
        current_phase = None

        def _record_phase(phase_name: str) -> None:
            nonlocal current_phase
            if phase_name != current_phase:
                elapsed = round(time.monotonic() - start, 1)
                phases.append({"phase": phase_name, "at": f"{elapsed}s"})
                current_phase = phase_name
                logger.info(
                    f"editor_wait_ready: [{phase_name}] at {elapsed}s"
                )

        _record_phase("waiting")

        # Last observed main-thread inactivity while blocked (None = not blocked)
        last_blocked_s: float | None = None

        while True:
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                _record_phase("timeout")
                if last_blocked_s is not None:
                    # Honest answer: the relay/WebSocket is alive but the Unity
                    # main thread is frozen - this is NOT 'ready' and waiting
                    # longer will not help.
                    return json.dumps({
                        "status": "blocked",
                        "error": f"Unity main thread blocked "
                                 f"(inactive for {last_blocked_s:.0f}s) after "
                                 f"waiting {timeout:.0f}s",
                        "main_thread_inactive_s": round(last_blocked_s, 1),
                        "hint": "Unity Editor appears frozen - use unity_kill "
                                "or unity_restart",
                        "state": unity.state.value,
                        "elapsed": round(elapsed, 1),
                        "phases": phases,
                    })
                return json.dumps({
                    "status": "timeout",
                    "error": f"Unity not ready after {timeout:.0f}s",
                    "state": unity.state.value,
                    "elapsed": round(elapsed, 1),
                    "phases": phases,
                })

            # If connected, verify with heartbeat
            if unity.connected:
                hb = await unity.heartbeat()
                if hb and hb.get("status") == "ok":
                    # The heartbeat is answered on a Unity background thread and
                    # succeeds even while the main thread is frozen. Check the
                    # reported pump inactivity before trusting anything else.
                    mt = unity.main_thread_inactive_s
                    if mt is not None and mt >= MAIN_THREAD_STALL_THRESHOLD:
                        last_blocked_s = mt
                        _record_phase("blocked")
                        await asyncio.sleep(poll_interval)
                        continue
                    # Also check if Unity is still compiling/importing
                    try:
                        status_result = await unity._execute_tool_call(
                            "editor_get_status", {})
                        status_data = json.loads(status_result)
                        error_text = status_data.get("error")
                        if error_text:
                            # Dispatch failed (main thread stalled, dispatcher
                            # unavailable, reload) - NOT ready. Never fall
                            # through to 'ready' on an error response.
                            mt = unity.main_thread_inactive_s
                            if mt is not None and mt >= MAIN_THREAD_STALL_THRESHOLD:
                                last_blocked_s = mt
                                _record_phase("blocked")
                            else:
                                _record_phase("busy")
                            await asyncio.sleep(poll_interval)
                            continue
                        last_blocked_s = None
                        is_playing = status_data.get("isPlaying", False)
                        if is_playing:
                            _record_phase("playing")
                            return json.dumps({
                                "status": "playing",
                                "error": "Unity is in play mode. Stop the simulation first (sim_stop) before recompiling or waiting for ready.",
                                "state": unity.state.value,
                                "isPlaying": True,
                                "waited": round(elapsed, 1),
                                "phases": phases,
                            })
                        is_compiling = status_data.get("isCompiling", False)
                        is_updating = status_data.get("isUpdating", False)
                        if is_compiling and is_updating:
                            _record_phase("compiling+importing")
                        elif is_compiling:
                            _record_phase("compiling")
                        elif is_updating:
                            _record_phase("importing")
                        if not is_compiling and not is_updating:
                            _record_phase("ready")
                            return json.dumps({
                                "status": "ready",
                                "state": unity.state.value,
                                "tools_count": len(unity.tools),
                                "waited": round(elapsed, 1),
                                "phases": phases,
                            })
                    except Exception:
                        pass  # Fall through to retry
            else:
                # Track disconnection/reload phases
                state_val = unity.state.value
                if state_val == "reloading":
                    _record_phase("domain_reload")
                elif state_val == "reconnecting":
                    _record_phase("reconnecting")
                else:
                    _record_phase(f"disconnected({state_val})")

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

    # --- Process control tools (work even when Unity is frozen or dead) ---
    # Pure Python/OS-side: no Unity WebSocket roundtrip involved. These are the
    # rescue path when the Unity main thread hangs ('Hold on' dialog, frozen
    # editor) and normal tools only time out.

    # Unity editor exe remembered from the last kill (preferred restart source)
    _remembered_unity_exe: list = [None]

    async def unity_kill() -> str:
        """Force-kill the Unity Editor process of THIS project.

        Process-selective: only Unity.exe processes whose -projectpath command
        line matches this project are killed (including asset import workers).
        Other Unity instances and this Python server are never touched.
        Works even when Unity is completely frozen - pure OS-level operation.
        """
        loop = asyncio.get_running_loop()
        project_root = _get_project_root()
        result = await loop.run_in_executor(
            None, _kill_unity_processes, project_root)
        if result["exe"]:
            _remembered_unity_exe[0] = result["exe"]
        # Drop our stale websocket so the watchdog reconnects cleanly later
        try:
            await unity.disconnect()
        except Exception:
            pass
        if not result["killed"]:
            return json.dumps({
                "status": "no_process",
                "message": "No Unity.exe with matching -projectpath found",
                "project_root": str(project_root),
            })
        return json.dumps({
            "status": "killed",
            "killed_pids": [p["pid"] for p in result["killed"]],
            "processes": result["killed"],
            "project_root": str(project_root),
        })

    unity_kill.__name__ = "unity_kill"
    mcp_server.add_tool(
        unity_kill,
        name="unity_kill",
        description=(
            "Force-kill the frozen/hanging Unity Editor process of THIS project "
            "(matched via -projectpath; other Unity instances stay untouched). "
            "Pure OS-level - works even when Unity is completely unresponsive."
        ),
    )

    async def unity_restart() -> str:
        """Kill the Unity Editor of THIS project and start it again.

        Kills all matching Unity processes, waits until the PIDs are gone
        (timeout 15s), then starts Unity detached with -projectpath. The
        Unity exe is remembered from the killed process; fallback is the
        matching Unity Hub installation for ProjectSettings/ProjectVersion.txt.
        """
        loop = asyncio.get_running_loop()
        project_root = _get_project_root()

        kill_result = await loop.run_in_executor(
            None, _kill_unity_processes, project_root)
        if kill_result["exe"]:
            _remembered_unity_exe[0] = kill_result["exe"]
        try:
            await unity.disconnect()
        except Exception:
            pass

        # Wait until the killed PIDs are really gone before restarting
        killed_pids = [p["pid"] for p in kill_result["killed"]]
        still_alive = list(killed_pids)
        deadline = time.monotonic() + KILL_WAIT_TIMEOUT
        while still_alive and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            checks = [
                await loop.run_in_executor(None, _pid_alive, pid)
                for pid in still_alive
            ]
            still_alive = [pid for pid, alive in zip(still_alive, checks) if alive]
        if still_alive:
            return json.dumps({
                "status": "kill_timeout",
                "error": f"Unity PIDs still alive after "
                         f"{KILL_WAIT_TIMEOUT:.0f}s: {still_alive}",
                "hint": "Retry unity_restart or kill manually via Task Manager",
                "project_root": str(project_root),
            })

        # Resolve Unity.exe: remembered from kill > Unity Hub installation
        exe = _remembered_unity_exe[0]
        if not exe or not Path(exe).exists():
            exe = await loop.run_in_executor(None, _find_unity_exe, project_root)
        if not exe:
            return json.dumps({
                "status": "failed",
                "error": "Unity.exe not found (no running instance to learn "
                         "from and no matching Unity Hub installation)",
                "project_root": str(project_root),
            })

        try:
            new_pid = await loop.run_in_executor(
                None, _start_unity, exe, project_root)
        except OSError as e:
            return json.dumps({
                "status": "failed",
                "error": f"Unity start failed: {e}",
                "exe": exe,
            })

        # Reset relay state so the watchdog reconnects once Unity is up
        unity._reconnect_attempts = 0
        if unity.state == State.ERROR:
            unity._set_state(State.RECONNECTING)
        await _trigger_watchdog()

        return json.dumps({
            "status": "started",
            "pid": new_pid,
            "exe": exe,
            "killed_pids": killed_pids,
            "project_root": str(project_root),
            "hint": "Unity is starting - use editor_wait_ready to wait for readiness",
        })

    unity_restart.__name__ = "unity_restart"
    mcp_server.add_tool(
        unity_restart,
        name="unity_restart",
        description=(
            "Kill the Unity Editor of THIS project and start it again with "
            "-projectpath (rescue for a frozen editor). Waits for the killed "
            "PIDs to disappear, then launches Unity detached. Works even when "
            "Unity is completely unresponsive."
        ),
    )

    # screenshot_editor is intercepted Python-side in call_tool() using Win32
    # PrintWindow API for reliable capture even when Unity is backgrounded.
    # Falls back to the C# implementation if Python capture fails.

    # --- WebViewer bridge ---
    wv_bridge: WebViewerBridge | None = None
    if not no_webviewer:
        wv_bridge = WebViewerBridge(
            host=ws_host,
            port=web_port,
            mcp_server=mcp_server,
        )
        logger.info(f"WebViewer bridge created (port {web_port})")
    else:
        logger.info("WebViewer bridge disabled (--no-webviewer)")

    return mcp_server, unity, registered_names, watchdog_holder, wv_bridge


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
                    save_schema_cache(unity.tools, unity._schema_version, cache_dir,
                                      instructions=unity._instructions)
                    # Apply fresh instructions from Unity
                    if unity._instructions:
                        mcp_server._mcp_server.instructions = unity._instructions
                    # Clear registered names after domain reload so new tools get registered
                    if was_reloading:
                        old_count = len(registered_names)
                        registered_names.clear()
                        logger.info(f"Watchdog: cleared {old_count} registered tools for re-registration after reload")
                    register_tools(mcp_server, unity.tools, unity, registered_names)
                    delay = RECONNECT_BASE_DELAY
                    # Notify MCP client that tool list changed
                    if hasattr(mcp_server, '_notify_tools_changed'):
                        await mcp_server._notify_tools_changed()

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
    parser.add_argument(
        "--web-port", type=int, default=DEFAULT_WV_PORT,
        help=f"WebViewer WebSocket port (default: {DEFAULT_WV_PORT})"
    )
    parser.add_argument(
        "--no-webviewer", action="store_true",
        help="Disable WebViewer bridge (no web_* tools)"
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

    mcp_server, unity, registered_names, watchdog_holder, wv_bridge = create_server(
        ws_host=args.ws_host,
        ws_port=ws_port,
        auth_token=args.auth_token,
        cache_dir=args.cache_dir,
        http_port=args.http_port,
        web_port=args.web_port,
        no_webviewer=args.no_webviewer,
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

    # Start WebViewer bridge as background task
    _wv_bridge_started = False

    async def _ensure_wv_bridge():
        nonlocal _wv_bridge_started
        if _wv_bridge_started or wv_bridge is None:
            return
        _wv_bridge_started = True
        logger.info("Starting WebViewer bridge")
        asyncio.ensure_future(wv_bridge.start())

    # Hook watchdog + WV bridge into FastMCP lifecycle (primary start mechanism)
    try:
        async def _init_handler(notification):
            await _ensure_watchdog()
            await _ensure_wv_bridge()
        mcp_server._mcp_server.notification_handlers[InitializedNotification] = _init_handler
        logger.debug("Watchdog + WV bridge hooked into MCP initialized notification")
    except Exception as e:
        logger.warning(f"Could not hook watchdog into MCP lifecycle: {e}")

    # Enable tools_changed notification so clients re-discover tools after
    # the watchdog registers new ones from Unity.
    # Monkey-patch create_initialization_options to set tools_changed=True.
    _orig_create_init = mcp_server._mcp_server.create_initialization_options
    def _patched_create_init(notification_options=None, experimental_capabilities=None):
        from mcp.server.lowlevel.server import NotificationOptions
        opts = notification_options or NotificationOptions()
        opts.tools_changed = True
        return _orig_create_init(opts, experimental_capabilities)
    mcp_server._mcp_server.create_initialization_options = _patched_create_init

    # Store session reference so watchdog can send tool list change notifications.
    # Monkey-patch the low-level server's run() to capture the session.
    _active_session = [None]
    _orig_run = mcp_server._mcp_server.run
    async def _patched_run(read_stream, write_stream, initialization_options, **kwargs):
        from mcp.server.session import ServerSession
        from contextlib import AsyncExitStack
        import anyio
        async with AsyncExitStack() as stack:
            lifespan_context = await stack.enter_async_context(
                mcp_server._mcp_server.lifespan(mcp_server._mcp_server))
            session = await stack.enter_async_context(
                ServerSession(read_stream, write_stream, initialization_options,
                              stateless=kwargs.get("stateless", False)))
            _active_session[0] = session
            logger.info("MCP session captured for tool list change notifications")
            async with anyio.create_task_group() as tg:
                async for message in session.incoming_messages:
                    tg.start_soon(
                        mcp_server._mcp_server._handle_message,
                        message, session, lifespan_context,
                        kwargs.get("raise_exceptions", False))
    mcp_server._mcp_server.run = _patched_run

    # Expose session reference and notification helper for the watchdog
    async def notify_tools_changed():
        session = _active_session[0]
        if session:
            try:
                await session.send_tool_list_changed()
                logger.info("Sent tools/list_changed notification to MCP client")
            except Exception as e:
                logger.debug(f"Could not send tools/list_changed: {e}")
    mcp_server._notify_tools_changed = notify_tools_changed

    wv_info = f", webviewer={args.ws_host}:{args.web_port}" if wv_bridge else ", webviewer=disabled"
    logger.info(
        f"Starting realvirtual MCP Server (mode={args.mode}, "
        f"ws={args.ws_host}:{ws_port}{wv_info})"
    )

    if args.mode == "sse":
        mcp_server.run(transport="sse")
    else:
        mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
