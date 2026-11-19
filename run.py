"""Single-command runner for both backend and frontend development servers.

Usage:
    python run.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path


def stream_output(process: subprocess.Popen, prefix: str) -> None:
    """Stream stdout/stderr from a subprocess line by line with a prefix."""
    if process.stdout is None:
        return
    for line in iter(process.stdout.readline, ""):
        if not line:
            break
        print(f"[{prefix}] {line.rstrip()}", flush=True)


def main() -> int:
    os.environ["PYTHONUNBUFFERED"] = "1"
    root_dir = Path(__file__).resolve().parent
    frontend_dir = root_dir / "frontend"

    if not frontend_dir.exists():
        print(f"Error: Frontend directory not found at {frontend_dir}", file=sys.stderr, flush=True)
        return 1

    print("=" * 70, flush=True)
    print("Starting LLM Reliability & Evaluation Platform", flush=True)
    print("  Backend API:  http://127.0.0.1:8000 (Interactive Docs: /api/v1/docs)", flush=True)
    print("  Frontend UI:  http://localhost:5173", flush=True)
    print("  Press Ctrl+C to terminate both servers.", flush=True)
    print("=" * 70, flush=True)

    is_windows = sys.platform == "win32"

    # Backend command
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "llm_reliability.api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
        "--reload-dir",
        str(root_dir / "src"),
    ]

    # Frontend command
    frontend_cmd = "npm run dev" if is_windows else ["npm", "run", "dev"]

    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(root_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(frontend_dir),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=is_windows,
    )

    t_backend = threading.Thread(
        target=stream_output, args=(backend_proc, "backend"), daemon=True
    )
    t_frontend = threading.Thread(
        target=stream_output, args=(frontend_proc, "frontend"), daemon=True
    )

    t_backend.start()
    t_frontend.start()

    def shutdown(signum=None, frame=None):
        print("\nShutting down development servers...")
        try:
            if is_windows:
                # Force kill process tree on Windows to ensure subshells close cleanly
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(backend_proc.pid)],
                    capture_output=True,
                )
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(frontend_proc.pid)],
                    capture_output=True,
                )
            else:
                backend_proc.terminate()
                frontend_proc.terminate()
        except Exception:
            pass
        print("Both servers stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            b_poll = backend_proc.poll()
            f_poll = frontend_proc.poll()
            if b_poll is not None:
                print(f"[backend] Server process exited with code {b_poll}")
                shutdown()
            if f_poll is not None:
                print(f"[frontend] Vite process exited with code {f_poll}")
                shutdown()
            threading.Event().wait(0.5)
    except KeyboardInterrupt:
        shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
