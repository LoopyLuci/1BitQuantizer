"""Shared fixtures for backend integration tests."""
from __future__ import annotations

import os
import socket
import subprocess
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_PORT = int(os.getenv("BITFORGE_BACKEND_PORT", "8125"))
BACKEND_HOST = "127.0.0.1"
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
BACKEND_CMD = [
    os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
    "-m",
    "bitforge.run_api",
]


def _wait(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.25)
    return False


@pytest.fixture(scope="session", autouse=True)
def _ensure_backend():
    """Start the FastAPI backend if it is not already reachable."""
    if _wait(BACKEND_HOST, BACKEND_PORT, timeout=2):
        yield
        return

    env = os.environ.copy()
    env.setdefault("BITFORGE_BACKEND_PORT", str(BACKEND_PORT))
    proc = subprocess.Popen(
        BACKEND_CMD,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        if not _wait(BACKEND_HOST, BACKEND_PORT, timeout=60):
            raise RuntimeError("backend did not start within timeout")
        yield
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
