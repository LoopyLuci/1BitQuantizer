"""BitForge standalone desktop launcher.

Embeds the FastAPI backend and shows the frontend via pywebview.
No external browser, no manually-started servers.
"""
from __future__ import annotations

import os
import site
import socket
import sys
import threading
import time
from pathlib import Path

import uvicorn
import webview


def _find_repo_root(start: Path) -> Path:
    """Best-effort repo root resolution for dev and PyInstaller exe."""
    bundle_root = Path(getattr(sys, "_MEIPASS", "")) if hasattr(sys, "_MEIPASS") else None
    candidates = [start.resolve()]
    if bundle_root is not None:
        candidates.insert(0, bundle_root)
    for parent in candidates:
        if parent.exists() and (parent / "tauri-gui" / "dist" / "index.html").exists():
            return parent
    if start.name.lower() == "app.py":
        return start.parents[1]
    return start.parents[2]


ROOT = _find_repo_root(Path(__file__))
FRONTEND_DIST = ROOT / "tauri-gui" / "dist"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

for _p in site.getsitepackages() + [site.getusersitepackages()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_BACKEND_SERVICE_PATH = ROOT / "backend_service.py"


def _start_backend(port: int = 8125) -> None:
    os.environ.setdefault("BITFORGE_BACKEND_PORT", str(port))
    backend_app = None
    if _BACKEND_SERVICE_PATH.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "backend_service", str(_BACKEND_SERVICE_PATH)
        )
        backend_service = importlib.util.module_from_spec(spec)
        sys.modules["backend_service"] = backend_service
        spec.loader.exec_module(backend_service)
        backend_app = backend_service.app
    if backend_app is None:
        from backend_service import app as backend_app

    uvicorn.run(backend_app, host="127.0.0.1", port=port, log_level="warning")


class BitForgeAPI:
    def __init__(self, backend_port: int = 8125):
        self.backend_port = backend_port

    def health(self) -> dict:
        import requests

        try:
            r = requests.get(
                f"http://127.0.0.1:{self.backend_port}/api/health", timeout=2
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def minimize(self) -> None:
        webview.windows[0].minimize()

    def close(self) -> None:
        webview.windows[0].destroy()


def _wait_for_backend(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


def _pick_free_port(preferred: int = 8125, limit: int = 10) -> int:
    import random

    tried = {preferred}
    for _ in range(limit):
        port = preferred if preferred not in tried else random.randint(8125, 8225)
        tried.add(port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"no free backend port found among {sorted(tried)}")


def main() -> int:
    backend_port = int(os.getenv("BITFORGE_BACKEND_PORT", "0"))
    if backend_port <= 0:
        backend_port = _pick_free_port(8125, limit=12)
    index_html = FRONTEND_DIST / "index.html"

    if not index_html.exists():
        print(f"[launcher] frontend dist not found at {index_html}")
        print("[launcher] run: cd tauri-gui && npm run build")
        return 1

    print(f"[launcher] using frontend dist: {FRONTEND_DIST}")
    print(f"[launcher] starting embedded backend on 127.0.0.1:{backend_port}")
    backend_thread = threading.Thread(
        target=_start_backend, args=(backend_port,), daemon=True
    )
    backend_thread.start()

    if not _wait_for_backend(backend_port):
        print("[launcher] backend failed to start")
        return 1
    print("[launcher] backend ready")

    api = BitForgeAPI(backend_port=backend_port)
    entry = f"http://127.0.0.1:{backend_port}/"
    webview.create_window(
        "BitForge",
        entry,
        js_api=api,
        width=1200,
        height=800,
        resizable=True,
        text_select=True,
    )
    webview.start(debug=False, http_server=False, gui="edgechromium")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
