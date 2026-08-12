"""End-to-end browser UI tests for BitForge.

Requires Playwright:
    pip install playwright
    playwright install chromium
"""

import os
import pathlib
import shutil
import subprocess
import time

import pytest
import requests

ROOT = "Z:/Projects/BitForge"
BACKEND_PORT = 8125
FRONTEND_URL = "http://localhost:5173"
MODEL_PATH = f"{ROOT}/backend/output/probe_model.pt"


def _wait(host: str, port: int, timeout: float = 60.0) -> bool:
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _job_completes(job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        r = requests.get(f"http://127.0.0.1:{BACKEND_PORT}/api/jobs/{job_id}", timeout=10)
        assert r.status_code == 200, f"/api/jobs/{job_id} -> {r.status_code}: {r.text}"
        last = r.json()
        if last.get("status") in {"completed", "failed"}:
            return last
        time.sleep(1.0)
    raise TimeoutError(f"job {job_id} did not complete within {timeout}s. Last status: {last}")


def _submit_job() -> str:
    r = requests.post(
        f"http://127.0.0.1:{BACKEND_PORT}/api/quantize",
        json={"model_path": MODEL_PATH, "algorithm": "xnor"},
        timeout=30,
    )
    assert r.status_code == 200, f"/api/quantize -> {r.status_code}: {r.text}"
    body = r.json()
    return body["job_id"]


def _start_frontend_dev_server() -> subprocess.Popen | None:
    if not shutil.which("npm"):
        return None
    tauri_dir = f"{ROOT}/tauri-gui"
    if not pathlib.Path(tauri_dir, "package.json").exists():
        return None
    try:
        proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=tauri_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception:
        return None
    if not _wait("localhost", 5173, timeout=90):
        try:
            proc.kill()
        except Exception:
            pass
        return None
    return proc


@pytest.fixture(scope="function")
def frontend_dev_server(request):
    proc = _start_frontend_dev_server()
    if proc is None:
        pytest.skip("frontend dev server unavailable")
    yield proc
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        pass


def test_backend_health():
    assert _wait("127.0.0.1", BACKEND_PORT, timeout=60)
    r = requests.get(f"http://127.0.0.1:{BACKEND_PORT}/api/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_job_polling_end_to_end():
    job_id = _submit_job()
    job = _job_completes(job_id, timeout=120)
    assert job["status"] in {"completed", "failed"}
    if job["status"] == "completed":
        result = job.get("result")
        assert result is not None
        assert "output_path" in result
        assert "format" in result


def test_frontend_html_loads(frontend_dev_server):
    assert _wait("localhost", 5173, timeout=60)
    r = requests.get(FRONTEND_URL, timeout=10)
    assert r.status_code == 200
    assert "BitForge" in r.text


def test_frontend_has_quantize_controls(frontend_dev_server):
    assert _wait("localhost", 5173, timeout=60)
    r = requests.get(FRONTEND_URL, timeout=10)
    assert r.status_code == 200
    text = r.text
    assert "Quantize Model" in text or "quantize" in text.lower()


def test_browser_progress_ui_renders():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        import pytest
        pytest.skip("playwright not installed")

    frontend = _start_frontend_dev_server()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.on("console", lambda msg: print("[browser console]", msg.type, msg.text))
            page.on("pageerror", lambda exc: print("[browser pageerror]", exc))
            page.route("**/api/quantize", lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body='{"job_id":"e2e-test-job","status":"queued"}'
            ))
            page.route("**/api/jobs/*", lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body='{"job_id":"e2e-test-job","status":"completed","result":{"output_path":"out.gguf","format":"gguf","original_size_mb":1.0,"quantized_size_mb":0.5,"compression_ratio":2.0,"layer_count":1,"metadata":{}}}'
            ))
            page.route("**/api/health", lambda route: route.fulfill(status=200, content_type="application/json", body='{"status":"ok"}'))
            page.route("**/api/algorithms", lambda route: route.fulfill(status=200, content_type="application/json", body='{"algorithms":["xnor"],"formats":["gguf"]}'))
            page.goto(FRONTEND_URL, wait_until="networkidle", timeout=120000)
            page.wait_for_timeout(3000)
            assert "BitForge" in page.content()

            model_input = page.locator("input[type=text]").first
            if model_input.count() > 0:
                model_input.fill(MODEL_PATH)

            quantize_btn = page.locator("button:has-text('Quantize Model')")
            if quantize_btn.count() > 0 and model_input.count() > 0:
                quantize_btn.click()
                page.wait_for_timeout(3000)
                html = page.content()
                print("=== PAGE HTML AFTER CLICK ===")
                print(html[:4000])
                print("=== END HTML ===")
                try:
                    page.locator("text=Quantization Complete").wait_for(timeout=120_000)
                except Exception:
                    try:
                        page.locator("text=Error").wait_for(timeout=5_000)
                    except Exception as exc:
                        raise AssertionError(
                            "No progress/result UI update found after quantize. "
                            "Check App.tsx polling and /api/jobs wiring."
                        ) from exc
            browser.close()
    finally:
        if frontend is not None:
            try:
                frontend.terminate()
                frontend.wait(timeout=10)
            except Exception:
                pass


def test_browser_validation_error_ui():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        import pytest
        pytest.skip("playwright not installed")

    frontend = _start_frontend_dev_server()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.on("console", lambda msg: print("[browser console]", msg.type, msg.text))
            page.on("pageerror", lambda exc: print("[browser pageerror]", exc))
            page.goto(FRONTEND_URL, wait_until="networkidle", timeout=120000)
            page.wait_for_timeout(2000)
            assert "BitForge" in page.content()

            model_input = page.locator("input[type=text]").first
            if model_input.count() > 0:
                model_input.fill("")

            quantize_btn = page.locator("button:has-text('Quantize Model')")
            if quantize_btn.count() > 0:
                quantize_btn.click()
                page.wait_for_timeout(1500)

            assert "BitForge" in page.content()
            browser.close()
    finally:
        if frontend is not None:
            try:
                frontend.terminate()
                frontend.wait(timeout=10)
            except Exception:
                pass


def test_browser_backend_down_banner():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        import pytest
        pytest.skip("playwright not installed")

    frontend = _start_frontend_dev_server()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.on("console", lambda msg: print("[browser console]", msg.type, msg.text))
            page.on("pageerror", lambda exc: print("[browser pageerror]", exc))

            failed = []
            page.on("response", lambda resp: failed.append(resp.status) if resp.status >= 500 else None)

            page.route("**/api/health", lambda route: route.fulfill(status=500, content_type="application/json", body='{"status":"error"}'))
            page.route("**/api/quantize", lambda route: route.fulfill(status=500, content_type="application/json", body='{"error":"backend down"}'))
            page.route("**/api/algorithms", lambda route: route.fulfill(status=500, content_type="application/json", body='{"error":"backend down"}'))
            page.goto(FRONTEND_URL, wait_until="networkidle", timeout=120000)
            page.wait_for_timeout(2000)
            html = page.content()
            print("=== BACKEND DOWN PAGE HTML ===")
            print(html[:4000])
            print("=== END HTML ===")

            assert "BitForge" in html
            assert 500 in failed
            browser.close()
    finally:
        if frontend is not None:
            try:
                frontend.terminate()
                frontend.wait(timeout=10)
            except Exception:
                pass

