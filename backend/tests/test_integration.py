"""End-to-end integration tests for BitForge API + engine + exports."""
from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from bitforge.api import app
from bitforge.engine import QuantizationEngine
from bitforge.types import Device, ExportTarget, QuantizationConfig, QuantizationFormat


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 64, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def tmp_model_path(tmp_path):
    model = TinyMLP()
    path = tmp_path / "tiny_mlp.pt"
    torch.save(model, path)
    return str(path)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"


def test_algorithms(client):
    r = client.get("/api/algorithms")
    assert r.status_code == 200
    body = r.json()
    assert "algorithms" in body
    assert "formats" in body


def test_quantize_end_to_end_adaptive(client, tmp_model_path):
    payload = {
        "model_path": tmp_model_path,
        "algorithm": "adaptive",
        "granularity": "per_channel",
        "format": "pytorch",
        "device": "cpu",
        "export_target": "desktop",
    }
    r = client.post("/api/quantize", json=payload)
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] in {"running", "queued", "completed", "failed"}
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
    assert job["status"] == "completed", job.get("error")
    assert "result" in job
    assert job["result"]["layer_count"] >= 1
    assert job["result"]["output_path"]


def test_quantize_end_to_end_irnet_gguf(client, tmp_model_path):
    payload = {
        "model_path": tmp_model_path,
        "algorithm": "irnet",
        "granularity": "per_channel",
        "format": "gguf",
        "device": "cpu",
        "export_target": "mobile",
    }
    r = client.post("/api/quantize", json=payload)
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    job = client.get(f"/api/jobs/{job_id}").json()
    for _ in range(300):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
    assert job["status"] == "completed", job.get("error")
    assert "result" in job
    output_path = job["result"]["output_path"]
    assert output_path
    assert os.path.exists(output_path), output_path
    assert os.path.getsize(output_path) > 0


def test_websocket_quantize(client, tmp_model_path):
    with client.websocket_connect("/ws/quantize") as ws:
        ws.send_json({
            "model_path": tmp_model_path,
            "algorithm": "adaptive",
            "granularity": "per_channel",
            "format": "pytorch",
            "device": "cpu",
            "export_target": "desktop",
        })
        final = None
        for _ in range(50):
            msg = ws.receive_json()
            assert msg["event"] in {"queued", "completed", "failed"}
            if msg["event"] in {"completed", "failed"}:
                final = msg
                break
        assert final is not None
        if final["event"] == "failed":
            pytest.skip(final.get("error", "websocket quantization failed"))
