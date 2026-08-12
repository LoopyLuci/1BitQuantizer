"""Backend API tests covering repo-id/Hugging Face quantize flow."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def test_hub_info_endpoint_requires_model():
    from bitforge.api import app

    client = TestClient(app)
    response = client.get("/api/hub/info")
    assert response.status_code == 422


def test_hub_info_endpoint_rejects_local_path():
    from bitforge.api import app

    client = TestClient(app)
    response = client.get("/api/hub/info", params={"model": str(Path(__file__).parent)})
    assert response.status_code == 400


def test_hub_info_endpoint_accepts_repo_id(monkeypatch):
    from bitforge.api import app

    class FakeInfo:
        id = "user/repo"
        sha = "abc123"
        tags = ["transformers", "pytorch"]

    def fake_model_info(repo_id, revision="main"):
        assert repo_id == "user/repo"
        return FakeInfo()

    class FakeModule:
        class HfApi:
            @staticmethod
            def model_info(repo_id: str, revision: str = "main"):
                return fake_model_info(repo_id, revision)

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeModule)

    client = TestClient(app)
    response = client.get("/api/hub/info", params={"model": "user/repo"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "user/repo"
    assert payload["revision"] == "abc123"
    assert "transformers" in payload["tags"]


def test_parse_hf_model_normalizes_urls_and_revisions():
    from bitforge.engine import QuantizationEngine

    assert QuantizationEngine._parse_hf_model("https://huggingface.co/org/model") == ("org/model", "main")
    assert QuantizationEngine._parse_hf_model("org/model#v1.0") == ("org/model", "v1.0")
    assert QuantizationEngine._parse_hf_model("http://huggingface.co/org/model") == ("org/model", "main")
    assert QuantizationEngine._parse_hf_model("huggingface.co/org/model") == ("org/model", "main")
    with pytest.raises(ValueError):
        QuantizationEngine._parse_hf_model("not-a-valid-id")