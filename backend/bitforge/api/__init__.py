"""FastAPI backend for BitForge with full quantization pipeline."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from bitforge.engine import QuantizationEngine, _LayerStatsCollector, _XNORLinear, _BinarizeLinear
from bitforge.types import (
    Device,
    ExportTarget,
    LayerStats,
    ModelSource,
    QuantizationConfig,
    QuantizationFormat,
    QuantizationResult,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state: Dict[str, Any] = {"tasks": set()}
    yield state
    for task in list(state["tasks"]):
        if not task.done():
            task.cancel()
    if state["tasks"]:
        await asyncio.gather(*state["tasks"], return_exceptions=True)


app = FastAPI(title="BitForge API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path("output").resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = BASE_DIR / "jobs.db"


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            started_at REAL NOT NULL,
            finished_at REAL,
            payload TEXT,
            result TEXT,
            error TEXT
        )
        """
    )
    conn.commit()
    return conn


# In-memory job tracking
_jobs: Dict[str, Dict] = {}
_task_registry: set = set()


def _persist_job(job_id: str, row: Dict[str, Any]) -> None:
    payload = row.get("payload")
    result = row.get("result")
    error = row.get("error")
    with _db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs(job_id, status, started_at, finished_at, payload, result, error)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                row["status"],
                row.get("started_at", time.time()),
                row.get("finished_at"),
                json.dumps(payload) if payload is not None else None,
                json.dumps(result) if result is not None else None,
                error,
            ),
        )


def _load_jobs() -> None:
    with _db() as conn:
        rows = conn.execute("SELECT job_id, status, started_at, finished_at, payload, result, error FROM jobs").fetchall()
    for row in rows:
        job_id = row["job_id"]
        data: Dict[str, Any] = {
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "payload": json.loads(row["payload"]) if row["payload"] else None,
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": row["error"],
        }
        _jobs[job_id] = data


_load_jobs()


class QuantizeRequest(BaseModel):
    model_path: str
    repo_id: Optional[str] = None
    revision: str = "main"
    algorithm: str = "xnor"
    granularity: str = "per_channel"
    group_size: int = 32
    calibrate: bool = False
    calibration_batches: int = 100
    quantize_embeddings: bool = False
    quantize_final_norm: bool = False
    quantize_bias: bool = False
    format: QuantizationFormat = QuantizationFormat.GGUF
    export_target: ExportTarget = ExportTarget.MOBILE
    device: Device = Device.AUTO
    layer_exclude_pattern: Optional[str] = None
    layer_include_pattern: Optional[str] = None
    layer_exclude_names: List[str] = []
    layer_include_names: List[str] = []
    optimize_for_inference: bool = True
    include_tokenizer: bool = True


class ExportRequest(BaseModel):
    job_id: str
    format: QuantizationFormat = QuantizationFormat.GGUF


def _resolve_device(device: Device) -> torch.device:
    if device == Device.AUTO:
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except AttributeError:
            pass
        return torch.device("cpu")
    return torch.device(device.value)


def _resolve_hf_model(model: str) -> tuple[str, str]:
    if not QuantizationEngine._is_hf_model(model):
        raise ValueError("Expected a Hugging Face repo ID or https://huggingface.co/... URL")
    return QuantizationEngine._parse_hf_model(model)


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": "0.1.0", "cuda": str(torch.cuda.is_available())}


@app.get("/api/hub/info")
async def hub_info(model: str, revision: str = "main") -> Dict[str, Any]:
    model = model.strip()
    if not model:
        return JSONResponse(status_code=400, content={"error": "model query parameter is required"})

    if not QuantizationEngine._is_hf_model(model):
        return JSONResponse(status_code=400, content={"error": "Provide a Hugging Face repo ID or https://huggingface.co/... URL"})

    try:
        repo_id, resolved_revision = QuantizationEngine._parse_hf_model(model)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(repo_id, revision=revision or resolved_revision)
        return {
            "model": repo_id,
            "revision": getattr(info, "sha", None),
            "tags": list(getattr(info, "tags", []) or []),
            "id": getattr(info, "id", repo_id),
        }
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"huggingface_hub error: {exc}"})


@app.post("/api/quantize")
async def quantize(req: QuantizeRequest) -> Dict[str, Any]:
    errors: List[str] = []
    allowed_algorithms = {"xnor", "binarize", "irnet", "adaptive", "binarynet", "bilevel"}
    allowed_granularity = {"per_tensor", "per_channel", "per_group"}
    allowed_formats = {f.value for f in QuantizationFormat}
    allowed_targets = {t.value for t in ExportTarget}
    allowed_devices = {d.value for d in Device}

    if not req.repo_id and not req.model_path:
        errors.append("Provide model_path or repo_id")
    if req.model_path and not Path(req.model_path).exists():
        errors.append(f"model_path does not exist: {req.model_path}")
    if req.algorithm not in allowed_algorithms:
        errors.append(f"algorithm must be one of: {sorted(allowed_algorithms)}")
    if req.granularity not in allowed_granularity:
        errors.append(f"granularity must be one of: {sorted(allowed_granularity)}")
    if req.group_size < 1:
        errors.append("group_size must be positive")
    if req.format.value not in allowed_formats:
        errors.append(f"format must be one of: {sorted(allowed_formats)}")
    if req.export_target.value not in allowed_targets:
        errors.append(f"export_target must be one of: {sorted(allowed_targets)}")
    if req.device.value not in allowed_devices:
        errors.append(f"device must be one of: {sorted(allowed_devices)}")

    if errors:
        return JSONResponse(status_code=400, content={"error": "; ".join(errors)})

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"job_id": job_id, "status": "queued", "started_at": time.time(), "result": None, "payload": req.model_dump()}
    _persist_job(job_id, _jobs[job_id])

    async def _run_job() -> None:
        try:
            source = ModelSource(
                path=req.model_path if not req.repo_id else None,
                repo_id=req.repo_id,
                revision=req.revision,
                is_local=not req.repo_id,
            )
            config = QuantizationConfig(
                algorithm=req.algorithm,
                granularity=req.granularity,
                group_size=req.group_size,
                calibrate=req.calibrate,
                calibration_batches=req.calibration_batches,
                quantize_embeddings=req.quantize_embeddings,
                quantize_final_norm=req.quantize_final_norm,
                quantize_bias=req.quantize_bias,
                format=req.format,
                export_target=req.export_target,
                device=req.device,
                layer_exclude_pattern=req.layer_exclude_pattern,
                layer_include_pattern=req.layer_include_pattern,
                layer_exclude_names=req.layer_exclude_names,
                layer_include_names=req.layer_include_names,
                optimize_for_inference=req.optimize_for_inference,
                include_tokenizer=req.include_tokenizer,
            )
            engine = QuantizationEngine(config)
            result = engine.quantize_from_source(source)
            _jobs[job_id]["status"] = "completed" if result.success else "failed"
            _jobs[job_id]["result"] = {
                "output_path": str(result.output_path),
                "format": result.format.value,
                "original_size_mb": result.original_size_mb,
                "quantized_size_mb": result.quantized_size_mb,
                "compression_ratio": result.compression_ratio,
                "layer_count": result.layer_count,
                "metadata": result.metadata,
            }
            if not result.success:
                _jobs[job_id]["error"] = result.error
            _jobs[job_id]["finished_at"] = time.time()
            _persist_job(job_id, _jobs[job_id])
        except Exception as exc:
            import traceback
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = traceback.format_exc()
            _jobs[job_id]["finished_at"] = time.time()
            _persist_job(job_id, _jobs[job_id])

    try:
        task = asyncio.get_running_loop().create_task(_run_job())
        _task_registry.add(task)
        task.add_done_callback(lambda t: _task_registry.discard(t))
    except Exception as exc:
        import traceback
        return JSONResponse(status_code=500, content={"error": traceback.format_exc()})
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> Dict[str, Any]:
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    res = job.get("result")
    payload: Dict[str, Any] = {"job_id": job_id, "status": job["status"]}
    if job.get("error"):
        payload["error"] = job["error"]
    if res:
        if isinstance(res, dict):
            payload["result"] = res
        else:
            payload["result"] = {
                "output_path": str(res.output_path),
                "format": res.format.value,
                "original_size_mb": res.original_size_mb,
                "quantized_size_mb": res.quantized_size_mb,
                "compression_ratio": res.compression_ratio,
                "layer_count": res.layer_count,
                "metadata": res.metadata,
            }
    return payload


@app.get("/api/files/{path:path}")
async def get_file(path: str) -> FileResponse:
    safe = (BASE_DIR / path).resolve()
    if not str(safe).startswith(str(BASE_DIR)):
        return JSONResponse(status_code=403, content={"error": "forbidden"})
    if not safe.exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(safe)


@app.post("/api/export")
async def export_model(req: ExportRequest) -> Dict[str, Any]:
    job = _jobs.get(req.job_id)
    if not job or not job.get("result"):
        return JSONResponse(status_code=404, content={"error": "job not found"})
    return {"job_id": req.job_id, "format": req.format.value, "path": str(job["result"].output_path)}


@app.websocket("/ws/quantize")
async def ws_quantize(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            job_id = str(uuid.uuid4())
            _jobs[job_id] = {"status": "queued", "started_at": time.time(), "result": None}
            await ws.send_json({"event": "queued", "job_id": job_id})
            
            def job() -> Dict[str, Any]:
                try:
                    req = QuantizeRequest(**msg)
                    source = ModelSource(path=req.model_path, repo_id=req.repo_id, revision=req.revision, is_local=not req.repo_id)
                    config = QuantizationConfig(
                        algorithm=req.algorithm,
                        granularity=req.granularity,
                        group_size=req.group_size,
                        calibrate=req.calibrate,
                        calibration_batches=req.calibration_batches,
                        quantize_embeddings=req.quantize_embeddings,
                        quantize_final_norm=req.quantize_final_norm,
                        quantize_bias=req.quantize_bias,
                        format=req.format,
                        export_target=req.export_target,
                        device=req.device,
                        layer_exclude_pattern=req.layer_exclude_pattern,
                        layer_include_pattern=req.layer_include_pattern,
                        layer_exclude_names=req.layer_exclude_names,
                        layer_include_names=req.layer_include_names,
                        optimize_for_inference=req.optimize_for_inference,
                    )
                    engine = QuantizationEngine(config)
                    result = engine.quantize_from_source(source)
                    _jobs[job_id]["status"] = "completed" if result.success else "failed"
                    _jobs[job_id]["result"] = {
                        "output_path": str(result.output_path),
                        "format": result.format.value,
                        "original_size_mb": result.original_size_mb,
                        "quantized_size_mb": result.quantized_size_mb,
                        "compression_ratio": result.compression_ratio,
                        "layer_count": result.layer_count,
                        "metadata": result.metadata,
                    }
                    _jobs[job_id]["finished_at"] = time.time()
                    _persist_job(job_id, _jobs[job_id])
                    return {
                        "event": "completed" if result.success else "failed",
                        "job_id": job_id,
                        "result": {
                            "output_path": str(result.output_path),
                            "format": result.format.value,
                            "original_size_mb": result.original_size_mb,
                            "quantized_size_mb": result.quantized_size_mb,
                            "compression_ratio": result.compression_ratio,
                            "layer_count": result.layer_count,
                            "metadata": result.metadata,
                        },
                    }
                except Exception as exc:
                    _jobs[job_id]["status"] = "failed"
                    _jobs[job_id]["error"] = str(exc)
                    _jobs[job_id]["finished_at"] = time.time()
                    _persist_job(job_id, _jobs[job_id])
                    return {"event": "failed", "job_id": job_id, "error": str(exc)}
            
            final = await asyncio.get_event_loop().run_in_executor(None, job)
            await ws.send_json(final)
    except WebSocketDisconnect:
        pass


@app.get("/api/algorithms")
async def list_algorithms() -> Dict[str, List[str]]:
    return {"algorithms": ["xnor", "binarize", "irnet", "adaptive", "binarynet", "bilevel"], "formats": [f.value for f in QuantizationFormat]}
