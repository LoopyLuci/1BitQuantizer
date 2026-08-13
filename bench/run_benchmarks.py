"""Benchmark suite for BitForge quantization."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from bitforge.engine import QuantizationEngine, count_layers
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


class MediumMLP(nn.Module):
    def __init__(self, in_dim: int = 512, hidden: int = 1024, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BenchmarkResult:
    name: str
    model_class: str
    parameter_count: int
    original_latency_ms: float
    quantized_latency_ms: float
    speedup: float
    output_path: Optional[str] = None
    compression_ratio: Optional[float] = None
    layer_count: Optional[int] = None
    original_size_mb: Optional[float] = None
    quantized_size_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _benchmark(model: nn.Module, device: torch.device, repeats: int = 50, input_dim: int = 128) -> float:
    model.eval()
    x = torch.randn(1, input_dim, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        return (time.perf_counter() - start) / repeats * 1000.0


def run_algorithm(name: str, algorithm: str, model_cls, input_dim: int, fmt: QuantizationFormat = QuantizationFormat.PYTORCH) -> BenchmarkResult:
    device = torch.device("cpu")
    model = model_cls().to(device)
    param_count = _count_parameters(model)
    original_latency = _benchmark(model, device, input_dim=input_dim)

    config = QuantizationConfig(algorithm=algorithm, granularity="per_channel", format=fmt)
    engine = QuantizationEngine(config)
    qm, stats = engine.quantize_model(model)

    src = type("Src", (), {"is_local": True, "path": None, "repo_id": None})()
    result = engine._export(qm, stats, src)
    qm.to(device)
    quantized_latency = _benchmark(qm, device, input_dim=input_dim)

    return BenchmarkResult(
        name=name,
        model_class=model_cls.__name__,
        parameter_count=param_count,
        original_latency_ms=original_latency,
        quantized_latency_ms=quantized_latency,
        speedup=original_latency / max(1e-6, quantized_latency),
        output_path=str(result.output_path),
        compression_ratio=result.compression_ratio,
        layer_count=result.layer_count,
        original_size_mb=result.original_size_mb,
        quantized_size_mb=result.quantized_size_mb,
    )


def main() -> int:
    cases = [
        ("tiny-adaptive", "adaptive", TinyMLP, 128),
        ("tiny-irnet", "irnet", TinyMLP, 128),
        ("tiny-xnor", "xnor", TinyMLP, 128),
        ("tiny-binarize", "binarize", TinyMLP, 128),
        ("medium-adaptive", "adaptive", MediumMLP, 512),
        ("medium-irnet", "irnet", MediumMLP, 512),
        ("medium-xnor", "xnor", MediumMLP, 512),
        ("medium-binarize", "binarize", MediumMLP, 512),
    ]

    results: List[BenchmarkResult] = []
    for name, algorithm, model_cls, input_dim in cases:
        try:
            res = run_algorithm(name, algorithm, model_cls, input_dim)
            results.append(res)
            print(f"[bench] {name}: params={res.parameter_count:,} orig={res.original_latency_ms:.3f}ms quant={res.quantized_latency_ms:.3f}ms speedup={res.speedup:.2f}x compression={res.compression_ratio:.2f}x")
        except Exception as exc:
            print(f"[bench] {name}: {exc}")

    out_path = Path("output") / "benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    print(f"[bench] saved {len(results)} results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
