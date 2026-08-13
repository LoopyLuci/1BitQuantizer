"""Benchmark suite for BitForge quantization."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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


@dataclass
class BenchmarkResult:
    name: str
    original_latency_ms: float
    quantized_latency_ms: float
    speedup: float
    output_path: Optional[Path] = None
    compression_ratio: Optional[float] = None
    layer_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "original_latency_ms": self.original_latency_ms,
            "quantized_latency_ms": self.quantized_latency_ms,
            "speedup": self.speedup,
            "output_path": str(self.output_path) if self.output_path else None,
            "compression_ratio": self.compression_ratio,
            "layer_count": self.layer_count,
        }


def _benchmark(model: nn.Module, device: torch.device, repeats: int = 50) -> float:
    model.eval()
    x = torch.randn(1, 128, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        return (time.perf_counter() - start) / repeats * 1000.0


def run_algorithm(name: str, algorithm: str, fmt: QuantizationFormat = QuantizationFormat.PYTORCH) -> BenchmarkResult:
    device = Device.AUTO
    if device == Device.AUTO:
        d = torch.device("cpu")
    else:
        d = torch.device(device.value)

    model = TinyMLP().to(d)
    original_latency = _benchmark(model, d)

    config = QuantizationConfig(algorithm=algorithm, granularity="per_channel", format=fmt)
    engine = QuantizationEngine(config)
    qm, stats = engine.quantize_model(model)
    result = engine._export(qm, stats, type("Src", (), {"is_local": True, "path": None, "repo_id": None})())
    qm.to(d)
    quantized_latency = _benchmark(qm, d)

    return BenchmarkResult(
        name=name,
        original_latency_ms=original_latency,
        quantized_latency_ms=quantized_latency,
        speedup=original_latency / max(1e-6, quantized_latency),
        output_path=result.output_path,
        compression_ratio=result.compression_ratio,
        layer_count=result.layer_count,
    )


def main() -> int:
    algorithms = ["adaptive", "irnet", "xnor", "binarize"]
    results: List[BenchmarkResult] = []
    for alg in algorithms:
        try:
            res = run_algorithm(alg, alg)
            results.append(res)
            print(f"[bench] {alg}: orig={res.original_latency_ms:.3f}ms quant={res.quantized_latency_ms:.3f}ms speedup={res.speedup:.2f}x")
        except Exception as exc:
            print(f"[bench] {alg}: {exc}")

    out_path = Path("output") / "benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    print(f"[bench] saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
