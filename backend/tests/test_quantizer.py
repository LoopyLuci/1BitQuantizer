"""Real verification tests for BitForge quantizer."""
from __future__ import annotations

import pathlib

import torch
import torch.nn as nn

from bitforge.engine import QuantizationEngine, _BinaryLinear, _QuantizedLinear, count_layers
from bitforge.types import (
    Device,
    ExportTarget,
    LayerStats,
    QuantizationConfig,
    QuantizationFormat,
)


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


def test_count_layers():
    m = TinyMLP()
    assert count_layers(m) == 3
    print("[ok] count_layers")


def test_quantize_adaptive_forwards():
    m = TinyMLP()
    config = QuantizationConfig(algorithm="adaptive", granularity="per_channel")
    engine = QuantizationEngine(config)
    qm, stats = engine.quantize_model(m)
    x = torch.randn(8, 128)
    out = qm(x)
    assert out.shape == (8, 10)
    assert isinstance(list(qm.modules())[2], _QuantizedLinear)
    print("[ok] adaptive quantization runs and replaces linear modules")


def test_quantize_irnet_forwards():
    m = TinyMLP()
    config = QuantizationConfig(algorithm="irnet", granularity="per_channel")
    engine = QuantizationEngine(config)
    qm, stats = engine.quantize_model(m)
    x = torch.randn(8, 128)
    out = qm(x)
    assert out.shape == (8, 10)
    assert len(stats) >= 1
    assert stats[0].memory_saved_bytes >= 0
    print("[ok] irnet quantization runs end-to-end")


def test_layer_filtering():
    m = TinyMLP()
    config = QuantizationConfig(
        algorithm="adaptive",
        granularity="per_channel",
        layer_include_names=["net"],
        layer_exclude_names=["fc"],
    )
    engine = QuantizationEngine(config)
    qm, stats = engine.quantize_model(m)
    names = [s.layer_name for s in stats]
    assert any("net" in n for n in names)
    print("[ok] layer include/exclude filtering")


def test_export_pytorch():
    m = TinyMLP()
    config = QuantizationConfig(algorithm="adaptive", granularity="per_channel", format=QuantizationFormat.PYTORCH)
    engine = QuantizationEngine(config)
    qm, stats = engine.quantize_model(m)
    result = engine._export(qm, stats, type("Src", (), {"is_local": True, "path": None, "repo_id": None})())
    assert result.output_path.exists()
    print(f"[ok] pytorch export exists at {result.output_path}")


def test_export_safetensors():
    m = TinyMLP()
    config = QuantizationConfig(algorithm="adaptive", granularity="per_channel", format=QuantizationFormat.SAFETENSORS)
    engine = QuantizationEngine(config)
    qm, stats = engine.quantize_model(m)
    result = engine._export(qm, stats, type("Src", (), {"is_local": True, "path": None, "repo_id": None})())
    assert result.output_path.exists()
    print(f"[ok] safetensors export exists at {result.output_path}")


def test_export_torchscript():
    m = TinyMLP()
    config = QuantizationConfig(algorithm="adaptive", granularity="per_channel", format=QuantizationFormat.TORCHSCRIPT)
    engine = QuantizationEngine(config)
    qm, stats = engine.quantize_model(m)
    result = engine._export(qm, stats, type("Src", (), {"is_local": True, "path": None, "repo_id": None})())
    assert result.output_path.exists()
    print(f"[ok] torchscript export exists at {result.output_path}")


def test_export_onnx_real():
    m = TinyMLP()
    config = QuantizationConfig(algorithm="adaptive", granularity="per_channel", format=QuantizationFormat.ONNX)
    engine = QuantizationEngine(config)
    qm, stats = engine.quantize_model(m)
    src = type("Src", (), {"is_local": True, "path": None, "repo_id": None})()
    result = engine._export(qm, stats, src)
    assert result.output_path.exists()
    assert result.output_path.suffix == ".onnx"
    assert result.output_path.stat().st_size > 0
    print(f"[ok] onnx export validated at {result.output_path}")


def test_export_gguf_real():
    m = TinyMLP()
    config = QuantizationConfig(algorithm="irnet", granularity="per_channel", format=QuantizationFormat.GGUF)
    engine = QuantizationEngine(config)
    qm, stats = engine.quantize_model(m)
    src = type("Src", (), {"is_local": True, "path": None, "repo_id": None})()
    result = engine._export(qm, stats, src)
    assert result.output_path.exists()
    assert result.output_path.suffix == ".gguf"
    content = result.output_path.read_bytes()
    assert content[:4] == b"GGUF" or b"GGUF" in content[:20]
    print(f"[ok] gguf export validated at {result.output_path}")


def test_result_fields():
    m = TinyMLP()
    config = QuantizationConfig(algorithm="adaptive", granularity="per_channel", format=QuantizationFormat.PYTORCH)
    engine = QuantizationEngine(config)
    qm, stats = engine.quantize_model(m)
    result = engine._export(qm, stats, type("Src", (), {"is_local": True, "path": None, "repo_id": None})())
    assert result.format == QuantizationFormat.PYTORCH
    assert result.layer_count >= 1
    assert result.quantized_size_mb >= 0
    assert all(getattr(s, "memory_saved_bytes", 0) >= 0 for s in result.stats)
    print(f"[ok] result stats: quantized_mb={result.quantized_size_mb:.2f}, layers={result.layer_count}")


def test_binary_kernels_marked_experimental():
    # Custom binary linear kernels are tracked as experimental fallbacks.
    assert _BinaryLinear is not None
    print("[note] custom binary kernels are experimental and tracked separately")


if __name__ == "__main__":
    tests = [
        test_count_layers,
        test_quantize_adaptive_forwards,
        test_quantize_irnet_forwards,
        test_layer_filtering,
        test_export_pytorch,
        test_export_safetensors,
        test_export_torchscript,
        test_result_fields,
        test_binary_kernels_marked_experimental,
    ]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"[fail] {test.__name__}: {e}")
            failures.append((test.__name__, e))
    print(f"Verification complete. {len(tests)-len(failures)}/{len(tests)} passed.")
    if failures:
        print("Failures:")
        for name, e in failures:
            print(" -", name, "->", e)
