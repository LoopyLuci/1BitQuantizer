"""Real quantization engine with multiple algorithms and multi-device support."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from bitforge.types import Device, ModelSource, QuantizationConfig, QuantizationFormat, QuantizationResult

# Reuse lightweight internal helpers without changing engine behavior.
from bitforge.types import LayerStats


class _LayerStatsCollector:
    def __init__(self):
        self.stats: List[LayerStats] = []

    def add(self, name, original, original_weight, quantized_weight, scale):
        self.stats.append(
            LayerStats(
                layer_name=name,
                layer_type=type(original).__name__,
                original_shape=tuple(original_weight.shape),
                quantized_shape=tuple(quantized_weight.shape),
                original_params=int(original_weight.numel()),
                quantized_params=int(quantized_weight.numel()),
                scale_mean=float(getattr(scale, "mean", lambda: scale)().item() if hasattr(scale, "mean") else scale),
                scale_std=0.0,
            )
        )


class _XNORLinear(nn.Module):
    def __init__(self, in_features, out_features, group_size=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("packed_weight", torch.empty(out_features, in_features, dtype=torch.bool))
        self.register_buffer("scale", torch.empty(out_features, 1))

    @classmethod
    def from_linear(cls, module: nn.Linear, group_size=32):
        layer = cls(module.in_features, module.out_features, group_size)
        w = module.weight.detach()
        layer.scale.data = w.abs().mean(dim=1, keepdim=True)
        layer.packed_weight.data = w.sign() > 0
        return layer

    def forward(self, x):
        w = self.packed_weight.float() if hasattr(self, "packed_weight") else self.weight
        return torch.nn.functional.linear(x, w, self.bias)


class _BinarizeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("packed_weight", torch.empty(out_features, in_features, dtype=torch.bool))
        self.register_buffer("scale", torch.empty(out_features, 1))

    @classmethod
    def from_linear(cls, module: nn.Linear):
        layer = cls(module.in_features, module.out_features)
        w = module.weight.detach()
        layer.scale.data = w.abs().mean(dim=1, keepdim=True)
        layer.packed_weight.data = w.sign() > 0
        return layer

    def forward(self, x):
        w = self.packed_weight.float() if hasattr(self, "packed_weight") else self.weight
        return torch.nn.functional.linear(x, w, self.bias)


class _QuantizedLinear(nn.Module):
    def __init__(self, original: nn.Linear, packed_weight, scale):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.packed_weight = packed_weight
        self.scale = scale
        if original.bias is not None:
            self.bias = original.bias.clone()
        else:
            self.register_buffer("bias", torch.zeros(self.out_features))

    def forward(self, x):
        w = self.packed_weight.float() if hasattr(self, "packed_weight") else self.weight
        return torch.nn.functional.linear(x, w, self.bias)


# Backward-compatible alias used by existing tests.
_BinaryLinear = _QuantizedLinear


def count_layers(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d)))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class QuantizationEngine:
    """Real quantization engine with multiple algorithms and multi-device support."""

    def __init__(self, config: QuantizationConfig, stats_collector=None):
        self.config = config
        self.stats_collector = stats_collector or _LayerStatsCollector()
        self._device = self._resolve_device(config.device)

    @staticmethod
    def _parse_hf_model(model: str):
        """Accept repo IDs or full Hugging Face links and normalize them."""
        text = model.strip()
        if "://" in text:
            text = text.split("://", 1)[1]
        text = text.strip("/")
        revision = "main"
        if "#" in text:
            text, revision = text.split("#", 1)
        if "/" in text:
            host, _, path = text.partition("/")
            if "." in host or host == "huggingface.co":
                text = path
        if not text or "/" not in text:
            raise ValueError(f"Invalid Hugging Face model identifier: {model!r}")
        return text, revision

    @staticmethod
    def _is_hf_model(model: str) -> bool:
        if not model:
            return False
        text = model.strip()
        if text.startswith(("https://huggingface.co/", "http://huggingface.co/", "www.")):
            return True
        if text.startswith("huggingface.co/"):
            return True
        if "/" in text and not Path(text).exists():
            return True
        return False

    def _resolve_device(self, device: Device) -> torch.device:
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize_model(self, model: nn.Module) -> Tuple[nn.Module, List[LayerStats]]:
        model = model.to(self._device)
        if self.config.granularity == "per_group" and self.config.group_size <= 0:
            raise ValueError("group_size must be > 0 for per_group quantization")
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and self._should_quantize_layer(name):
                self._replace_linear(model, name, module)
        return model, self.stats_collector.stats

    def quantize_from_source(self, source: ModelSource, progress_callback=None) -> QuantizationResult:
        started = time.perf_counter()
        try:
            model = self._load_model(source)
            model, stats = self.quantize_model(model)
            result = self._export(model, stats, source)
            result.elapsed_seconds = time.perf_counter() - started
            return result
        except Exception as exc:
            return QuantizationResult(
                output_path=Path("."),
                format=self.config.format,
                original_size_mb=0.0,
                quantized_size_mb=0.0,
                compression_ratio=1.0,
                layer_count=0,
                success=False,
                error=str(exc),
                elapsed_seconds=time.perf_counter() - started,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _should_quantize_layer(self, name: str) -> bool:
        if self.config.layer_exclude_pattern and self.config.layer_exclude_pattern in name:
            return False
        if self.config.layer_include_pattern and self.config.layer_include_pattern not in name:
            return False
        if self.config.layer_exclude_names and any(n.lower() in name.lower() for n in self.config.layer_exclude_names):
            return False
        if self.config.layer_include_names and not any(n.lower() in name.lower() for n in self.config.layer_include_names):
            return False
        if not self.config.quantize_embeddings and ("embed" in name.lower() or "embeddings" in name.lower()):
            return False
        if not self.config.quantize_final_norm and ("norm" in name.lower() or "ln_" in name.lower()):
            return False
        if not self.config.quantize_first_linear and "layer.0.attention.dense" in name.lower():
            return False
        if not self.config.quantize_last_linear and "lm_head" in name.lower():
            return False
        return True

    def _replace_linear(self, model: nn.Module, name: str, module: nn.Linear) -> None:
        alg = self.config.algorithm
        if alg == "xnor":
            q = _XNORLinear.from_linear(module, group_size=self.config.group_size)
        elif alg == "binarize":
            q = _BinarizeLinear.from_linear(module)
        else:
            w = module.weight.detach()
            out_f, in_f = w.shape
            scale = w.abs().mean(dim=1, keepdim=True)
            packed = w.sign()
            if self.config.granularity == "per_group":
                scale = scale.reshape(out_f, -1, self.config.group_size).mean(dim=-1).unsqueeze(-1)
            q = _QuantizedLinear(module, packed, scale)

        self.stats_collector.add(name, module, module.weight, q.packed_weight if hasattr(q, "packed_weight") else module.weight, q.scale)
        self._set_attr(model, name, q)

    def _set_attr(self, root: nn.Module, name: str, value: nn.Module) -> None:
        parts = name.split(".")
        for p in parts[:-1]:
            root = getattr(root, p)
        setattr(root, parts[-1], value)

    def _load_model(self, source: ModelSource) -> nn.Module:
        if source.is_local:
            if source.path and Path(source.path).is_dir():
                from transformers import AutoModel
                return AutoModel.from_pretrained(source.path)
            else:
                import torch
                try:
                    return torch.load(source.path, map_location=self._device, weights_only=False)
                except TypeError:
                    return torch.load(source.path, map_location=self._device)
        else:
            if not source.repo_id:
                raise ValueError("repo_id is required for hub models")
            from huggingface_hub import HfApi
            api = HfApi()
            path = api.snapshot_download(repo_id=source.repo_id, revision=source.revision)
            from transformers import AutoModel
            return AutoModel.from_pretrained(path)

    def _export(self, model: nn.Module, stats: List[LayerStats], source: ModelSource) -> QuantizationResult:
        fmt = self.config.format
        output_dir = Path("output") / f"quantized_{int(time.time())}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"model.{fmt.value}"

        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        orig_est_mb = model_size_mb * 8

        if fmt == QuantizationFormat.PYTORCH:
            model.eval()
            torch.save(model.state_dict(), output_path)
        elif fmt == QuantizationFormat.SAFETENSORS:
            model.eval()
            from safetensors.torch import save_file
            save_file(model.state_dict(), str(output_path))
        elif fmt == QuantizationFormat.TORCHSCRIPT:
            scripted = torch.jit.script(model.eval())
            scripted.save(str(output_path))
        elif fmt == QuantizationFormat.ONNX:
            dummy = _build_dummy_input(model, self._device)
            torch.onnx.export(model, dummy, str(output_path))
        elif fmt == QuantizationFormat.GGUF:
            self._export_gguf(model, output_path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        qsize_mb = output_path.stat().st_size / (1024 * 1024)
        return QuantizationResult(
            output_path=output_path,
            format=fmt,
            original_size_mb=orig_est_mb,
            quantized_size_mb=qsize_mb,
            compression_ratio=orig_est_mb / max(1e-6, qsize_mb),
            layer_count=len(stats),
            stats=stats,
            metadata={
                "algorithm": self.config.algorithm,
                "granularity": self.config.granularity,
                "group_size": self.config.group_size,
                "device": str(self._device),
                "preserve_embeddings": self.config.quantize_embeddings,
                "export_target": self.config.export_target.value,
            },
        )

    def _export_gguf(self, model: nn.Module, path: Path) -> None:
        try:
            from gguf import GGUFWriter
        except Exception:
            raise RuntimeError("Install 'gguf' to enable GGUF export: pip install gguf")
        writer = GGUFWriter(str(path), "llama")
        tensors: Dict[str, "np.ndarray"] = {}
        for name, param in model.state_dict().items():
            arr = param.detach().cpu().numpy()
            tensors[name] = arr
            writer.add_tensor(name, arr)
        writer.add_quantization_version(2)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()


def _build_dummy_input(model: nn.Module, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Build dummy input for export using the first linear layer's in_features."""
    in_features = None

    def _find_in_features(m: nn.Module) -> Optional[int]:
        if hasattr(m, "in_features") and isinstance(getattr(m, "in_features"), int):
            return m.in_features
        for child in m.children():
            val = _find_in_features(child)
            if val is not None:
                return val
        return None

    in_features = _find_in_features(model)
    if in_features is None:
        in_features = 1
    return (torch.randn(1, in_features, device=device),)
