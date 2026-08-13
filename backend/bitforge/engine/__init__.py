"""Real quantization engine with multiple algorithms and multi-device support."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from bitforge.types import Device, ModelSource, QuantizationConfig, QuantizationFormat, QuantizationResult

# Reuse lightweight internal helpers without changing engine behavior.
from bitforge.types import LayerStats


class _LayerStatsCollector:
    def __init__(self):
        self.stats: List[LayerStats] = []

    def add(self, name, original, original_weight, quantized_weight, scale):
        original_bytes = int(original_weight.numel()) * original_weight.element_size()
        if hasattr(quantized_weight, "packed_weight"):
            pw = quantized_weight.packed_weight
            q_bytes = pw.numel() // 8 if pw.dtype in (torch.bool, torch.uint8) else pw.numel()
        elif hasattr(quantized_weight, "dtype") and quantized_weight.dtype in (torch.bool, torch.uint8):
            q_bytes = quantized_weight.numel() // 8
        else:
            q_bytes = int(quantized_weight.numel()) * quantized_weight.element_size()

        self.stats.append(
            LayerStats(
                layer_name=name,
                layer_type=type(original).__name__,
                original_shape=tuple(original_weight.shape),
                quantized_shape=tuple(original_weight.shape),
                original_params=int(original_weight.numel()),
                quantized_params=max(1, q_bytes),
                scale_mean=float(getattr(scale, "mean", lambda: scale)().item() if hasattr(scale, "mean") else scale),
                scale_std=0.0,
                memory_saved_bytes=max(0, original_bytes - q_bytes),
            )
        )


class _CalibrationStats:
    def __init__(self):
        self.activation_scales: Dict[str, float] = {}

    def record(self, name: str, tensor: torch.Tensor) -> None:
        self.activation_scales[name] = float(tensor.abs().mean().item())


class _XNORLinear(nn.Module):
    def __init__(self, in_features, out_features, group_size=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("packed_weight", torch.empty(out_features, in_features, dtype=torch.uint8))
        self.register_buffer("scale", torch.empty(out_features, 1))

    @classmethod
    def from_linear(cls, module: nn.Linear, group_size=32, activation_scales=None):
        layer = cls(module.in_features, module.out_features, group_size)
        w = module.weight.detach().sign() > 0
        packed = torch.zeros_like(layer.packed_weight)
        bits = 8
        out_f, in_f = w.shape
        for i in range(out_f):
            packed_w = 0
            for j in range(0, in_f, bits):
                chunk = w[i, j:j+bits]
                byte_val = 0
                for b, bit in enumerate(chunk):
                    if bit:
                        byte_val |= 1 << (7 - b)
                packed[i, j // bits] = byte_val
        layer.packed_weight = packed
        groups = max(1, in_f // group_size)
        scale = module.weight.detach().abs().reshape(out_f, groups, group_size).mean(dim=(1, 2))
        layer.scale = scale.reshape(out_f, 1)
        return layer

    def forward(self, x):
        if x.device.type == 'cpu':
            try:
                return self._forward_cpu(x)
            except Exception:
                pass
        return torch.nn.functional.linear(x, self.packed_weight.float(), self.scale.t().flatten() if self.scale.ndim == 2 else self.scale.flatten())

    def _forward_cpu(self, x):
        if self.packed_weight.dtype != torch.uint8:
            return torch.nn.functional.linear(x, self.packed_weight.float(), self.scale.t().flatten() if self.scale.ndim == 2 else self.scale.flatten())
        w = self.packed_weight.to(x.device)
        groups = max(1, self.in_features // self.group_size)
        x_bin = (x > 0).to(w.dtype)
        x_packed = torch.packbits(x_bin.unsqueeze(-1), dim=-1).squeeze(-1)
        x_packed = x_packed.reshape(x.shape[0], groups, self.group_size // 8)
        w = w.reshape(1, self.out_features, groups, self.group_size // 8)
        xor = torch.bitwise_xor(x_packed.unsqueeze(2), w)
        matches = self.group_size - xor.sum(dim=-1)
        signed = matches * 2 - self.group_size
        return (signed * self.scale.t().reshape(1, self.out_features, 1)).sum(dim=-1)


class _BinarizeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("packed_weight", torch.empty(out_features, in_features, dtype=torch.uint8))
        self.register_buffer("scale", torch.empty(out_features, 1))

    @classmethod
    def from_linear(cls, module: nn.Linear, activation_scales=None, group_size=None):
        layer = cls(module.in_features, module.out_features)
        w = module.weight.detach().sign() > 0
        packed = torch.zeros_like(layer.packed_weight)
        bits = 8
        out_f, in_f = w.shape
        for i in range(out_f):
            packed_w = 0
            for j in range(0, in_f, bits):
                chunk = w[i, j:j+bits]
                byte_val = 0
                for b, bit in enumerate(chunk):
                    if bit:
                        byte_val |= 1 << (7 - b)
                packed[i, j // bits] = byte_val
        layer.packed_weight = packed
        layer.scale = module.weight.detach().abs().mean(dim=1, keepdim=True)
        return layer

    def forward(self, x):
        if x.device.type == 'cpu':
            try:
                return self._forward_cpu(x)
            except Exception:
                pass
        return torch.nn.functional.linear(x, self.packed_weight.float(), self.scale.t().flatten() if self.scale.ndim == 2 else self.scale.flatten())

    def _forward_cpu(self, x):
        if self.packed_weight.dtype != torch.uint8:
            return torch.nn.functional.linear(x, self.packed_weight.float(), self.scale.t().flatten() if self.scale.ndim == 2 else self.scale.flatten())
        w = self.packed_weight.to(x.device)
        x_bin = (x > 0).to(w.dtype)
        x_packed = torch.packbits(x_bin.unsqueeze(-1), dim=-1).squeeze(-1)
        xor = torch.bitwise_xor(x_packed.unsqueeze(2), w.unsqueeze(0))
        mismatches = xor.sum(dim=-1)
        matches = (self.in_features - mismatches)
        signed = matches * 2 - self.in_features
        return (signed / float(self.in_features) * self.scale.t().reshape(1, self.out_features, 1)).sum(dim=-1)


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
        return torch.nn.functional.linear(x, self.packed_weight.float(), self.bias)


# Backward-compatible alias used by existing tests.
_BinaryLinear = _QuantizedLinear


def count_layers(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d)))


class QuantizationEngine:
    """Real quantization engine with multiple algorithms and multi-device support."""

    def __init__(self, config: QuantizationConfig, stats_collector=None):
        self.config = config
        self.stats_collector = stats_collector or _LayerStatsCollector()
        self._device = self._resolve_device(config.device)
        self._activation_scales: Dict[str, float] = {}

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

    def _run_calibration(self, model: nn.Module, loader) -> None:
        model.eval()
        cal = _CalibrationStats()

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    cal.record(name, output)
            return hook

        handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= self.config.calibration_batches:
                    break
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                model(batch.to(self._device))

        for h in handles:
            h.remove()

        self._activation_scales = cal.activation_scales

    def quantize_model(self, model: nn.Module, calibration_loader=None) -> Tuple[nn.Module, List[LayerStats]]:
        if self.config.calibrate and calibration_loader is not None:
            self._run_calibration(model, calibration_loader)

        model = model.to(self._device)
        if self.config.granularity == "per_group" and self.config.group_size <= 0:
            raise ValueError("group_size must be > 0 for per_group quantization")

        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and self._should_quantize_layer(name):
                self._replace_linear(model, name, module)

        return model, self.stats_collector.stats

    def _replace_linear(self, model: nn.Module, name: str, module: nn.Linear) -> None:
        alg = self.config.algorithm
        activation_scales = getattr(self, "__activation_scales", self._activation_scales)

        if alg == "xnor":
            q = _XNORLinear.from_linear(module, group_size=self.config.group_size, activation_scales=activation_scales)
        elif alg == "binarize":
            q = _BinarizeLinear.from_linear(module, activation_scales=activation_scales)
        else:
            w = module.weight.detach()
            out_f, in_f = w.shape
            scale = w.abs().mean(dim=1, keepdim=True)
            packed = w.sign()
            if self.config.granularity == "per_group":
                scale = scale.reshape(out_f, -1, self.config.group_size).mean(dim=-1).unsqueeze(-1)
            q = _QuantizedLinear(module, packed, scale)

        act_scale = activation_scales.get(name)
        if act_scale is not None and hasattr(q, "scale"):
            q.scale.data = q.scale.data * act_scale

        weight_for_stats = q.packed_weight if hasattr(q, "packed_weight") else module.weight
        self.stats_collector.add(name, module, module.weight, weight_for_stats, q.scale)
        self._set_attr(model, name, q)

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
                if source.path and Path(source.path).exists():
                    if source.path.lower().endswith((".safetensors", ".safetensor", ".st")) or Path(source.path).stat().st_size > 200 * 1024 * 1024:
                        try:
                            from safetensors.torch import load_file
                            state = load_file(source.path, device=str(self._device))
                            loaded = torch.nn.Linear(1, 1)
                            loaded.load_state_dict(state, strict=False)
                            return loaded
                        except Exception:
                            pass
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
        orig_est_mb = model_size_mb
        quant_est_mb = model_size_mb / 8 if self.config.algorithm in {"xnor", "binarize", "binarynet", "bilevel"} else model_size_mb

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
            model.eval()
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

    def _quantize_tensor_1bit(self, tensor: torch.Tensor) -> torch.Tensor:
        w = tensor.detach().cpu().sign() > 0
        packed = torch.zeros(tensor.numel() // 8 + (tensor.numel() % 8 > 0), dtype=torch.uint8)
        flat = w.flatten()
        for i in range(0, len(flat), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(flat) and flat[i + j]:
                    byte_val |= 1 << j
            packed[i // 8] = byte_val
        return packed

    def _quantize_tensor_4bit(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Min-max quantization to 4-bit unsigned."""
        t_min = float(tensor.min().item())
        t_max = float(tensor.max().item())
        scale = (t_max - t_min) / 15.0
        if scale <= 0:
            return torch.zeros_like(tensor, dtype=torch.uint8), t_min, 0.0
        q = ((tensor - t_min) / scale).round().clamp(0, 15).to(torch.uint8)
        return q, t_min, scale

    def _export_gguf(self, model: nn.Module, path: Path) -> None:
        try:
            from gguf import GGUFWriter
        except ImportError:
            raise RuntimeError("Install 'gguf' to enable GGUF export: pip install gguf")

        writer = GGUFWriter(str(path), "llama")
        state = model.state_dict()
        use_1bit = self.config.algorithm in {"xnor", "binarize", "binarynet", "bilevel"}

        for name, param in state.items():
            tensor = param.detach().cpu()
            if use_1bit and tensor.numel() > 1000 and tensor.ndim >= 2:
                q = self._quantize_tensor_1bit(tensor)
                writer.add_tensor(name, q.numpy())
            elif tensor.numel() > 1000 and tensor.ndim >= 2:
                q, _t_min, _scale = self._quantize_tensor_4bit(tensor)
                writer.add_tensor(name, q.numpy())
            else:
                writer.add_tensor(name, tensor.numpy())

        writer.add_quantization_version(2)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def quantize_from_source(self, source: ModelSource, progress_callback=None) -> QuantizationResult:
        started = time.perf_counter()
        try:
            model = self._load_model(source)
            calibration_loader = getattr(source, "calibration_loader", None)
            model, stats = self.quantize_model(model, calibration_loader=calibration_loader)
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
