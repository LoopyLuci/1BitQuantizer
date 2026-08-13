# Mobile Deployment Guide

BitForge exports quantized models for on-device inference. This guide covers the supported formats and how to run them on iOS, Android, and edge runtimes.

## Supported formats

- `gguf` — preferred for mobile; 1-bit or 4-bit packed tensors, compatible with llama.cpp, MLX, and mobile GGUF loaders.
- `safetensors` — safe serialization for frameworks that support custom model loading.
- `pytorch` — for TorchScript Lite or custom mobile pipelines.

## Recommended mobile path: GGUF

1. In BitForge, select:
   - Algorithm: `XNOR` or `Binarize`
   - Format: `GGUF`
   - Target: `Mobile`
   - Device: `Auto` or `CPU`

2. Export the job from the Jobs panel or `/api/export`.

3. Copy the resulting `model.gguf` into your mobile app bundle or download directory.

## iOS / MLX

Use `llama.cpp` or an MLX-compatible loader. Example inference sketch:

```swift
import llama

let path = Bundle.main.url(forResource: "model", withExtension: "gguf")!
let ctx = llama_context_create(path)
let prompt = "Hello from BitForge mobile."
let tokens = ctx.tokenize(prompt)
let result = ctx.generate(tokens)
print(result)
```

## Android

Use `llama.cpp` Android builds or ONNX Runtime Mobile with a converted model.

```kotlin
val model = File(filesDir, "model.gguf")
val ctx = LlamaContext(model)
val output = ctx.generate("Hello from BitForge mobile.")
println(output)
```

## Web / edge runtime

For edge or web inference, use the GGUF file with WebGPU-enabled runtimes if available, or convert to ONNX and use ONNX Runtime Web.

## Notes

- 1-bit GGUF reduces model size and memory bandwidth, which is the main mobile benefit.
- Performance on CPU-only phones varies by algorithm and `group_size`; benchmark with `bench/run_benchmarks.py`.
- For very large models, use `per_group` quantization and a modest `group_size` to retain accuracy.
