# BitForge

Production-grade 1-bit quantization toolkit for running large models on mobile and edge devices.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Features

- **Multiple 1-bit algorithms**: adaptive, IRNet, XNOR, binarize, BinaryNet, bilevel
- **Flexible granularity**: per_tensor, per_channel, per_group
- **Export formats**: PyTorch, SafeTensors, TorchScript, ONNX, GGUF
- **Export targets**: desktop, mobile, edge, server
- **GUI + CLI + API**: Tauri 2 desktop shell, command-line interface, FastAPI backend
- **Fine-grained control**: layer filtering, calibration, embedding/norm handling, scale precision, etc.
- **Multi-device**: CPU, CUDA, Metal, Vulkan, OpenCL
- **Custom binary kernels**: experimental packed-weight inference paths

## Installation

```bash
# Clone and install backend with all extras
cd Z:/Projects/BitForge/backend
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[gui,mcp]"

# Install GUI dependencies and build frontend
cd Z:/Projects/BitForge/tauri-gui
npm install
npm run build
```

## Verified Windows Workflow

### Source / dev
```bash
cd Z:/Projects/BitForge
Z:/Projects/BitForge/backend/.venv/Scripts/python.exe -m launcher
```

This opens the BitForge GUI and exposes the REST API on `http://127.0.0.1:8125` by default. If that port is busy, the launcher auto-selects a free localhost port and serves the UI from there.

### Standalone packaged build
```bash
# 1. Build the frontend
cd Z:/Projects/BitForge/tauri-gui
npm install
npm run build

# 2. Build the executable
cd Z:/Projects/BitForge/launcher
Z:/Projects/BitForge/backend/.venv/Scripts/python.exe -m PyInstaller BitForge.spec

# 3. Output artifact
# launcher/dist/BitForge.exe
# or portable zip:
powershell Compress-Archive -Path 'launcher/dist/*' -DestinationPath '../dist/BitForge-portable.zip' -Force
```

Run the packaged app:
```bat
Z:\Projects\BitForge\launcher\dist\BitForge.exe
```

If you prefer the browser-based dev workflow:

```bash
# Terminal 1: backend
cd Z:/Projects/BitForge/backend
Z:/Projects/BitForge/backend/.venv/Scripts/python.exe -m bitforge.run_api

# Terminal 2: frontend
cd Z:/Projects/BitForge/tauri-gui
npm run dev
# Open http://localhost:5173
```

## Controlling BitForge

BitForge can be controlled through several interfaces:

### CLI

```bash
cd Z:/Projects/BitForge/backend
.venv\Scripts\activate
bitforge quantize --model-path ./model.pt --algorithm adaptive --format gguf
```

Subcommands:
- `quantize` — quantize a model
- `mcp` — start the MCP server
- `api` — start the REST API server on port 8125

### REST API

```bash
cd Z:/Projects/BitForge/backend
.venv\Scripts\activate
python -m bitforge.run_api
# or use the unified CLI:
bitforge api --port 8125
```

The API exposes health, quantization, job management, and algorithm discovery endpoints.

### MCP server

```bash
cd Z:/Projects/BitForge/backend
.venv\Scripts\activate
# Ensure the MCP extra is installed:
pip install -e ".[mcp]"
bitforge mcp
```

See `docs/MCP.md` for Hermes, Claude Desktop, and VS Code configuration.

### Standalone desktop GUI (pywebview)

```bash
cd Z:/Projects/BitForge
.venv\Scripts\activate
python -m launcher
# or: python launcher/app.py
```

This starts an embedded FastAPI backend and serves the built Tauri frontend from `tauri-gui/dist/` in one window with no manual servers or external browsers required.

### Native Tauri desktop

Requires Evergreen WebView2 runtime.

```bat
Z:\Projects\BitForge\scripts\run_tauri_native.bat
```

## Quick Start

### GUI

```bash
cd Z:/Projects/BitForge/tauri-gui
npm run dev
```

Windows:
```bat
Z:\Projects\BitForge\scripts\start.bat
```

### CLI

```bash
cd Z:/Projects/BitForge/backend
.venv\Scripts\activate
python -m bitforge.cli quantize --model path/to/model.pt --algorithm adaptive --format pytorch
```

### API

```bash
cd Z:/Projects/BitForge/backend
.venv\Scripts\activate
python -m bitforge.run_api
# Server starts at http://127.0.0.1:8125
# Override port with: set BITFORGE_BACKEND_PORT=8126 && python -m bitforge.run_api
```

## Supported Formats

| Format | Extension | Mobile Runtime |
|--------|-----------|----------------|
| PyTorch | `.pt` | PyTorch Mobile |
| SafeTensors | `.safetensors` | SafeTensors Mobile |
| TorchScript | `.torchscript` | PyTorch Mobile |
| ONNX | `.onnx` | ONNX Runtime Mobile |
| GGUF | `.gguf` | llama.cpp, MLC LLM |

## Quantization Options

- **Algorithm**: `adaptive`, `irnet`, `xnor`, `binarize`, `binarynet`, `bilevel`
- **Granularity**: `per_tensor`, `per_channel`, `per_group`
- **Group size**: 32, 64, 128, etc. (for per_group)
- **Calibration**: enable/disable, number of batches
- **Layer filtering**: include/exclude by name or pattern
- **Scale precision**: `float32`, `float16`, `bfloat16`
- **Binary precision**: `int1`, `bool`
- **Straight-Through Estimator**: enable/disable for gradient flow
- **Stochastic rounding**: enable/disable

## Mobile/Edge Deployment

### ONNX Runtime Mobile

1. Quantize model to ONNX format
2. Install ONNX Runtime Mobile in your app
3. Load and run the `.onnx` model

```java
// Android example
ort.InferenceSession session = new ort.InferenceSession(modelPath);
```

### GGUF / llama.cpp

1. Quantize model to GGUF format
2. Convert using `llama.cpp` tools
3. Run on mobile with llama.cpp bindings

```bash
# Convert with llama.cpp
llama-quantize model.gguf model-quantized.gguf
```

### PyTorch Mobile

1. Quantize and export as PyTorch or TorchScript
2. Include in Android/iOS app with PyTorch Mobile
3. Load and run inference

```java
// Android example
Module module = Module.load(modelPath);
Tensor output = module.forward(input);
```

## Benchmarking

Run the built-in benchmarks:

```bash
cd Z:/Projects/BitForge/backend
.venv\Scripts\activate
python -m bitforge.cli benchmark --model path/to/model.pt --algorithm adaptive
```

Benchmark output includes:
- Layer-by-layer quantization stats
- Memory savings per layer
- Compression ratio
- Inference latency
- Accuracy metrics (if validation data provided)

## Accuracy Validation

Provide a calibration dataset during quantization:

```bash
python -m bitforge.cli quantize \
  --model model.pt \
  --algorithm adaptive \
  --calibrate \
  --calibration-batches 100 \
  --accuracy-threshold 0.95
```

The engine will:
1. Run calibration batches to collect activation statistics
2. Quantize the model
3. Validate accuracy against the threshold
4. Report per-layer error metrics

## Project Structure

```
Z:/Projects/BitForge/
├── backend/
│   ├── bitforge/
│   │   ├── engine/      # Quantization algorithms and kernels
│   │   ├── api/         # FastAPI backend
│   │   ├── cli.py       # CLI interface
│   │   └── types.py     # Configuration and result types
│   ├── tests/           # Test suite
│   └── output/          # Quantized model outputs
├── tauri-gui/
│   ├── src/             # React frontend
│   ├── src-tauri/       # Tauri/Rust backend
│   └── package.json     # Frontend dependencies
└── scripts/
    ├── launch_gui.py    # Unified GUI launcher
    ├── start.bat        # Windows startup script
    ├── verify.py        # System verification script
    ├── run_tauri_native.bat  # Native Tauri launch with logging
    └── build_tauri_release.bat  # Release build + artifact discovery
```

## Testing

Run the full backend suite:
```bash
cd Z:/Projects/BitForge/backend
.venv\\Scripts\\activate
python -m pytest tests/ -v
```

Run GUI integration tests:
```bash
cd Z:/Projects/BitForge/backend
.venv\\Scripts\\activate
python -m pytest tests/test_gui_integration.py -v
```

Quick system verification:
```bash
python Z:/Projects/BitForge/scripts/verify.py
```

Verified end-to-end workflow:
1. Start backend: `cd Z:/Projects/BitForge/backend && .venv\Scripts\activate && python -m bitforge.run_api`
2. Start frontend: `cd Z:/Projects/BitForge/tauri-gui && npm run dev`
3. Open browser: `http://localhost:5173`
4. Run verification: `python Z:/Projects/BitForge/scripts/verify.py`

The verification script checks:
- Backend health at `http://127.0.0.1:8125/api/health`
- Frontend availability at `http://localhost:5173`
- A complete quantization job lifecycle

## Scripts

- `scripts/verify.py` — verifies backend health, frontend availability, and runs a test quantization job.
- `scripts/launch_gui.py` — starts the backend if needed, waits for health, then launches the Tauri frontend dev server.
- `scripts/start.bat` — Windows startup script for launching the GUI.
- `scripts/run_tauri_native.bat` — launches the native Tauri desktop binary and logs output to `logs/tauri-native.log`.
- `scripts/build_tauri_release.bat` — runs `cargo tauri build` and prints the NSIS/MSI artifact paths.

## Troubleshooting

### Desktop launch fails with WebView2 resource-in-use error

Restart Edge WebView2 runtime, or use the browser-based launcher:
```bash
python scripts/launch_gui.py
```

### Port 5173 already in use

Another Tauri dev server is running; reuse it or stop the existing process.

### Port 8125 already in use

Another backend instance is running; reuse it or stop the existing process.

### ONNX export fails with shape mismatch

Ensure the model is in eval mode before export:
```python
model.eval()
torch.onnx.export(model, dummy_input, "model.onnx")
```

### GGUF export requires `gguf` package

Install the GGUF writer:
```bash
pip install gguf
```

### Custom binary kernels are experimental

XNOR and binarize custom kernels are marked as experimental fallbacks. Verified inference paths use adaptive/irnet quantization with `_QuantizedLinear`.

## License

MIT
