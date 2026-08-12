# BitForge MCP Server

This server exposes BitForge quantization workflows through the
[Model Context Protocol](https://modelcontextprotocol.io/).

## Install

```bash
cd backend
.venv/Scripts/python -m pip install mcp
```

## Run

```bash
cd Z:/Projects/BitForge/backend
Z:/Projects/BitForge/backend/.venv/Scripts/python.exe -m bitforge.mcp_server
```

Or use the console script:

```bash
bitforge-mcp
```

## Available tools

- `health_check` - Backend health and CUDA availability
- `list_algorithms` - Supported algorithms, formats, and export targets
- `quantize` - Quantize a model from a local path or Hugging Face repo
- `get_job` - Get status/result of a quantization job
- `list_jobs` - List recent quantization jobs

## Available resources

- `bitforge://job/{job_id}/artifact` - Quantized model artifact path
- `bitforge://job/{job_id}/result` - Job result JSON

## Available prompts

- `quantize-mobile-model` - Quantize a model for mobile with defaults
- `export-quantized-model` - Export an existing job to another format

## MCP client configuration

### Hermes

```yaml
# hermes config.yaml
mcp:
  servers:
    bitforge:
      command: "Z:/Projects/BitForge/backend/.venv/Scripts/python"
      args: ["-m", "bitforge.mcp_server"]
      cwd: "Z:/Projects/BitForge/backend"
```

### Claude Desktop

```json
{
  "mcpServers": {
    "bitforge": {
      "command": "Z:/Projects/BitForge/backend/.venv/Scripts/python",
      "args": ["-m", "bitforge.mcp_server"],
      "cwd": "Z:/Projects/BitForge/backend"
    }
  }
}
```

### VS Code

```json
{
  "mcp": {
    "servers": {
      "bitforge": {
        "command": "Z:/Projects/BitForge/backend/.venv/Scripts/python",
        "args": ["-m", "bitforge.mcp_server"],
        "cwd": "Z:/Projects/BitForge/backend"
      }
    }
  }
}
```

## Example usage

```
tool: health_check
  arguments: {}

tool: list_algorithms
  arguments: {}

tool: quantize
  arguments:
    model: "C:/Models/llama-7b"
    algorithm: "xnor"
    granularity: "per_channel"
    format: "gguf"
    export_target: "mobile"
    device: "auto"

tool: get_job
  arguments:
    job_id: "123e4567-e89b-12d3-a456-426614174000"

tool: list_jobs
  arguments: {}
```

## Alternative control methods

### CLI

```bash
bitforge --model-path ./model.bin --algorithm xnor --format gguf --export-gguf
```

### REST API

```bash
cd backend
.venv/Scripts/python -m bitforge.run_api
# Server on http://127.0.0.1:8125
```

### GUI

```bash
cd Z:/Projects/BitForge
backend/.venv/Scripts/python launcher/app.py
```

### Tauri native desktop

Requires Evergreen WebView2 runtime.

```bash
scripts/run_tauri_native.bat
```
