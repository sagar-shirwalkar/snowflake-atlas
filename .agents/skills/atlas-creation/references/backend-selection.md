# Backend Selection Reference

## Overview

The Atlas embedding backend is resolved at **run time** (not build time) by both `atlas-build` and `atlas-rag`. The same bundle works with any backend — only the inference runtime differs.

## Resolution Algorithm

```python
def resolve_backend(prefer: str | None = None) -> tuple[str, str]:
    """
    Returns (backend, reason) where backend ∈ {"mlx", "onnx-gpu", "onnx-cpu"}
    """
    # Priority order (highest first):
    # 1. CLI argument (--prefer)
    # 2. Environment variable (ATLAS_EMBED_BACKEND)
    # 3. Config file (~/.config/atlas.toml [backend] prefer = "...")
    # 4. Auto-detect
```

### Input Normalization

| CLI / Env / Config Value | Canonical Backend |
|--------------------------|-------------------|
| `apple`, `mlx` | `mlx` |
| `nvidia`, `gpu`, `cuda`, `onnx-gpu` | `onnx-gpu` |
| `cpu`, `onnx-cpu` | `onnx-cpu` |
| `auto`, `""`, `None` | Auto-detect |

### Auto-Detect Logic

```python
def auto_detect() -> tuple[str, str]:
    # 1. Apple Silicon + MLX importable?
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core
            return "mlx", "Apple Silicon detected and MLX is importable"
        except ImportError:
            pass
    
    # 2. NVIDIA GPU + CUDA provider available?
    if has_nvidia_gpu() and has_onnxruntime_gpu():
        return "onnx-gpu", "NVIDIA GPU detected and CUDA provider is available"
    
    # 3. Portable floor
    return "onnx-cpu", "no fast path available; using portable ONNX+CPU"
```

### Probe Functions

```python
def has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and "GPU" in out.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False

def has_onnxruntime_gpu() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False

def has_mlx() -> bool:
    try:
        import mlx.core
        return True
    except ImportError:
        return False
```

## Backend Comparison

| Property | MLX | ONNX-CUDA | ONNX-CPU |
|----------|-----|-----------|----------|
| **Platform** | Apple Silicon (M1/M2/M3/M4) | Linux + NVIDIA GPU | Any (Linux, Windows, Intel Mac) |
| **Dependency** | `mlx>=0.20` | `onnxruntime-gpu>=1.18` | `onnxruntime>=1.18` |
| **Build Speed** | ~5-10× CPU | ~3-5× CPU | Baseline |
| **Query Latency** | 1-2 ms | 1-2 ms | 50-200 ms |
| **Memory** | Unified (ANE/GPU) | VRAM + System | System RAM |
| **Model Source** | Converted `.npy` weights | ONNX (Xenova) | ONNX (Xenova) |
| **Bundle Model** | Not used (uses cache) | `model/onnx/model.onnx` | `model/onnx/model.onnx` |

## Build-Time vs Run-Time Backend

| Phase | Backend Used | Purpose |
|-------|--------------|---------|
| `atlas-build` | Resolved via `--prefer` / auto | Embed all chunks, write `embeddings.f16.npy` |
| `atlas-rag` startup | Resolved via `--prefer` / auto | Load embedder for query encoding |
| `atlas-doctor` | Auto (always) | Diagnostic probe only |

**Critical**: The bundle stores **only embeddings** (float16 numpy array). The backend used at build time is recorded in `manifest.json` for provenance but does **not** constrain the backend at run time.

## Override Examples

```bash
# Force ONNX+CPU (reproducible CI builds)
atlas-build --prefer cpu --output ./bundle
ATLAS_EMBED_BACKEND=cpu atlas-rag --bundle ./bundle

# Force MLX (Apple Silicon, even if auto would pick it)
atlas-build --prefer apple --output ./bundle
atlas-rag --prefer mlx --bundle ./bundle

# Force CUDA (Linux + NVIDIA, requires onnxruntime-gpu)
atlas-build --prefer nvidia --output ./bundle
atlas-rag --prefer cuda --bundle ./bundle

# Config file (~/.config/atlas.toml)
[backend]
prefer = "nvidia"
```

## Fallback Behavior

| Requested | Available? | Actual | Log Message |
|-----------|------------|--------|-------------|
| `mlx` | ✓ MLX + Apple Silicon | `mlx` | "user override (or auto) and MLX available on Apple Silicon" |
| `mlx` | ✗ MLX or not Apple Silicon | `onnx-cpu` | "MLX requested but not importable; using ONNX+CPU fallback" |
| `onnx-gpu` | ✓ CUDA provider + GPU | `onnx-gpu` | "user override and CUDA provider available" |
| `onnx-gpu` | ✗ No GPU or no CUDA provider | `onnx-cpu` | "CUDA requested but no NVIDIA GPU; using ONNX+CPU fallback" |
| `onnx-cpu` | Always | `onnx-cpu` | "user requested ONNX+CPU" |
| `auto` | MLX available | `mlx` | "Apple Silicon detected and MLX is importable" |
| `auto` | No MLX, CUDA available | `onnx-gpu` | "NVIDIA GPU detected and CUDA provider is available" |
| `auto` | Neither | `onnx-cpu` | "no fast path available; using portable ONNX+CPU" |

## Embedder Factory

```python
def get_embedder(model_id: str | Path, prefer: str | None = None) -> Embedder:
    backend, _reason = resolve_backend(prefer)
    
    if backend == "mlx":
        from .mlx import MlxEmbedder
        return MlxEmbedder(model_id)
    
    if backend == "onnx-gpu":
        from .onnx import OnnxEmbedder
        return OnnxEmbedder(model_id, prefer_gpu=True)
    
    from .onnx import OnnxEmbedder
    return OnnxEmbedder(model_id, prefer_gpu=False)
```

## Model Resolution at Runtime

```python
# In rag_server.py Bundle.__init__
bundled_model = self.bundle_dir / "model" / "onnx" / "model.onnx"
if bundled_model.is_file():
    model_id: str | Path = self.bundle_dir / "model"  # Local bundled ONNX
else:
    model_id = DEFAULT_MODEL_ID  # "Xenova/bge-base-en-v1.5" from HF cache

try:
    self.embedder = get_embedder(model_id, prefer=prefer)
except Exception as e:
    logger.warning("Preferred embedder failed, falling back to ONNX+CPU", error=str(e))
    self.embedder = get_embedder(DEFAULT_MODEL_ID, prefer="cpu")
```

## MLX Weight Conversion (One-Time)

MLX cannot load ONNX directly. Weights must be converted from PyTorch:

```bash
# Run once on Apple Silicon machine with PyTorch + MLX
cd tools/
python convert_bge_to_mlx.py \
    --model BAAI/bge-base-en-v1.5 \
    --output ~/.cache/atlas/mlx/bge-base-en-v1.5
```

This creates `.npy` weight files that `MlxEmbedder` loads. The conversion script:
1. Loads `BAAI/bge-base-en-v1.5` from HF (PyTorch)
2. Converts to MLX format (fp16)
3. Saves to cache dir with expected naming

## CI/CD Considerations

```yaml
# .github/workflows/build-bundle.yml
jobs:
  smoke-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Sync deps (CPU only)
        run: uv sync --extra build
      - name: Smoke test
        run: atlas-smoke --bundle ./test-bundle --repo ./test-repo
        # Uses ONNX-CPU (no GPU in GitHub Actions)
  
  # Build bundle locally, not in CI
  # maintainer runs: atlas-build --prefer auto --output ./data/rag-bundle
  # then: scripts/publish-bundle.sh
```

## Troubleshooting Backend Selection

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Slow queries on M-series | `atlas-doctor` shows `onnx-cpu` | `uv sync --extra mlx`; verify `mlx` importable |
| CUDA not detected | `atlas-doctor` shows `onnx-cpu` reason "no NVIDIA GPU" | Install `onnxruntime-gpu`; check `nvidia-smi` |
| MLX import fails | `atlas-doctor` shows MLX MISS | `uv sync --extra mlx`; Python 3.11+ required |
| Wrong backend in CI | Build used MLX but CI has no Apple Silicon | Build with `--prefer cpu` for portable bundles |
| Model load error | "Preferred embedder failed, falling back" | Check `manifest.json` has `model/onnx/model.onnx` |

## Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `ATLAS_EMBED_BACKEND` | `auto`, `apple`, `mlx`, `nvidia`, `gpu`, `cuda`, `cpu`, `onnx-cpu` | Overrides CLI `--prefer` |
| `ATLAS_MODEL_CACHE` | Path | Overrides HF cache dir for model downloads |