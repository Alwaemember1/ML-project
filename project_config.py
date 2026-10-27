"""不依赖PyTorch的路径与本地配置，便于先诊断再安装环境。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FROZEN_SHA256 = "1c464873a01b10e1dce89211859a0a9bfe1448a765ecfaef4ed57089436c6b2f"
RUNTIME = Path(os.environ.get("LEAF_RUNTIME_DIR", str(ROOT / ".venv/smoke_tmp"))).expanduser().resolve()


def default_record_path():
    explicit = os.environ.get("LEAF_MODEL_RECORD")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else ROOT / path
    local = ROOT / "FINAL_MODEL.local.json"
    return local if local.is_file() else ROOT / "FINAL_MODEL.json"


def resolve_record_file(value, record_path):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(record_path).resolve().parent / path


def configure_runtime():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR", "XDG_CACHE_HOME"):
        os.environ[key] = str(RUNTIME)
    cuda_cache = RUNTIME / "cuda_cache"
    cuda_cache.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_CACHE_PATH"] = str(cuda_cache)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
