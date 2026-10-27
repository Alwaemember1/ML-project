"""只读诊断Python、依赖、本地权重和结果文件；不下载、不训练、不读取数据集。"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys

from project_config import ROOT, FROZEN_SHA256, default_record_path, resolve_record_file


def diagnose(check_gpu=False):
    checks, versions = [], {}
    for name in ("Pillow", "torch", "torchvision", "streamlit"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            checks.append(f"缺少{name}，请按README安装依赖")
    record_path = default_record_path()
    model_hash = None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["checkpoint_sha256"] != FROZEN_SHA256 or record["status"] != "frozen_final_model":
            raise ValueError("冻结记录不匹配")
        checkpoint = resolve_record_file(record["checkpoint"], record_path)
        with checkpoint.open("rb") as file:
            model_hash = hashlib.file_digest(file, "sha256").hexdigest()
        if model_hash != FROZEN_SHA256:
            raise ValueError("模型文件SHA256不符")
        metrics = json.loads(resolve_record_file(record["evaluation_metrics"], record_path).read_text(encoding="utf-8"))
        if metrics["total"] != 9192 or len(metrics["classes"]) != 39:
            raise ValueError("指标文件不匹配")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        checks.append(f"模型/指标配置不可用：{exc}；先取得权重，再执行python -m scripts.configure_model --checkpoint <文件位置>")
    gpu = "未检测；加--gpu可查询，CPU演示不要求GPU"
    if check_gpu:
        try:
            import torch
            gpu = {"cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        except (ImportError, OSError, RuntimeError) as exc:
            checks.append(f"PyTorch加载失败：{exc}")
    return {"ok": not checks, "python": platform.python_version(), "executable": sys.executable,
            "project": str(ROOT), "configuration": str(record_path), "dependencies": versions,
            "model_sha256": model_hash, "gpu": gpu, "problems": checks,
            "training_or_test_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="另外导入torch查询实际CUDA状态，不训练或下载")
    args = parser.parse_args()
    report = diagnose(args.gpu)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
