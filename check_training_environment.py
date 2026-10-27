"""记录环境；可选执行 GPU 张量计算及 ResNet18 前向传播。无训练代码。"""

import argparse
import importlib.metadata
import json
import platform
import struct
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="新 JSON 文件路径，拒绝覆盖")
    parser.add_argument("--gpu", action="store_true", help="运行 GPU 和随机权重模型前向测试")
    parser.add_argument("--pretrained", action="store_true", help="另外下载并测试 ImageNet 预训练权重")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖：{args.output}")
    result = {"python": sys.version, "python_version": platform.python_version(),
              "executable": sys.executable, "architecture": platform.machine(),
              "bits": struct.calcsize("P") * 8, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
              "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
              "training_started": False, "pretrained_weights_used": False}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, errors="replace", timeout=15, check=True)
        result["nvidia_smi"] = gpu.stdout.strip()
        if args.gpu:
            import torch
            import torchvision
            from torchvision.models import ResNet18_Weights, resnet18

            result.update(torch=torch.__version__, torchvision=torchvision.__version__,
                          cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
            if not result["cuda_available"]:
                raise RuntimeError("torch.cuda.is_available() 为 False；不以 CPU 测试冒充 GPU 测试")
            device = torch.device("cuda:0")
            result.update(gpu_name=torch.cuda.get_device_name(0),
                          compute_capability=list(torch.cuda.get_device_capability(0)),
                          compiled_cuda_architectures=torch.cuda.get_arch_list())
            # 简单确定性矩阵乘法，比较 CPU 参考值，确认实际产生了 GPU 结果。
            source = torch.arange(1024, dtype=torch.float32).reshape(32, 32) / 1024
            expected = source @ source.T
            with torch.inference_mode():
                gpu_source = source.to(device)
                actual = gpu_source @ gpu_source.T
                torch.cuda.synchronize()
                error = (actual.cpu() - expected).abs().max().item()
                if not torch.allclose(actual.cpu(), expected, rtol=1e-4, atol=1e-5):
                    raise RuntimeError(f"GPU 矩阵运算结果不匹配，最大误差 {error}")
                result.update(gpu_tensor_test_passed=True, tensor_device=str(actual.device), max_abs_error=error)
                model = resnet18(weights=None).eval().to(device)
                inputs = torch.zeros(1, 3, 224, 224, device=device)
                predictions = model(inputs)
                torch.cuda.synchronize()
                assert list(predictions.shape) == [1, 1000] and torch.isfinite(predictions).all().item()
                result.update(random_resnet18_forward_passed=True, random_output_shape=list(predictions.shape),
                              model_device=str(next(model.parameters()).device))
                del model, predictions
                if args.pretrained:
                    # 下载缓存位于项目 reports 下，不修改系统路径或原始图片。
                    torch.hub.set_dir(str(args.output.resolve().parent / "torch_cache"))
                    weights = ResNet18_Weights.IMAGENET1K_V1
                    model = resnet18(weights=weights).eval().to(device)
                    predictions = model(inputs)
                    torch.cuda.synchronize()
                    assert list(predictions.shape) == [1, 1000] and torch.isfinite(predictions).all().item()
                    result.update(pretrained_weights_used=True, pretrained_weights=str(weights),
                                  pretrained_url=weights.url, pretrained_resnet18_forward_passed=True,
                                  pretrained_output_shape=list(predictions.shape))
            result["gpu_test_passed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
