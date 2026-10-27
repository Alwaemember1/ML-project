"""smoke 训练/独立预测共用的模型、确定性预处理和 D 盘缓存设置。"""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".venv" / "smoke_tmp"
RUNTIME.mkdir(parents=True, exist_ok=True)
for key in ("TEMP", "TMP", "TMPDIR", "XDG_CACHE_HOME"):
    os.environ[key] = str(RUNTIME)
os.environ["CUDA_CACHE_PATH"] = str(RUNTIME / "cuda_cache")
Path(os.environ["CUDA_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

LIMITATION = "仅用于验证训练程序；增强同源关系未排除，验证指标不作为最终泛化评价。"


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def sha256_file(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def seed_torch(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def preprocessing_info():
    preset = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return {"weights": "ResNet18_Weights.IMAGENET1K_V1", "rgb": True,
            "exif_transpose": True, "resize_size": preset.resize_size[0],
            "crop_size": preset.crop_size[0], "mean": preset.mean, "std": preset.std,
            "interpolation": preset.interpolation.value, "antialias": preset.antialias,
            "random_augmentation": False, "pixel_scale": "uint8 / 255 -> float32"}


def build_transform(info):
    return transforms.Compose([
        transforms.Resize(info["resize_size"], interpolation=transforms.InterpolationMode(info["interpolation"]),
                          antialias=info["antialias"]),
        transforms.CenterCrop(info["crop_size"]),
        transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(info["mean"], info["std"]),
    ])


def build_training_transform(info):
    """正式训练使用适度裁剪/翻转；复用同一套张量转换与归一化。"""
    deterministic = build_transform(info)
    return transforms.Compose([
        transforms.RandomResizedCrop(info["crop_size"], scale=(0.8, 1.0), ratio=(0.9, 1.1),
                                     interpolation=transforms.InterpolationMode(info["interpolation"]),
                                     antialias=info["antialias"]),
        transforms.RandomHorizontalFlip(p=0.5),
        *deterministic.transforms[2:],
    ])


def pretrained_classifier(cache, class_count):
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(cache, map_location="cpu", weights_only=True), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.fc = torch.nn.Linear(model.fc.in_features, class_count)
    return model


def image_tensor(path, transform):
    # 只读原图；RGB 转换和缩放在内存中完成。
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        result = transform(rgb)
        rgb.close()
    return result


def validate_mapping(mapping):
    if len(mapping) != 39 or sorted(mapping.values()) != list(range(39)):
        raise ValueError("本次 smoke 必须使用现有 39 类、0..38 的完整映射")


def model_for_classes(class_count):
    model = resnet18(weights=None)  # 从不触发网络下载。
    model.fc = torch.nn.Linear(model.fc.in_features, class_count)
    return model


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
