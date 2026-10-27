"""冻结模型的共用单图推理：不读取数据集/划分，不下载权重，不保存上传图片。"""

import hashlib
import io
import json
import math
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# 此模块先于 Streamlit 导入，复用已有 D 盘缓存和已验证的模型/预处理构造。
from smoke_utils import ROOT, build_transform, model_for_classes, validate_mapping
from project_config import FROZEN_SHA256, default_record_path, resolve_record_file as resolve_config_file

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

FINAL_RECORD = default_record_path()
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_FORMATS = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".jpe": "JPEG", ".jfif": "JPEG",
    ".png": "PNG", ".webp": "WEBP", ".bmp": "BMP",
    ".tif": "TIFF", ".tiff": "TIFF", ".gif": "GIF", ".avif": "AVIF",
}
SUPPORTED_FORMATS_TEXT = "JPG / JPEG / JPE / JFIF / PNG / WebP / BMP / TIFF / GIF / AVIF"


class InferenceError(RuntimeError):
    """可以直接向用户展示的加载/输入错误。"""


def read_final_record(path=FINAL_RECORD):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if record["status"] != "frozen_final_model" or record["checkpoint_sha256"] != FROZEN_SHA256:
            raise ValueError("冻结状态或 SHA256 与本项目锁定记录不一致")
        return record
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InferenceError(f"无法读取有效的冻结记录：{path}；{exc}") from exc


def resolve_record_file(value, record_path=FINAL_RECORD):
    """交付包支持相对冻结记录的路径；原项目已有绝对路径保持不变。"""
    return resolve_config_file(value, record_path)


def model_cache_key(record_path=FINAL_RECORD):
    """文件或冻结记录变化时使网页缓存失效；缺失模型也不会继续使用旧缓存。"""
    path = Path(record_path)
    record = read_final_record(path)
    checkpoint = resolve_record_file(record["checkpoint"], path)
    try:
        stat = checkpoint.stat()
    except OSError as exc:
        raise InferenceError(f"模型文件不存在或无法访问：{checkpoint}；{exc}") from exc
    return (str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest(),
            str(checkpoint), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass
class Predictor:
    model: object
    transform: object
    class_to_idx: dict
    preprocessing: dict
    device: str
    checkpoint: str
    checkpoint_sha256: str
    lock: object = field(default_factory=threading.Lock)

    def predict(self, rgb_image):
        # decode_image 已统一 EXIF 方向和 RGB，变换与原验证代码完全共用。
        tensor = self.transform(rgb_image)
        with self.lock, torch.inference_mode():
            self.model.eval()
            logits = self.model(tensor.unsqueeze(0).to(self.device)).cpu()
            if logits.shape != (1, 39) or not torch.isfinite(logits).all().item():
                raise InferenceError("模型输出异常：需要 39 个有限的类别分数。")
            probabilities = logits.softmax(dim=1)[0]
            values, indices = probabilities.topk(3)
        names = {index: name for name, index in self.class_to_idx.items()}
        return {
            "device": self.device, "checkpoint_sha256": self.checkpoint_sha256,
            "output_shape": list(logits.shape), "logits": logits[0].tolist(),
            "probabilities": probabilities.tolist(),
            "top3": [{"class_idx": index, "class_name": names[index], "probability": value}
                     for index, value in zip(indices.tolist(), values.tolist())],
        }


def load_predictor(record_path=FINAL_RECORD, device="auto"):
    record = read_final_record(record_path)
    path = resolve_record_file(record["checkpoint"], record_path)
    try:
        # 同一只读句柄先核对哈希再加载，完整 state_dict 不需要预训练下载。
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != record["checkpoint_sha256"]:
                raise InferenceError(f"模型 SHA256 不符，已停止加载。实际：{digest}")
            handle.seek(0)
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
        for key in ("class_to_idx", "model_config", "preprocessing", "split_id", "epoch"):
            if checkpoint[key] != record[key]:
                raise InferenceError(f"检查点与冻结记录的 {key} 不一致，已停止加载。")
        mapping = checkpoint["class_to_idx"]
        validate_mapping(mapping)
        config = checkpoint["model_config"]
        info = checkpoint["preprocessing"]
        if config["architecture"] != "resnet18" or config["num_classes"] != 39:
            raise InferenceError("冻结模型必须是 ResNet18 / 39 类。")
        if not info["rgb"] or not info["exif_transpose"] or info["random_augmentation"]:
            raise InferenceError("检查点不是本项目已验证的 RGB / EXIF 确定性预处理。")
        if device not in ("auto", "cpu", "cuda:0"):
            raise InferenceError(f"不支持的设备：{device}")
        chosen = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if chosen == "cuda:0" and not torch.cuda.is_available():
            raise InferenceError("请求了 GPU，但当前 CUDA 不可用。可明确选择 CPU 重试。")
        model = model_for_classes(39)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.requires_grad_(False).eval().to(chosen)
        return Predictor(model, build_transform(info), mapping, info, chosen, str(path), digest)
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(f"模型加载失败：{path}；{type(exc).__name__}: {exc}") from exc


def decode_image(data, filename):
    """只在内存完整解码；后缀、真实格式、大小都核验，保留 Pillow 安全限制。"""
    if len(data) > MAX_FILE_BYTES:
        raise InferenceError("图片文件过大：最多允许 10 MiB。")
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_FORMATS:
        raise InferenceError(f"支持的图片格式：{SUPPORTED_FORMATS_TEXT}（仅限单帧静态图片）。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                detected = image.format
                if detected != ALLOWED_FORMATS[extension]:
                    raise InferenceError(f"后缀与实际图片格式不一致：{extension} / {detected}。")
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise InferenceError("图片像素过大：最多允许 2000 万像素，请先缩小图片。")
                if getattr(image, "n_frames", 1) != 1:
                    raise InferenceError("请上传单帧静态图片，不支持动画或多帧图片。")
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                rgb = ImageOps.exif_transpose(image).convert("RGB")
        return rgb, {"filename": Path(filename).name, "format": detected,
                     "width": width, "height": height, "bytes": len(data)}
    except InferenceError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InferenceError(f"图片触发 Pillow 安全限制：{exc}") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
        raise InferenceError(f"图片损坏或无法解码：{type(exc).__name__}: {exc}") from exc


def read_experiment_metrics(record_path=FINAL_RECORD):
    record = read_final_record(record_path)
    path = resolve_record_file(record["evaluation_metrics"], record_path)
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        for key in ("accuracy", "macro_f1"):
            if not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
                raise ValueError(f"无效指标 {key}")
        if len(metrics["classes"]) != 39 or metrics["total"] != 9192:
            raise ValueError("不是预期的 39 类 / 9192 张固定测试记录")
        return metrics, path
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InferenceError(f"无法读取已有实验指标：{path}；{exc}") from exc
