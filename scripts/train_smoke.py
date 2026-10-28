"""每类最多 30 张，仅训练 ResNet18 分类头 2 轮；非正式评价。"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from smoke_utils import (LIMITATION, ROOT, build_transform, image_tensor, preprocessing_info,
                         save_json, seed_torch, sha256_file, tensor_digest, validate_mapping,
                         pretrained_classifier)

import torch
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

BASELINE = ROOT / "reports" / "check_20261016_133031_170321"
AUDIT = ROOT / "reports" / "pretrain_20261016_230415"
CACHED_WEIGHTS = AUDIT / "torch_cache" / "checkpoints" / "resnet18-f37072fd.pth"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["relative_path", "class_name", "class_idx", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def smoke_split(output, mapping):
    # 仅读取旧清单，不遍历全量图片目录，不重新计算全量图片哈希。
    images = {row["relative_path"]: row for row in read_csv(BASELINE / "image_manifest.csv")}
    hashes = read_csv(AUDIT / "sha256_manifest.csv")
    representatives = {}
    for row in sorted(hashes, key=lambda item: item["relative_path"]):
        original = images[row["relative_path"]]
        if row["error"] or original["status"] != "valid" or original["class_name"] != row["class_name"]:
            raise ValueError("旧图片清单与哈希清单不一致或包含失败记录")
        digest = row["sha256"]
        if digest in representatives and representatives[digest]["class_name"] != row["class_name"]:
            raise ValueError("存在跨类别字节重复，需先解决标签冲突")
        representatives.setdefault(digest, {"relative_path": row["relative_path"],
                                            "class_name": row["class_name"],
                                            "class_idx": mapping[row["class_name"]], "sha256": digest})
    if len(images) != len(hashes) or len(images) != 61486:
        raise ValueError("已有清单条目数量不一致")
    buckets = defaultdict(list)
    for row in representatives.values():
        buckets[row["class_name"]].append(row)
    rng = random.Random(42)
    train, validation, counts = [], [], []
    for name in sorted(mapping, key=mapping.get):
        candidates = sorted(buckets[name], key=lambda item: item["relative_path"])
        selected = rng.sample(candidates, min(30, len(candidates)))
        if len(selected) < 2:
            raise ValueError(f"{name} 无法同时提供训练和验证样本")
        train_count = min(20, len(selected) - 1)
        train.extend(selected[:train_count])
        validation.extend(selected[train_count:])
        counts.append({"class_name": name, "train": train_count, "validation": len(selected) - train_count})
    all_selected = train + validation
    if len({row["sha256"] for row in all_selected}) != len(all_selected):
        raise AssertionError("smoke 划分中存在重复 SHA256")
    write_csv(output / "smoke_train.csv", train)
    write_csv(output / "smoke_validation.csv", validation)
    save_json(output / "smoke_class_to_idx.json", mapping)
    save_json(output / "smoke_split_info.json", {
        "purpose": "smoke/debug", "limitation": LIMITATION, "seed": 42,
        "representative_rule": "同一 SHA256 保留相对路径字典序最小的一个文件，再逐类随机抽样",
        "train_count": len(train), "validation_count": len(validation), "test_set_created": False,
        "known_byte_duplicate_overlap": 0, "augmentation_lineage_excluded": False, "classes": counts,
        "image_manifest": str(BASELINE / "image_manifest.csv"),
        "sha256_manifest": str(AUDIT / "sha256_manifest.csv"),
    })
    return train, validation


class SmokeImages(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return image_tensor(self.root / row["relative_path"], self.transform), row["class_idx"]


def backbone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not name.startswith("fc.")}


def verify_frozen(model, before):
    changed = [name for name, value in model.state_dict().items()
               if not name.startswith("fc.") and not torch.equal(value.detach().cpu(), before[name])]
    gradients = [name for name, value in model.named_parameters() if not name.startswith("fc.")
                 and (value.requires_grad or value.grad is not None)]
    if changed or gradients:
        raise AssertionError(f"冻结失败：state={changed}, gradients={gradients}")
    return {"backbone_parameters_and_buffers_unchanged": True,
            "checked_state_tensors": len(before), "backbone_gradients_absent": True,
            "batchnorm_buffer_tensors": sum("running_" in name or "num_batches_tracked" in name for name in before)}


def run_epoch(model, loader, criterion, optimizer, device, training, log, log_every=10):
    # 不调用 model.train()：整网 eval，仅分类头 train，主干 BN 统计始终固定。
    model.eval()
    if training:
        model.fc.train()
    if any(module.training for name, module in model.named_modules() if name and not name.startswith("fc")):
        raise AssertionError("冻结主干中有模块处于 train 模式")
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss_sum, correct, samples, nonzero_gradient = 0.0, 0, 0, False
    with torch.set_grad_enabled(training):
        for step, (inputs, labels) in enumerate(loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if logits.shape != (len(labels), 39):
                raise AssertionError("模型输出不是 39 类")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("loss 非有限值")
            if training:
                loss.backward()
                for parameter in model.fc.parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                        raise FloatingPointError("分类头梯度缺失或包含非有限值")
                    nonzero_gradient |= bool(torch.count_nonzero(parameter.grad).item())
                optimizer.step()
            count = len(labels)
            loss_sum += loss.item() * count  # 按样本加权，最后不足一批也正确计入。
            correct += (logits.argmax(dim=1) == labels).sum().item()
            samples += count
            if step % log_every == 0 or step == len(loader):
                log(f"  {'train' if training else 'validation'} batch {step}/{len(loader)}, samples={samples}")
    torch.cuda.synchronize()
    if training and not nonzero_gradient:
        raise AssertionError("没有检测到非零分类头梯度")
    return {"samples": samples, "loss": loss_sum / samples, "accuracy": correct / samples,
            "correct": correct, "seconds": time.perf_counter() - started,
            "head_nonzero_gradient_observed": nonzero_gradient if training else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16, choices=[4, 8, 16])
    args = parser.parse_args()
    output = ROOT / "reports" / datetime.now().strftime("smoke_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "smoke_training.log").open("x", encoding="utf-8") as log_file:
        def log(message):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(f"smoke output: {output}")
        log(LIMITATION)
        overall = time.perf_counter()
        (output / "readme_before_smoke.md").write_text((ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；不回退 CPU 冒充 GPU 训练")
        if not CACHED_WEIGHTS.is_file():
            raise FileNotFoundError("已缓存预训练权重不存在；本工具不会下载")
        pretrained_hash = sha256_file(CACHED_WEIGHTS)
        if not pretrained_hash.startswith("f37072fd"):
            raise ValueError("本地预训练缓存哈希与 ResNet18 V1 不符")
        seed_torch(42)
        random.seed(42)
        device = torch.device("cuda:0")
        config = json.loads((ROOT / "dataset_config.json").read_text(encoding="utf-8-sig"))
        data = (ROOT / config["dataset_root"]).resolve()
        mapping = json.loads((BASELINE / "class_to_idx.json").read_text(encoding="utf-8"))
        validate_mapping(mapping)
        train_rows, val_rows = smoke_split(output, mapping)
        # 仅复核这次选中的图片，保证旧 SHA256 记录对当前小样本仍有效。
        for row in train_rows + val_rows:
            path = (data / row["relative_path"]).resolve()
            if not path.is_relative_to(data) or sha256_file(path) != row["sha256"]:
                raise ValueError(f"所选图片与旧哈希不一致：{row['relative_path']}")
        log(f"selected: train={len(train_rows)}, validation={len(val_rows)}, classes={len(mapping)}, SHA256 overlap=0")
        info = preprocessing_info()
        transform = build_transform(info)
        # 与 torchvision 官方权重的变换逐元素比较，验证自包含预处理配置。
        with Image.open(data / val_rows[0]["relative_path"]) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            if not torch.equal(transform(rgb), ResNet18_Weights.IMAGENET1K_V1.transforms()(rgb)):
                raise AssertionError("预处理与预训练权重要求不一致")
            rgb.close()
        generator = torch.Generator().manual_seed(42)
        train_loader = DataLoader(SmokeImages(data, train_rows, transform), batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, generator=generator)
        val_loader = DataLoader(SmokeImages(data, val_rows, transform), batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        model = pretrained_classifier(CACHED_WEIGHTS, len(mapping))
        model.to(device).eval()
        before = backbone_state(model)
        head_before = {name: value.detach().cpu().clone() for name, value in model.fc.named_parameters()}
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        log(f"GPU={torch.cuda.get_device_name(0)}, batch_size={args.batch_size}, num_workers=0, pretrained=local cache")
        log(f"trainable parameters={sum(value.numel() for value in model.parameters() if value.requires_grad)}")
        metrics = []
        training_started = time.perf_counter()
        for epoch in range(1, 3):
            trained = run_epoch(model, train_loader, criterion, optimizer, device, True, log)
            validated = run_epoch(model, val_loader, criterion, optimizer, device, False, log)
            frozen = verify_frozen(model, before)
            metrics.append({"epoch": epoch, "train": trained, "validation": validated, "frozen_check": frozen})
            log(f"epoch {epoch}/2 | train loss={trained['loss']:.6f} accuracy={trained['accuracy']:.6f} "
                f"({trained['correct']}/{trained['samples']}) seconds={trained['seconds']:.3f} | "
                f"validation loss={validated['loss']:.6f} accuracy={validated['accuracy']:.6f} "
                f"({validated['correct']}/{validated['samples']}) seconds={validated['seconds']:.3f} | backbone unchanged=True")
        training_seconds = time.perf_counter() - training_started
        head_changes = {}
        for name, value in model.fc.named_parameters():
            delta = value.detach().cpu() - head_before[name]
            head_changes[name] = {"updated": not torch.equal(value.detach().cpu(), head_before[name]),
                                  "l2_delta": delta.norm().item(), "max_abs_delta": delta.abs().max().item()}
        if not all(item["updated"] for item in head_changes.values()):
            raise AssertionError("分类头部分参数没有更新")
        frozen = verify_frozen(model, before)
        model.eval()
        reference_row = val_rows[0]
        reference_tensor = image_tensor(data / reference_row["relative_path"], transform)
        with torch.inference_mode():
            reference_logits = model(reference_tensor.unsqueeze(0).to(device)).cpu()
        save_json(output / "smoke_reference.json", {
            "training_pid": os.getpid(),
            "relative_path": reference_row["relative_path"], "image_sha256": reference_row["sha256"],
            "input_tensor_sha256": tensor_digest(reference_tensor), "class_to_idx": mapping,
            "output_shape": list(reference_logits.shape), "logits": reference_logits[0].tolist(),
            "probabilities": reference_logits.softmax(dim=1)[0].tolist(),
        })
        model_config = {"architecture": "resnet18", "num_classes": len(mapping), "backbone_frozen": True,
                        "pretrained_weights": "IMAGENET1K_V1", "pretrained_cache": str(CACHED_WEIGHTS),
                        "pretrained_sha256": pretrained_hash}
        checkpoint = output / "smoke_resnet18_head_2epochs.pt"
        with checkpoint.open("xb") as file:
            torch.save({"format": "leaf_smoke_v1", "state_dict": {name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()}, "class_to_idx": mapping,
                        "model_config": model_config, "preprocessing": info, "dataset_root": str(data),
                        "mapping_source": str(BASELINE / "class_to_idx.json"),
                        "training_config": {"epochs": 2, "batch_size": args.batch_size, "learning_rate": 0.001,
                                            "optimizer": "Adam", "loss": "CrossEntropyLoss", "seed": 42,
                                            "num_workers": 0}, "limitation": LIMITATION}, file)
        result = {"purpose": "smoke/debug", "limitation": LIMITATION, "output_dir": str(output),
                  "training_pid": os.getpid(),
                  "python": sys.version, "executable": sys.executable,
                  "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                  "device": str(device), "class_count": len(mapping), "train_count": len(train_rows),
                  "validation_count": len(val_rows), "selected_sha256_verified": True,
                  "preprocessing_matches_pretrained": True, "epochs": metrics,
                  "training_validation_seconds": training_seconds,
                  "total_seconds_through_checkpoint": time.perf_counter() - overall,
                  "frozen_check": frozen, "head_parameter_changes": head_changes,
                  "checkpoint": str(checkpoint), "model_config": model_config, "preprocessing": info,
                  "independent_reload_status": "pending: run predict_smoke.py in a separate process"}
        save_json(output / "smoke_training_summary.json", result)
        log(f"head updated={head_changes}; frozen check={frozen}")
        log(f"training+validation seconds={training_seconds:.3f}; total through checkpoint={result['total_seconds_through_checkpoint']:.3f}")
        log(f"checkpoint saved: {checkpoint}")
        log("2 epochs complete; training stopped. Independent reload verification is a separate command.")


if __name__ == "__main__":
    main()
