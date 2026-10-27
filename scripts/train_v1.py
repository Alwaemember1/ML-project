"""冻结主干的完整训练，可选锁定均衡视图；最多10轮，不构造测试DataLoader。"""

import argparse
import csv
import json
import math
import os
import random
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

from smoke_utils import (ROOT, build_training_transform, build_transform, image_tensor,
                         preprocessing_info, pretrained_classifier, save_json, seed_torch,
                         sha256_file, tensor_digest)
from scripts.train_smoke import SmokeImages, backbone_state, verify_frozen, run_epoch, CACHED_WEIGHTS
from scripts.prepare_formal_split import SPLIT, LIMITATION, read_rows, verify_split

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader


def local_time():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def state_file(output, **state):
    temporary = output / "run_state.tmp"
    temporary.write_text(json.dumps({"pid": os.getpid(), "updated_at": local_time(), **state},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output / "run_state.json")


def draw_curves(history, output, best_epoch, filename="training_curves.png", title="V1"):
    # 使用现有 Pillow 输出独立 PNG；逐点来自真实 metrics.csv，不安装绘图库。
    image = Image.new("RGB", (1440, 650), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 25)
    blue, orange = "#1768ac", "#d2691e"
    for panel, metric in enumerate(("loss", "accuracy")):
        left, top, width, height = 90 + panel * 710, 95, 570, 420
        n = len(history)
        maximum = (max(row[f"{part}_{metric}"] for row in history for part in ("train", "validation")) * 1.1
                   if metric == "loss" else 1.0)
        maximum = max(maximum, 0.01)
        x = lambda epoch: left + (epoch - 1) * width / max(n - 1, 1)
        y = lambda value: top + height - value / maximum * height
        draw.text((left, 35), f"{title} {metric}: train / validation", fill="black", font=title_font)
        for tick in range(6):
            value = maximum * tick / 5
            yy = y(value)
            draw.line((left, yy, left + width, yy), fill="#dddddd")
            label = f"{value:.2f}" if metric == "loss" else f"{value:.0%}"
            draw.text((left - 65, yy - 10), label, fill="black", font=font)
        draw.line((left, top, left, top + height, left + width, top + height), fill="black", width=2)
        for row in history:
            xx = x(row["epoch"])
            draw.text((xx - 7, top + height + 12), str(row["epoch"]), fill="black", font=font)
        for part, color in (("train", blue), ("validation", orange)):
            points = [(x(row["epoch"]), y(row[f"{part}_{metric}"])) for row in history]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for xx, yy in points:
                draw.ellipse((xx - 4, yy - 4, xx + 4, yy + 4), fill=color)
        best_x = x(best_epoch)
        draw.line((best_x, top, best_x, top + height), fill="#777777", width=1)
        draw.text((left, 565), "Train (random crop + flip)", fill=blue, font=font)
        draw.text((left, 593), f"Validation (deterministic); best loss epoch {best_epoch}", fill=orange, font=font)
    draw.text((90, 625), "Raw epoch metrics; no smoothing. Model selection uses validation loss only. Test set not evaluated.",
              fill="#444444", font=font)
    image.save(output / filename)
    image.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument("--balanced-manifest", type=Path, help="仅读取已锁定的训练集均衡候选视图")
    parser.add_argument("--output-dir", type=Path, help="新的独立实验目录，拒绝覆盖")
    args = parser.parse_args()
    check = verify_split()
    purpose = "balanced_subset_head_comparison" if args.balanced_manifest else "v1_full_training"
    prefix = "balanced_head_" if args.balanced_manifest else "v1_head_"
    output = (args.output_dir.resolve() if args.output_dir else ROOT / "reports" /
              (prefix + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")))
    if not output.is_relative_to(ROOT / "reports"):
        raise ValueError("实验目录必须位于本项目reports内")
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    log_file = (output / "training.log").open("x", encoding="utf-8")

    def log(message):
        message = f"[{local_time()}] {message}"
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    state_file(output, stage="initializing", split_id=check["split_id"], test_images_opened=0)
    log(f"experiment={output}; split_id={check['split_id']}")
    overall = time.perf_counter()
    try:
        with (output / "status_before_v1.md").open("x", encoding="utf-8") as file:
            file.write((ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8"))
        save_json(output / "split_verification.json", check)
        if not torch.cuda.is_available():
            raise RuntimeError("现有 CUDA 环境不可用，拒绝自动改用 CPU")
        pretrained_hash = sha256_file(CACHED_WEIGHTS)
        if not pretrained_hash.startswith("f37072fd"):
            raise ValueError("已有 ImageNet 权重缓存校验失败，不下载替代")
        seed_torch(42)
        random.seed(42)
        split_config = json.loads((SPLIT / "split_config.json").read_text(encoding="utf-8"))
        mapping = json.loads((SPLIT / "class_to_idx.json").read_text(encoding="utf-8"))
        data = Path(split_config["dataset_root"])
        # 这里只读取训练、验证清单；verify_split 对测试仅核对清单元数据。
        train_rows, val_rows = read_rows(SPLIT / "train.csv"), read_rows(SPLIT / "validation.csv")
        balance_check, web_lock_before = None, None
        if args.balanced_manifest:
            from balanced_manifest import load_balanced_training
            train_rows, balance_check = load_balanced_training(args.balanced_manifest, SPLIT / "train.csv",
                                                               val_rows, mapping, check["split_id"])
            save_json(output / "balanced_manifest_verification.json", balance_check)
            web_record = json.loads((ROOT / "FINAL_MODEL.json").read_text(encoding="utf-8"))
            web_lock_before = {"record_sha256":sha256_file(ROOT / "FINAL_MODEL.json"),
                               "checkpoint":web_record["checkpoint"],
                               "checkpoint_sha256":sha256_file(web_record["checkpoint"])}
            if web_lock_before["checkpoint_sha256"] != web_record["checkpoint_sha256"]:
                raise ValueError("网页冻结模型哈希不符，停止新实验")
            save_json(output / "frozen_web_before.json", web_lock_before)
            log(f"balanced view verified: {balance_check['classes']} classes x {balance_check['per_class']}; web model preserved")
        for row in train_rows + val_rows:
            row["class_idx"] = int(row["class_idx"])
        info = preprocessing_info()
        augmentation = {"RandomResizedCrop": {"size": 224, "scale": [0.8, 1.0], "ratio": [0.9, 1.1]},
                        "RandomHorizontalFlip": {"p": 0.5}, "normalization": "same as ImageNet V1"}
        config = {"epochs_max": 10, "early_stopping_patience": 3, "monitor": "validation_loss",
                  "min_delta": 0, "optimizer": "Adam", "learning_rate": 0.001,
                  "loss": "CrossEntropyLoss", "initial_batch_size": 32, "actual_batch_size": args.batch_size,
                  "num_workers": 0, "seed": 42, "backbone_frozen": True, "batchnorm_eval": True,
                  "pretrained_cache": str(CACHED_WEIGHTS), "pretrained_sha256": pretrained_hash,
                  "started_from_smoke": False, "train_augmentation": augmentation,
                  "python": os.sys.version, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                  "device": "cuda:0", "gpu": torch.cuda.get_device_name(0), "training_pid": os.getpid(),
                  "started_at": local_time(), "split_id": check["split_id"], "split_directory": str(SPLIT),
                  "source_code_sha256": {name: sha256_file(ROOT / name) for name in
                                         ("scripts/train_v1.py", "smoke_utils.py", "project_config.py", "scripts/train_smoke.py",
                                          "scripts/prepare_formal_split.py", "scripts/predict_smoke.py")},
                  "purpose": purpose, "balanced_training_view": balance_check,
                  "original_web_model_preserved": web_lock_before,
                  "limitation": LIMITATION, "test_policy": "not loaded or evaluated"}
        if args.balanced_manifest:
            config["source_code_sha256"]["balanced_manifest.py"] = sha256_file(ROOT / "balanced_manifest.py")
            config["comparison_scope"] = "原固定验证集对照；下采样同时改变训练量和每轮更新次数，不能把差异单独归因于均衡；不看测试、不替换网页"
        (output / "source_snapshot").mkdir()
        for name in config["source_code_sha256"]:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        save_json(output / "training_config.json", config)
        generator = torch.Generator().manual_seed(42)
        train_loader = DataLoader(SmokeImages(data, train_rows, build_training_transform(info)),
                                  batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
        val_loader = DataLoader(SmokeImages(data, val_rows, build_transform(info)), batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        model = pretrained_classifier(CACHED_WEIGHTS, 39).to("cuda:0").eval()
        before = backbone_state(model)
        head_before = {name: value.detach().cpu().clone() for name, value in model.fc.named_parameters()}
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        log(f"train={len(train_rows)}, validation={len(val_rows)}; batch_size={args.batch_size}; workers=0; fresh ImageNet + new head")
        history, best_loss, best_epoch, best_path, stale, gate_seconds = [], math.inf, None, None, 0, 0.0

        def save_checkpoint(path, epoch):
            with path.open("xb") as file:
                torch.save({"format": "leaf_v1_head", "purpose": purpose, "epoch": epoch,
                            "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                            "class_to_idx": mapping, "model_config": {"architecture": "resnet18", "num_classes": 39,
                            "backbone_frozen": True, "pretrained_weights": "IMAGENET1K_V1"},
                            "preprocessing": info, "training_augmentation": augmentation, "training_config": config,
                            "dataset_root": str(data), "mapping_source": str(SPLIT / "class_to_idx.json"),
                            "split_id": check["split_id"], "split_directory": str(SPLIT),
                            "split_lock": json.loads((SPLIT / "split_lock.json").read_text(encoding="utf-8")),
                            "metrics": history[-1], "limitation": LIMITATION}, file)

        fields = ["epoch", "train_loss", "train_accuracy", "train_samples", "train_seconds", "validation_loss",
                  "validation_accuracy", "validation_samples", "validation_seconds", "learning_rate", "improved"]
        training_start = time.perf_counter()
        with (output / "metrics.csv").open("x", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, 11):
                state_file(output, stage="training", epoch=epoch, split_id=check["split_id"], test_images_opened=0)
                trained = run_epoch(model, train_loader, criterion, optimizer, "cuda:0", True, log, log_every=100)
                validated = run_epoch(model, val_loader, criterion, optimizer, "cuda:0", False, log, log_every=100)
                if trained["samples"] != len(train_rows) or validated["samples"] != len(val_rows):
                    raise AssertionError("每轮样本数不正确")
                frozen = verify_frozen(model, before)
                if not all(not torch.equal(value.detach().cpu(), head_before[name]) for name, value in model.fc.named_parameters()):
                    raise AssertionError("分类头未更新")
                improved = validated["loss"] < best_loss
                row = {"epoch": epoch, "train_loss": trained["loss"], "train_accuracy": trained["accuracy"],
                       "train_samples": trained["samples"], "train_seconds": trained["seconds"],
                       "validation_loss": validated["loss"], "validation_accuracy": validated["accuracy"],
                       "validation_samples": validated["samples"], "validation_seconds": validated["seconds"],
                       "learning_rate": optimizer.param_groups[0]["lr"], "improved": improved}
                history.append(row)
                writer.writerow(row)
                csv_file.flush()
                if improved:
                    best_loss, best_epoch, stale = validated["loss"], epoch, 0
                    best_path = output / "checkpoints" / f"best_epoch_{epoch:03}.pt"
                    save_checkpoint(best_path, epoch)
                else:
                    stale += 1
                save_json(output / f"epoch_{epoch:03}.json", {**row, "frozen_check": frozen,
                          "best_epoch": best_epoch, "best_validation_loss": best_loss, "patience_used": stale})
                log(f"EPOCH {epoch}/10 train loss={trained['loss']:.6f} acc={trained['accuracy']:.6f} sec={trained['seconds']:.3f}; "
                    f"val loss={validated['loss']:.6f} acc={validated['accuracy']:.6f} sec={validated['seconds']:.3f}; "
                    f"best={best_epoch}; patience={stale}/3; frozen=True")
                if epoch == 1:
                    save_json(output / "first_epoch_ready.json", {"epoch": 1, "split_id": check["split_id"],
                              "metrics": row, "frozen_check": frozen, "head_updated": True})
                    state_file(output, stage="waiting_first_epoch_review", epoch=1, split_id=check["split_id"],
                               gate_file=str(output / "continue_after_epoch1.json"), test_images_opened=0)
                    log("第 1 轮完成，等待检查 first_epoch_ready.json 后创建 continue_after_epoch1.json；进程仍存活。")
                    wait_start = time.perf_counter()
                    while not (output / "continue_after_epoch1.json").exists():
                        if time.perf_counter() - wait_start > 1800:
                            raise TimeoutError("等待第一轮核对超过 30 分钟，停止而不自动继续")
                        time.sleep(2)
                    gate = json.loads((output / "continue_after_epoch1.json").read_text(encoding="utf-8"))
                    if gate.get("split_id") != check["split_id"] or gate.get("checked_epoch") != 1:
                        raise ValueError("第 1 轮核对文件无效")
                    gate_seconds = time.perf_counter() - wait_start
                    log("第 1 轮已核对，继续同一进程和优化器训练。")
                if stale >= 3:
                    log("验证 loss 连续 3 轮未降低，触发早停。")
                    break
        training_seconds = time.perf_counter() - training_start - gate_seconds
        last_path = output / "checkpoints" / f"last_epoch_{history[-1]['epoch']:03}.pt"
        save_checkpoint(last_path, history[-1]["epoch"])
        last_frozen = verify_frozen(model, before)
        # 为独立重载选择预先固定的第一张验证图片，从不使用测试图片。
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best["state_dict"], strict=True)
        model.eval()
        reference_row = val_rows[0]
        tensor = image_tensor(data / reference_row["relative_path"], build_transform(info))
        with torch.inference_mode():
            logits = model(tensor.unsqueeze(0).to("cuda:0")).cpu()
        save_json(output / "best_validation_reference.json", {"training_pid": os.getpid(),
                  "relative_path": reference_row["relative_path"], "image_sha256": reference_row["sha256"],
                  "input_tensor_sha256": tensor_digest(tensor), "class_to_idx": mapping,
                  "output_shape": list(logits.shape), "logits": logits[0].tolist(),
                  "probabilities": logits.softmax(dim=1)[0].tolist(), "source_split": "validation"})
        draw_curves(history, output, best_epoch, title="Balanced subset" if args.balanced_manifest else "V1")
        if web_lock_before:
            web_after = {"record_sha256":sha256_file(ROOT / "FINAL_MODEL.json"),
                         "checkpoint":web_lock_before["checkpoint"],
                         "checkpoint_sha256":sha256_file(web_lock_before["checkpoint"])}
            if web_after != web_lock_before:
                raise AssertionError("网页冻结记录或原模型在实验期间改变")
            save_json(output / "frozen_web_after.json", {**web_after, "unchanged":True})
        summary = {"purpose": purpose, "status": "training_completed_reload_pending", "output_dir": str(output),
                   "epochs_completed": len(history), "stopped_reason": "early_stopping" if stale >= 3 else "max_epochs_10",
                   "best_epoch": best_epoch, "best_validation_loss": best_loss, "best_checkpoint": str(best_path),
                   "last_checkpoint": str(last_path), "history": history, "split": check,
                   "training_validation_wall_seconds_excluding_gate": training_seconds, "first_epoch_review_wait_seconds": gate_seconds,
                   "elapsed_main_seconds": time.perf_counter() - overall, "frozen_check": last_frozen,
                   "head_updated": True, "test_images_opened": 0, "test_evaluated": False, "training_config": config,
                   "finished_at": local_time(), "limitation": LIMITATION}
        save_json(output / "training_summary.json", summary)
        state_file(output, stage="completed", epochs=len(history), best_epoch=best_epoch, best_checkpoint=str(best_path),
                   test_images_opened=0, independent_reload="pending")
        log(f"训练完成并停止。best epoch={best_epoch}; best={best_path}; last={last_path}; "
            f"train/val wall excluding review={training_seconds:.3f}s; test not evaluated.")
    except BaseException as error:
        state_file(output, stage="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                   test_images_opened=0)
        log(traceback.format_exc())
        raise
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
