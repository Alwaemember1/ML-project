"""锁定已验收最佳模型的最终测试；不训练、不选模、不自动重复评价。"""

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from smoke_utils import (ROOT, build_transform, image_tensor, model_for_classes,
                         save_json, seed_torch, sha256_file, validate_mapping)
from scripts.prepare_formal_split import SPLIT, read_rows, verify_split
from evaluation_metrics import classification_metrics, confusion_pairs, audit_records
from evaluation_outputs import write_csv, heatmap, sample_cases, make_report

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

LIMITATION = "本划分已排除已知SHA256字节重复，但未排除增强同源关系，测试成绩可能偏乐观，不能直接代表独立叶片或真实拍摄场景的泛化能力。"


def now():
    return datetime.now(timezone(timedelta(hours=8)))


class FinalTestImages(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["relative_path"]).resolve()
        if not path.is_relative_to(self.root) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"测试输入与固定清单不符：{row['relative_path']}")
        return image_tensor(path, self.transform), int(row["class_idx"]), index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True, type=Path, help="原实验已验收的训练目录")
    parser.add_argument("--project-status", required=True, type=Path,
                        help="Git历史中的原实验验收说明快照，保留检查点路径核对")
    parser.add_argument("--implementation-fix-reason", help="仅真实实现错误修复后的明确重新评价原因；不能因模型分数重测")
    args = parser.parse_args()
    training_run = args.training_run.resolve()
    status_before = args.project_status.read_text(encoding="utf-8")
    completion = json.loads((training_run / "completion.json").read_text(encoding="utf-8"))
    config = json.loads((training_run / "training_config.json").read_text(encoding="utf-8"))
    training = json.loads((training_run / completion["training_report"]).read_text(encoding="utf-8"))
    checkpoint_path = Path(training["best_checkpoint"]).resolve()
    if completion["status"] != "completed_and_verified" or completion["best_epoch"] != 3 or training["best_epoch"] != 3:
        raise ValueError("训练验收或最佳轮次不是已确定的第3轮")
    if not checkpoint_path.is_relative_to(training_run) or checkpoint_path.name != "best_epoch_003.pt":
        raise ValueError("最佳检查点路径与实际训练记录不符")
    if checkpoint_path.relative_to(ROOT).as_posix() not in status_before.replace("\\", "/"):
        raise ValueError("原验收说明与训练记录中的最佳模型路径不一致")
    check = verify_split(SPLIT)
    if check["counts"]["test"] != 9192 or check["counts"] != completion["counts"]:
        raise ValueError("固定测试数量不符；停止，不重建划分")
    if len({completion["split_id"], training["split"]["split_id"], config["split_id"], check["split_id"]}) != 1:
        raise ValueError("验收、训练配置和固定划分标识不一致")
    if Path(config["split_directory"]).resolve() != SPLIT.resolve():
        raise ValueError("训练使用的划分目录不符")
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    lock = json.loads((SPLIT / "split_lock.json").read_text(encoding="utf-8"))
    mapping = checkpoint["class_to_idx"]
    validate_mapping(mapping)
    if mapping != json.loads((SPLIT / "class_to_idx.json").read_text(encoding="utf-8")):
        raise ValueError("检查点和固定划分类别映射不同")
    if checkpoint["split_lock"] != lock or checkpoint["split_id"] != check["split_id"] or checkpoint["epoch"] != 3:
        raise ValueError("检查点内部划分锁/轮次不符")
    if checkpoint["model_config"] != {"architecture": "resnet18", "num_classes": 39, "backbone_frozen": True,
                                      "pretrained_weights": "IMAGENET1K_V1"}:
        raise ValueError("检查点模型配置不符")
    if checkpoint["training_config"] != config or checkpoint["preprocessing"].get("random_augmentation") is not False:
        raise ValueError("检查点训练配置或确定性预处理不符")
    for name in ("smoke_utils.py", "scripts/prepare_formal_split.py"):
        if name not in config["source_code_sha256"]:
            raise ValueError("此训练记录使用历史源码结构；请审查原source_snapshot，不要修改来源锁或重新测试。")
        if sha256_file(ROOT / name) != config["source_code_sha256"][name]:
            raise ValueError(f"共用读取/预处理或划分实现已改变，先人工定位：{name}")
    registry_path = ROOT / "final_test_registry.json"
    if registry_path.exists():
        previous = json.loads(registry_path.read_text(encoding="utf-8"))
        if not args.implementation_fix_reason:
            raise PermissionError("最终测试已登记，拒绝静默重复。真实实现错误重评必须明确填写 --implementation-fix-reason 并保留旧结果。")
        if previous["checkpoint_sha256"] != checkpoint_hash or previous["split_id"] != check["split_id"]:
            raise ValueError("重评也不能更换锁定模型或划分")
    elif args.implementation_fix_reason:
        raise ValueError("没有既往评价，却提供了实现错误重评理由")
    names = [name for name, idx in sorted(mapping.items(), key=lambda pair: pair[1])]
    rows = read_rows(SPLIT / "test.csv")
    if len(rows) != 9192 or {row["class_name"] for row in rows} != set(mapping):
        raise ValueError("测试清单样本数或39类覆盖错误")
    if not torch.cuda.is_available():
        raise RuntimeError("现有 GPU 环境不可用，不重新安装或静默回退")
    seed_torch(42)
    model = model_for_classes(39)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.requires_grad_(False).eval().to("cuda:0")
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    del checkpoint["state_dict"]
    data_root = Path(checkpoint["dataset_root"]).resolve()
    split_config = json.loads((SPLIT / "split_config.json").read_text(encoding="utf-8"))
    if data_root != Path(split_config["dataset_root"]).resolve():
        raise ValueError("数据根目录与固定划分不一致")
    output = ROOT / "reports" / now().strftime("final_test_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    command = f"& '{sys.executable}' -m scripts.evaluate_final --training-run '{training_run}' --project-status '{args.project_status.resolve()}'"
    if args.implementation_fix_reason:
        command += " --implementation-fix-reason " + json.dumps(args.implementation_fix_reason, ensure_ascii=False)
    identity = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash, "epoch": 3,
                "training_run": str(training_run), "split_id": check["split_id"],
                "split_directory": str(SPLIT), "test_manifest_sha256": lock["artifact_sha256"]["test.csv"],
                "model_config": checkpoint["model_config"], "class_to_idx": mapping,
                "preprocessing": checkpoint["preprocessing"], "command": command, "locked_before_test": True}
    save_json(output / "model_identity.json", identity)
    save_json(output / "split_verification.json", check)
    with (output / "status_before_final_test.md").open("x", encoding="utf-8") as file:
        file.write(status_before)
    evaluation_config = {"started_at": now().isoformat(), "command": command, "training_run": str(training_run),
                         "python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
                         "torchvision": str(torchvision.__version__), "device": "cuda:0", "batch_size": 32,
                         "num_workers": 0, "shuffle": False, "seed_for_report_cases": 42,
                         "preprocessing": checkpoint["preprocessing"], "test_samples_expected": 9192,
                         "implementation_fix_reason": args.implementation_fix_reason,
                         "prior_evaluation": str(registry_path) if registry_path.exists() else None,
                         "script_sha256": {name: sha256_file(ROOT / name) for name in
                                            ("scripts/evaluate_final.py", "evaluation_metrics.py", "evaluation_outputs.py")},
                         "limitation": LIMITATION}
    save_json(output / "evaluation_config.json", evaluation_config)
    if not registry_path.exists():
        save_json(registry_path, {"registered_at": now().isoformat(), "checkpoint": str(checkpoint_path),
                  "checkpoint_sha256": checkpoint_hash, "split_id": check["split_id"],
                  "evaluation_dir": str(output), "policy": "测试已登记；禁止因模型表现反复评价或调参，真实实现错误需明确原因另存结果"})
    log_file = (output / "evaluation.log").open("x", encoding="utf-8")

    def log(message):
        message = f"[{now().isoformat(timespec='seconds')}] {message}"
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    try:
        log(f"preflight PASSED; final_test_output={output}")
        log(f"locked checkpoint={checkpoint_path}; sha256={checkpoint_hash}")
        log(f"split_id={check['split_id']}; test=9192; classes=39; eval + inference_mode; no parameter updates")
        loader = DataLoader(FinalTestImages(data_root, rows, build_transform(checkpoint["preprocessing"])),
                            batch_size=32, shuffle=False, num_workers=0)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        records, losses = [], []
        matrix = [[0] * 39 for _ in range(39)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as predictions_file:
            with torch.inference_mode():
                for step, (inputs, labels, indices) in enumerate(loader, 1):
                    if any(module.training for module in model.modules()):
                        raise AssertionError("模型未保持eval")
                    logits = model(inputs.to("cuda:0"))
                    batch_losses = criterion(logits, labels.to("cuda:0")).cpu().tolist()
                    if logits.shape != (len(labels), 39) or not torch.isfinite(logits).all().item():
                        raise AssertionError("输出维度或数值错误")
                    probabilities = logits.softmax(dim=1).cpu().tolist()
                    raw_logits = logits.cpu().tolist()
                    for local, index in enumerate(indices.tolist()):
                        source = rows[index]
                        scores = raw_logits[local]
                        top_ids = sorted(range(39), key=lambda idx: -scores[idx])[:3]
                        true_idx, pred_idx = int(source["class_idx"]), top_ids[0]
                        loss = batch_losses[local]
                        if not math.isfinite(loss):
                            raise FloatingPointError("逐图loss非有限")
                        record = {"relative_path": source["relative_path"], "sha256": source["sha256"],
                                  "true_idx": true_idx, "true_class": names[true_idx], "pred_idx": pred_idx,
                                  "pred_class": names[pred_idx], "pred_probability": probabilities[local][pred_idx],
                                  "correct": true_idx == pred_idx, "loss": loss,
                                  "top3": [{"class_idx": idx, "class_name": names[idx], "probability": probabilities[local][idx]}
                                           for idx in top_ids], "logits": scores, "probabilities": probabilities[local]}
                        predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records.append(record)
                        losses.append(loss)
                        matrix[true_idx][pred_idx] += 1
                    if step % 50 == 0 or step == len(loader):
                        log(f"test batch {step}/{len(loader)}, evaluated={len(records)}/9192")
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start
        if len(records) != 9192:
            raise AssertionError("没有完整评价测试集；不能发布部分成绩")
        changed = [key for key, value in model.state_dict().items() if not torch.equal(value.detach().cpu(), before[key])]
        if changed or sha256_file(checkpoint_path) != checkpoint_hash or any(value.grad is not None for value in model.parameters()):
            raise AssertionError(f"模型状态或文件发生改变：{changed}")
        metrics = classification_metrics(matrix, names)
        metrics.update(loss=math.fsum(losses) / len(losses), inference_seconds=inference_seconds,
                       model_state_tensors_unchanged=len(before), checkpoint_hash_unchanged=True,
                       test_image_sha256_verified=len(records), model_parameters_updated=False,
                       limitations=LIMITATION)
        save_json(output / "metrics.json", metrics)
        log(f"TEST: loss={metrics['loss']:.6f}; accuracy={metrics['accuracy']:.6f} ({metrics['correct']}/{metrics['total']}); "
            f"macro_P={metrics['macro_precision']:.6f}; macro_R={metrics['macro_recall']:.6f}; macro_F1={metrics['macro_f1']:.6f}; seconds={inference_seconds:.3f}")
        write_csv(output / "per_class_metrics.csv", metrics["classes"], list(metrics["classes"][0]))
        compact = []
        for row in records:
            item = {key: row[key] for key in ("relative_path", "sha256", "true_idx", "true_class", "pred_idx", "pred_class", "pred_probability", "correct", "loss")}
            for rank, top in enumerate(row["top3"], 1):
                item.update({f"top{rank}_idx": top["class_idx"], f"top{rank}_class": top["class_name"], f"top{rank}_probability": top["probability"]})
            compact.append(item)
        write_csv(output / "predictions.csv", compact, list(compact[0]))
        fields = ["true_idx", "true_class", *names]
        for normalized, filename in ((False, "confusion_counts.csv"), (True, "confusion_row_normalized.csv")):
            rows_out = [{"true_idx": idx, "true_class": names[idx],
                         **{name: (matrix[idx][pred] / sum(matrix[idx]) if normalized else matrix[idx][pred])
                            for pred, name in enumerate(names)}} for idx in range(39)]
            write_csv(output / filename, rows_out, fields)
        with (output / "predictions.jsonl").open(encoding="utf-8") as file:
            replayed = [json.loads(line) for line in file]
        audit = audit_records(replayed, names, matrix, metrics)
        save_json(output / "audit.json", audit)
        pairs = confusion_pairs(matrix, names)
        pair_fields = ["true_idx", "true_class", "pred_idx", "pred_class", "count", "true_support", "fraction_of_true_class", "fraction_of_source_errors"]
        write_csv(output / "confusion_pairs.csv", pairs, pair_fields)
        low_recall = sorted(metrics["classes"], key=lambda row: (row["recall"], row["class_idx"]))[:8]
        save_json(output / "error_analysis.json", {"lowest_recall": low_recall, "top_pairs_by_count": pairs[:10],
                  "top_pairs_by_class_fraction": sorted(pairs, key=lambda row: (-row["fraction_of_true_class"], -row["count"], row["true_idx"], row["pred_idx"]))[:10],
                  "observed_errors": metrics["errors"], "possible_causes_not_verified": True,
                  "labels_not_declared_wrong": True})
        heatmap(matrix, names, output / "confusion_counts.png")
        heatmap(matrix, names, output / "confusion_row_normalized.png", normalized=True)
        selections = sample_cases(records, data_root, output)
        make_report(output, metrics, identity, training, low_recall, pairs, selections, LIMITATION)
        save_json(output / "completion.json", {"status": "completed", "training_performed": False,
                  "hyperparameters_changed": False, "threshold_changed": False, "test_evaluated": True,
                  "test_total": metrics["total"], "model_parameters_updated": False,
                  "model_state_tensors_unchanged": len(before), "checkpoint_sha256_unchanged": True,
                  "prediction_records_audited": True, "finished_at": now().isoformat(),
                  "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
                  "split_id": check["split_id"], "implementation_fix_reason": args.implementation_fix_reason,
                  "test_cases_now_viewed_report_material": True})
        final_pointer = {**identity, "status": "frozen_final_model", "evaluation_dir": str(output),
                         "evaluation_metrics": str(output / "metrics.json"), "no_further_test_based_tuning": True,
                         "test_report_cases_are_viewed_material": True, "limitation": LIMITATION}
        if not (ROOT / "FINAL_MODEL.json").exists():
            save_json(ROOT / "FINAL_MODEL.json", final_pointer)
        log("FINAL TEST COMPLETE. Records audited; model unchanged; report examples now viewed. No training or tuning performed.")
    except BaseException as error:
        save_json(output / "failure.json", {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                  "records_written_if_any": len(records) if "records" in locals() else 0,
                  "reevaluation_requires_explicit_implementation_reason": True})
        log(traceback.format_exc())
        raise
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
