"""独立重载 smoke / v1 检查点，对指定验证参考图输出 Top3 并核对保存前结果。"""

import argparse
import json
import os
import time
from pathlib import Path

from smoke_utils import (build_transform, image_tensor, model_for_classes, save_json, seed_torch,
                         sha256_file, tensor_digest, validate_mapping)

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖：{args.output}")
    started = time.perf_counter()
    seed_torch(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if reference["training_pid"] == os.getpid():
        raise AssertionError("重载验证必须使用不同于训练进程的独立进程")
    mapping = checkpoint["class_to_idx"]
    validate_mapping(mapping)
    original_mapping = json.loads(Path(checkpoint["mapping_source"]).read_text(encoding="utf-8"))
    if mapping != reference["class_to_idx"] or mapping != original_mapping:
        raise AssertionError("保存前后或原始清单中的类别编号不一致")
    if checkpoint["model_config"]["num_classes"] != 39:
        raise AssertionError("检查点类别数不是 39")
    if "split_id" in checkpoint:
        from scripts.prepare_formal_split import verify_split, read_rows
        split_check = verify_split(Path(checkpoint["split_directory"]))
        if split_check["split_id"] != checkpoint["split_id"] or reference.get("source_split") != "validation":
            raise AssertionError("正式模型只允许用固定验证参考图验收")
        validation_rows = read_rows(Path(checkpoint["split_directory"]) / "validation.csv")
        if not any(row["relative_path"] == reference["relative_path"] and row["sha256"] == reference["image_sha256"]
                   for row in validation_rows):
            raise AssertionError("参考图不属于正式验证集")
    model = model_for_classes(39)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to("cuda:0")
    image = Path(checkpoint["dataset_root"]) / reference["relative_path"]
    if sha256_file(image) != reference["image_sha256"]:
        raise AssertionError("重载预测所用原图内容已变化")
    tensor = image_tensor(image, build_transform(checkpoint["preprocessing"]))
    if tensor_digest(tensor) != reference["input_tensor_sha256"]:
        raise AssertionError("保存前后预处理得到的输入张量不一致")
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0).to("cuda:0")).cpu()
    expected = torch.tensor([reference["logits"]], dtype=logits.dtype)
    if logits.shape != (1, 39) or not torch.isfinite(logits).all().item():
        raise AssertionError("重载输出形状或有限性错误")
    torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-6)
    probabilities = logits.softmax(dim=1)
    expected_probabilities = torch.tensor([reference["probabilities"]])
    torch.testing.assert_close(probabilities, expected_probabilities, rtol=1e-5, atol=1e-6)
    values, indices = probabilities[0].topk(3)
    idx_to_class = {index: name for name, index in mapping.items()}
    top3 = [{"class_idx": index, "class_name": idx_to_class[index], "probability": value}
            for value, index in zip(values.tolist(), indices.tolist())]
    result = {"purpose": checkpoint.get("purpose", "smoke/debug"), "limitation": checkpoint["limitation"],
              "split_id": checkpoint.get("split_id"), "test_images_opened": 0,
              "pid": os.getpid(), "checkpoint": str(args.checkpoint.resolve()), "image": str(image),
              "training_pid": reference["training_pid"], "independent_process_verified": True,
              "device": "cuda:0", "output_shape": list(logits.shape), "class_mapping_verified": True,
              "input_tensor_identical": True, "reload_prediction_matches": True,
              "logits_max_abs_difference": (logits - expected).abs().max().item(),
              "probability_max_abs_difference": (probabilities - expected_probabilities).abs().max().item(),
              "comparison_tolerance": {"rtol": 1e-5, "atol": 1e-6}, "top3": top3,
              "probabilities_sum": probabilities.sum().item(), "seconds": time.perf_counter() - started}
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
