"""校验已取得的冻结权重，创建本地推理配置；不下载、不加载或修改模型。"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from project_config import ROOT, FROZEN_SHA256


def configure(checkpoint, output=None, *, reference=None, metrics=None):
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output or ROOT / "FINAL_MODEL.local.json").resolve()
    reference = Path(reference or ROOT / "FINAL_MODEL.json").resolve()
    metrics = Path(metrics or ROOT / "docs/results/original_test_metrics.json").resolve()
    if output == ROOT / "FINAL_MODEL.json" or output == reference:
        raise ValueError("不能覆盖原始冻结记录")
    if output.exists():
        raise FileExistsError(f"本地配置已存在，拒绝覆盖：{output}")
    with checkpoint.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != FROZEN_SHA256:
        raise ValueError(f"权重不是本项目锁定版本。实际SHA256：{actual}")
    record = json.loads(reference.read_text(encoding="utf-8"))
    if record["checkpoint_sha256"] != actual or record["status"] != "frozen_final_model":
        raise ValueError("原始冻结记录状态或哈希不匹配")
    result = json.loads(metrics.read_text(encoding="utf-8"))
    if result["total"] != 9192 or len(result["classes"]) != 39:
        raise ValueError("不是本项目已有最终测试指标")
    try:
        checkpoint_value = Path(os.path.relpath(checkpoint, output.parent)).as_posix()
    except ValueError:  # Windows不同盘符不能构造相对路径。
        checkpoint_value = str(checkpoint)
    try:
        metrics_value = Path(os.path.relpath(metrics, output.parent)).as_posix()
    except ValueError:
        metrics_value = str(metrics)
    record.update(checkpoint=checkpoint_value, evaluation_metrics=metrics_value,
                  local_setup_note="仅修改本地文件位置，原始类别、预处理、模型SHA及历史来源字段不变。")
    with output.open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    return {"configuration": str(output), "checkpoint": str(checkpoint), "sha256_verified": True,
            "original_record_changed": False, "weights_downloaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="自行取得的完整冻结检查点")
    parser.add_argument("--output", type=Path, help="默认FINAL_MODEL.local.json，拒绝覆盖")
    args = parser.parse_args()
    try:
        print(json.dumps(configure(args.checkpoint, args.output), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"配置失败：{exc}\n")


if __name__ == "__main__":
    main()
