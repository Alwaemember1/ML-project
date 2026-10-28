"""创建老师演示交付副本；不删除原件，不训练，不修改原冻结记录或模型。"""

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "reports/course_submission_20261019_v1"


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-archive", action="store_true", help="仅创建目录，供后续离线环境打包使用")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "deliverables"):
        raise ValueError("交付副本必须位于本项目deliverables目录")
    from project_config import default_record_path, resolve_record_file
    record_file = default_record_path()
    record = json.loads(record_file.read_text(encoding="utf-8"))
    original_record_hash = sha256(record_file)
    checkpoint = resolve_record_file(record["checkpoint"], record_file)
    if sha256(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("冻结模型SHA256不符，停止打包")
    output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def copy(source, target):
        source, target = Path(source), output / target
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as file, source.open("rb") as origin:
            shutil.copyfileobj(origin, file)
        sources[str(target.relative_to(output)).replace("\\", "/")] = sha256(source)

    for name in ("app.py", "inference.py", "smoke_utils.py", "class_labels.py", "plant_care.py",
                 "predict_image.py", "project_config.py", "requirements.txt", "requirements-web.txt", "requirements-training.txt",
                 "requirements-cpu.txt", "requirements-gpu.txt",
                 "启动网页.bat", ".streamlit/config.toml"):
        copy(ROOT / name, name)
    copy(checkpoint, "models/final_resnet18.pt")
    copy(resolve_record_file(record["evaluation_metrics"], record_file), "results/metrics.json")
    # 只改交付副本中的位置字段；权重、类别、预处理、划分标识及模型SHA保持不变。
    portable = dict(record)
    portable["checkpoint"] = "models/final_resnet18.pt"
    portable["evaluation_metrics"] = "results/metrics.json"
    portable["delivery_notes"] = "未使用的训练/划分历史字段保持原实验记录；正常推理不访问这些位置。"
    (output / "FINAL_MODEL.json").write_text(json.dumps(portable, ensure_ascii=False, indent=2), encoding="utf-8")
    # 保留完整实验源码供审查；运行时只需要上面的少量文件。
    for path in sorted(ROOT.glob("*.py")):
        copy(path, f"experiment_source/{path.name}")
    for folder in ("scripts", "tests"):
        for path in sorted((ROOT / folder).glob("*.py")):
            copy(path, f"experiment_source/{folder}/{path.name}")
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        if (ROOT / name).is_file():
            copy(ROOT / name, name)
    for name in ("dataset_config.json", "final_test_registry.json", "COURSE_REQUIREMENTS.md", "PROJECT_STATUS.md"):
        copy(ROOT / name, f"experiment_source/{name}")
    for path in sorted((ROOT / "splits/formal_sha256_seed42_v1").glob("*")):
        if path.is_file():
            copy(path, f"experiment_source/splits/formal_sha256_seed42_v1/{path.name}")
    copy(record_file, "experiment_source/FINAL_MODEL.original.json")
    for name in ("ACCEPTANCE.md", "CARE_REFERENCE.md", "COURSE_REPORT_FINAL.md", "COURSE_REPORT_FINAL.html",
                 "DEMO_AND_DEFENSE.md", "report_layout_verification.json", "sources.json"):
        copy(COURSE / name, f"docs/{name}")
    for folder in ("assets", "evidence"):
        for path in sorted((COURSE / folder).rglob("*")):
            if path.is_file():
                copy(path, f"docs/{path.relative_to(COURSE).as_posix()}")
    copy(ROOT / "docs/TEACHER_HANDOFF.md", "docs/TEACHER_HANDOFF.md")
    copy(ROOT / "docs/TEACHER_README.md", "README.md")
    # 新的交付说明覆盖旧文档的历史端口提示，不修改已有报告本身。
    if sha256(record_file) != original_record_hash or sha256(checkpoint) != record["checkpoint_sha256"]:
        raise AssertionError("原冻结记录或模型在打包期间改变")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                "original_record_sha256": original_record_hash, "model_sha256": record["checkpoint_sha256"],
                "model_count": 1, "raw_dataset_included": False, "venv_included": False,
                "records": [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                             "sha256": sha256(path)} for path in files]}
    (output / "DELIVERY_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.no_archive:
        print(json.dumps({"output": str(output), "archive_created": False}, ensure_ascii=False))
        return
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zipped.write(path, arcname=f"{output.name}/{path.relative_to(output).as_posix()}")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise AssertionError("ZIP完整性检查失败")
    archive.with_suffix(".zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"output": str(output), "zip": str(archive), "zip_bytes": archive.stat().st_size,
                      "model_sha256": record["checkpoint_sha256"], "originals_unchanged": True,
                      "training_or_test_evaluation": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
