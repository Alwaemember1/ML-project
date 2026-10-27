"""导出GitHub源码候选ZIP，不带Git历史、环境、模型、原图或本机配置。"""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from project_config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="deliverables内不存在的.zip路径")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "deliverables") or output.suffix.lower() != ".zip":
        parser.error("输出必须是本项目deliverables内的.zip")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖：{output}")
    result = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=ROOT)
    candidates = sorted(set(result.decode("utf-8").split("\0")) - {""})
    blocked = {".git", ".venv", ".codex", ".agents", ".idea", "reports", "deliverables", "models",
               "__pycache__", ".runtime", "Plant_leave_diseases_dataset_with_augmentation"}
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in candidates:
            relative = Path(name)
            path = ROOT / relative
            if not path.is_file():  # Git可能仍列出尚未提交的已迁移路径。
                continue
            if blocked.intersection(relative.parts) or relative.suffix.lower() in {".pt", ".pth", ".zip", ".pem", ".key", ".pyc"}:
                raise ValueError(f"源码候选包含不应发布的文件：{name}")
            if relative.name == "FINAL_MODEL.local.json" or (relative.name.startswith("FINAL_MODEL.") and relative.name.endswith(".local.json")):
                raise ValueError(f"源码候选包含本机配置：{name}")
            data = path.read_bytes()
            archive.writestr(relative.as_posix(), data)
            records.append({"path": relative.as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("SOURCE_EXPORT.json", json.dumps({"kind": "source_only", "files": records,
                          "weights_and_data_included": False, "github_published": False}, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise AssertionError("源码ZIP校验失败")
    print(json.dumps({"output": str(output), "files": len(records), "bytes": output.stat().st_size,
                      "github_published": False, "model_or_data_included": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
