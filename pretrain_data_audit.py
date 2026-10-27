"""复用第一阶段清单：检查字节重复和来源线索，不重新全量解码、不划分数据。"""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def hash_image(task):
    root, row = task
    relative = row["relative_path"]
    path = (root / relative).resolve()
    try:
        if not path.is_relative_to(root):
            raise ValueError("清单路径越过数据集范围")
        before = path.stat()
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("文件在检查过程中发生变化")
        return {"class_name": row["class_name"], "relative_path": relative,
                "bytes": before.st_size, "sha256": digest, "error": ""}
    except (OSError, ValueError) as exc:
        return {"class_name": row["class_name"], "relative_path": relative,
                "bytes": "", "sha256": "", "error": f"{type(exc).__name__}: {exc}"}


def nearby_inventory(root):
    # 仅检查项目、数据集根目录、D: 顶层及少数相邻项目的第一层；绝不递归扫描磁盘。
    parents = [ROOT, root, ROOT.parent]
    for name in ("PythonProject", "PythonProject1", "PythonProject2", "ML_PY", "ML__Py"):
        candidate = ROOT.parent / name
        if candidate.is_dir():
            parents.append(candidate)
    records = []
    for parent in dict.fromkeys(parents):
        try:
            entries = sorted(parent.iterdir())
            matches = [str(path) for path in entries if re.search(
                r"plant|leaf|leav|dataset|archive|augment|original|raw", path.name, re.I)]
            records.append({"directory": str(parent), "depth": 1,
                            "matching_entries": matches, "entry_count": len(entries)})
        except OSError as exc:
            records.append({"directory": str(parent), "error": str(exc)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="第一阶段报告目录")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT / "dataset_config.json").read_text(encoding="utf-8-sig"))
    data = (ROOT / config["dataset_root"]).resolve()
    output = args.output_dir.resolve()
    if output.is_relative_to(data):
        raise ValueError("报告不能位于原始数据集内部")
    output.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline.resolve()
    summary = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    if Path(summary["dataset_root"]).resolve() != data:
        raise ValueError("配置数据根目录与旧报告不一致")
    with (baseline / "image_manifest.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != summary["total_images"]:
        raise ValueError("旧报告图片总数与清单不一致")
    save_json(output / "nearby_inventory.json", nearby_inventory(data))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["class_name"]].append(row)
    rng = random.Random(42)
    filename_stats = []
    for category, items in grouped.items():
        names = sorted(Path(row["relative_path"]).name for row in items)
        numbered = sum(bool(re.fullmatch(r"image\s*\(\d+\)\.(jpg|jpeg|png)", name, re.I)) for name in names)
        filename_stats.append({"class_name": category, "count": len(items),
                               "generic_numbered_names": numbered,
                               "examples": rng.sample(names, min(5, len(names)))})
    save_json(output / "filename_evidence.json", filename_stats)

    # 仅选 6 类、每类 3 张读取元数据并解码，供人工看样；不是全量解码检查。
    picked = []
    for category in ("Apple___Apple_scab", "Apple___Cedar_apple_rust", "Background_without_leaves",
                     "Potato___healthy", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"):
        picked.extend(rng.sample(grouped[category], 3))
    sheet = Image.new("RGB", (900, 6 * 205), "white")
    draw = ImageDraw.Draw(sheet)
    metadata = []
    for index, row in enumerate(picked):
        with Image.open(data / row["relative_path"]) as image:
            metadata.append({"relative_path": row["relative_path"], "format": image.format,
                             "size": image.size, "info": {key: str(value)[:2000] for key, value in image.info.items()},
                             "exif": {str(key): str(value)[:2000] for key, value in image.getexif().items()}})
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((280, 160))
            x, y = index % 3 * 300, index // 3 * 205
            sheet.paste(preview, (x, y))
            draw.text((x, y + 162), row["class_name"][:43], fill="black")
            draw.text((x, y + 180), Path(row["relative_path"]).name, fill="black")
            preview.close()
    if (output / "provenance_samples.jpg").exists():
        raise FileExistsError("拒绝覆盖已有样例")
    sheet.save(output / "provenance_samples.jpg")
    sheet.close()
    save_json(output / "sample_metadata.json", metadata)

    hashes, errors = defaultdict(list), []
    with (output / "sha256_manifest.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["class_name", "relative_path", "bytes", "sha256", "error"])
        writer.writeheader()
        # 只读文件字节，不调用图片解码；按文件做 SHA256 是线性工作量。
        with ThreadPoolExecutor(max_workers=8) as pool:
            for index, record in enumerate(pool.map(hash_image, ((data, row) for row in rows)), 1):
                writer.writerow(record)
                if record["error"]:
                    errors.append(record)
                else:
                    hashes[record["sha256"]].append(record)
                if index % 5000 == 0:
                    print(f"SHA256 已完成 {index}/{len(rows)}", flush=True)
    duplicates = [{"sha256": digest, "files": items} for digest, items in hashes.items() if len(items) > 1]
    cross_class = [item for item in duplicates if len({row["class_name"] for row in item["files"]}) > 1]
    save_json(output / "exact_duplicate_groups.json", duplicates)
    save_json(output / "hash_errors.json", errors)
    result = {"baseline_report": str(baseline), "dataset_root": str(data), "manifest_count": len(rows),
              "hashed_files": len(rows) - len(errors), "hash_errors": len(errors),
              "unique_sha256": len(hashes), "duplicate_groups": len(duplicates),
              "files_in_duplicate_groups": sum(len(item["files"]) for item in duplicates),
              "redundant_copies": sum(len(item["files"]) - 1 for item in duplicates),
              "cross_class_duplicate_groups": len(cross_class), "decoded_sample_count": len(picked),
              "generic_numbered_names": sum(item["generic_numbered_names"] for item in filename_stats),
              "reliable_original_grouping_established": False,
              "final_split_created": False,
              "hash_limit": "文件 SHA256 只发现字节完全相同的文件，不能排除旋转、翻转、缩放或重新编码后的同源图片泄漏。"}
    save_json(output / "data_audit_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
