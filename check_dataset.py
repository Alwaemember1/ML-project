"""只读检查分类图片：python check_dataset.py（在 PyCharm 中也可直接运行）。

类别文件夹名就是未来的标签；本脚本不划分数据、不训练、不写入原始目录。
"""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import textwrap
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent
# 与常见的 ImageFolder 图片后缀一致；不依赖文件名来判断图片是否损坏。
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".pbm", ".pnm"}


def project_path(value):
    """相对路径以脚本所在目录为起点，避免 PyCharm 工作目录不同导致找不到数据。"""
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def environment_info():
    """记录实际运行的解释器，而不是猜测终端中的 python 指向哪里。"""
    packages = {}
    for name in ("Pillow", "torch", "torchvision", "streamlit"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, errors="replace", timeout=15, check=False,
        )
        gpu_status = (gpu.stdout or gpu.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu_status = str(exc)
    return {
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "is_virtual_environment": sys.prefix != sys.base_prefix,
        "platform": platform.platform(),
        "packages": packages,
        "nvidia_smi": gpu_status,
        "cuda_in_pytorch_tested": False,
    }


def iter_class_files(folder, skipped_links):
    """递归读取类别内的文件，但不跟随链接，避免检查范围越过数据集。"""
    def raise_walk_error(error):
        raise error  # 不允许权限问题悄悄造成少统计。

    for current, dirs, files in os.walk(folder, followlinks=False, onerror=raise_walk_error):
        keep_dirs = []
        for name in sorted(dirs):
            path = Path(current) / name
            if path.is_symlink() or path.is_junction():
                skipped_links.append(str(path))
            else:
                keep_dirs.append(name)
        dirs[:] = keep_dirs
        for name in sorted(files):
            path = Path(current) / name
            if path.is_symlink():
                skipped_links.append(str(path))
            else:
                yield path


def load_font(size):
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError:
        return ImageFont.load_default(size=size)


def save_samples(samples, output_dir):
    """每类随机选一张可解码图片，缩略图只写到报告目录。"""
    columns, tile_width, tile_height = 5, 280, 265
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#edf2f0")
    draw = ImageDraw.Draw(sheet)
    font = load_font(14)
    sample_rows = []
    for index, (class_name, source, thumbnail) in enumerate(samples):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        if thumbnail is not None:
            sheet.paste(thumbnail, (x + (tile_width - thumbnail.width) // 2, y + 8))
            thumbnail.close()
        else:
            draw.text((x + 20, y + 70), "No valid image", fill="red", font=font)
        label = "\n".join(textwrap.wrap(f"{index}: {class_name}", width=32))
        draw.multiline_text((x + 10, y + 195), label, fill="#17261e", font=font, spacing=3)
        sample_rows.append({"class_name": class_name, "source_path": source})
    sheet.save(output_dir / "samples.jpg", quality=92)
    sheet.close()
    write_json(output_dir / "sample_sources.json", sample_rows)


def run_check(dataset_root, output_root, seed):
    if not dataset_root.is_dir():
        raise ValueError(f"数据集目录不存在：{dataset_root}")
    if output_root == dataset_root or output_root.is_relative_to(dataset_root):
        raise ValueError("报告目录不能位于原始数据集内部，请使用项目下的 reports。")

    children = sorted(dataset_root.iterdir(), key=lambda path: path.name)
    links = [str(path) for path in children if path.is_symlink() or path.is_junction()]
    classes = [path for path in children if path.is_dir() and str(path) not in links and not path.name.startswith(".")]
    if not classes:
        raise ValueError("找不到类别子目录。dataset_root 应直接包含各类别文件夹。")
    root_images = [path.name for path in children if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if root_images:
        raise ValueError("根目录存在未归类图片，请确认 dataset_root 指向类别文件夹的共同父目录。")
    split_names = {"train", "training", "val", "valid", "validation", "test", "testing"}
    if any(folder.name.lower() in split_names for folder in classes):
        raise ValueError("检测到训练/验证/测试层。请把 dataset_root 改成其中直接包含类别的一层。")

    # 每次新建报告目录，重复运行也不会覆盖上一次的结果。
    output_dir = output_root / datetime.now().strftime("check_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "environment.json", environment_info())
    started = datetime.now()
    rng = random.Random(seed)
    counts, corrupted, image_warnings, samples = [], [], [], []
    sizes, modes, extensions = Counter(), Counter(), Counter()
    root_other_files = [path.name for path in children if path.is_file()]
    manifest_fields = ["class_name", "relative_path", "status", "width", "height", "mode", "error"]

    print(f"数据集根目录：{dataset_root}", flush=True)
    print(f"报告目录：{output_dir}", flush=True)
    # newline='' 避免 Windows 下 CSV 出现多余空行；BOM 方便 Excel 显示中文。
    with (output_dir / "image_manifest.csv").open("w", encoding="utf-8-sig", newline="") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=manifest_fields)
        writer.writeheader()
        for class_id, folder in enumerate(classes):
            paths = list(iter_class_files(folder, links))
            image_paths = [path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS]
            ignored = len(paths) - len(image_paths)
            rng.shuffle(image_paths)
            valid, bad, thumbnail, sample_path = 0, 0, None, None
            for path in image_paths:
                relative = path.relative_to(dataset_root).as_posix()
                row = dict.fromkeys(manifest_fields, "")
                row.update(class_name=folder.name, relative_path=relative)
                recorded = []
                try:
                    with warnings.catch_warnings(record=True) as recorded:
                        warnings.simplefilter("always")
                        with Image.open(path) as image:
                            image.verify()  # 检查文件结构。
                        # verify 后必须重新打开；load 强制解码，能发现部分截断图片。
                        with Image.open(path) as image:
                            image.load()
                            width, height = image.size
                            mode = image.mode
                            if thumbnail is None:
                                preview = ImageOps.exif_transpose(image).convert("RGB")
                                preview.thumbnail((260, 180))
                                thumbnail = preview.copy()
                                preview.close()
                                sample_path = relative
                    valid += 1
                    sizes[f"{width}x{height}"] += 1
                    modes[mode] += 1
                    extensions[path.suffix.lower()] += 1
                    row.update(status="valid", width=width, height=height, mode=mode)
                except Exception as exc:
                    # 单张坏图不终止整个检查；记录路径，原图保留原位。
                    bad += 1
                    error = f"{type(exc).__name__}: {exc}"
                    corrupted.append({"class_name": folder.name, "relative_path": relative, "error": error})
                    row.update(status="corrupted", error=error)
                for warning in recorded:
                    image_warnings.append({"relative_path": relative, "warning": str(warning.message)})
                writer.writerow(row)
            counts.append({"class_id": class_id, "class_name": folder.name,
                           "total_images": len(image_paths), "valid_images": valid,
                           "corrupted_images": bad, "ignored_non_image_files": ignored})
            samples.append((folder.name, sample_path, thumbnail))
            print(f"[{class_id + 1}/{len(classes)}] {folder.name}: 共 {len(image_paths)}，可读 {valid}，损坏 {bad}", flush=True)

    total = sum(item["total_images"] for item in counts)
    if total == 0:
        raise ValueError("没有发现支持的图片。可能选中了外层目录；请检查真正的类别根目录。")
    with (output_dir / "class_counts.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    write_json(output_dir / "class_to_idx.json", {item["class_name"]: item["class_id"] for item in counts})
    write_json(output_dir / "corrupted_images.json", corrupted)
    write_json(output_dir / "image_warnings.json", image_warnings)
    save_samples(samples, output_dir)
    valid_counts = [item["valid_images"] for item in counts]
    summary = {
        "dataset_root": str(dataset_root), "output_dir": str(output_dir),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "sample_seed": seed, "class_count": len(classes),
        "total_images": total, "valid_images": total - len(corrupted),
        "corrupted_images": len(corrupted), "warning_count": len(image_warnings),
        "ignored_non_image_files": sum(item["ignored_non_image_files"] for item in counts),
        "root_non_image_files": root_other_files, "skipped_links": links,
        "min_class_images": min(valid_counts), "max_class_images": max(valid_counts),
        "max_min_ratio": round(max(valid_counts) / min(valid_counts), 3) if min(valid_counts) else None,
        "empty_or_unreadable_classes": [item["class_name"] for item in counts if item["valid_images"] == 0],
        "image_sizes": dict(sizes.most_common()), "image_modes": dict(modes.most_common()),
        "valid_image_extensions": dict(extensions.most_common()), "classes": counts,
        "limits": ["只检查支持后缀的图片能否完整解码，不证明标签正确。",
                   "未检测重复图片或增强图片的原图关联，尚未划分训练/验证/测试集。"],
    }
    write_json(output_dir / "summary.json", summary)
    lines = ["# 数据检查结果", "", f"数据集：`{dataset_root}`", "",
             f"类别：{len(classes)}；图片：{total}；可读：{total - len(corrupted)}；损坏：{len(corrupted)}。", "",
             f"各类可读图片最少 {min(valid_counts)} 张，最多 {max(valid_counts)} 张。", "",
             "| 类别 | 总数 | 可读 | 损坏 |", "| --- | ---: | ---: | ---: |"]
    lines.extend(f"| {item['class_name']} | {item['total_images']} | {item['valid_images']} | {item['corrupted_images']} |" for item in counts)
    lines.extend(["", "![每类一张样例](samples.jpg)", "", "说明：", "",
                  "- 损坏为 0 只表示图片能解码，不代表标签、图片内容或模型效果已验证。",
                  "- Background_without_leaves 若存在，会单独统计，后续需决定是否作为背景类别。",
                  "- 目录名含 augmentation：后续需核实增强来源，避免同源图片跨训练和验证集造成数据泄漏。",
                  "- 本次没有训练模型，也没有写入、移动或删除原始图片。"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n完成：{len(classes)} 类，{total} 张，损坏 {len(corrupted)} 张。", flush=True)
    print(f"查看 {output_dir / 'report.md'} 和 samples.jpg", flush=True)
    return 1 if corrupted or not all(valid_counts) else 0


def main():
    parser = argparse.ArgumentParser(description="只读检查分类图片数据集并生成报告")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "dataset_config.json"))
    parser.add_argument("--dataset-root", help="直接包含各类别文件夹的目录，覆盖配置")
    parser.add_argument("--output-root", help="报告保存目录，不能位于数据集内部")
    args = parser.parse_args()
    config = json.loads(project_path(args.config).read_text(encoding="utf-8-sig"))
    return run_check(project_path(args.dataset_root or config["dataset_root"]),
                     project_path(args.output_root or config["output_root"]),
                     int(config.get("sample_seed", 42)))


if __name__ == "__main__":
    # 相当于 Java 的 main 入口；import 此文件时不会自动扫描数据。
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"检查失败：{error}", file=sys.stderr)
        raise SystemExit(2)
