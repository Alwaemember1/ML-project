"""固定 SHA256 代表文件的正式实验划分；只读已有清单，不读取原图。"""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / "splits" / "formal_sha256_seed42_v1"
BASELINE = ROOT / "reports" / "check_20261016_133031_170321"
AUDIT = ROOT / "reports" / "pretrain_20261016_230415"
LIMITATION = "仅排除已知字节重复；增强同源关系未排除，不能宣称无增强同源泄漏。"


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def write_rows(path, rows, fields):
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_split(directory=SPLIT):
    directory = Path(directory)
    lock = json.loads((directory / "split_lock.json").read_text(encoding="utf-8"))
    for name, expected in lock["artifact_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"固定划分文件已改变：{name}，拒绝训练或自动重建")
    canonical = json.dumps(lock["artifact_sha256"], sort_keys=True).encode()
    if hashlib.sha256(canonical).hexdigest() != lock["split_id"]:
        raise ValueError("划分标识不匹配")
    config = json.loads((directory / "split_config.json").read_text(encoding="utf-8"))
    for source, expected in config["source_sha256"].items():
        if digest(source) != expected:
            raise ValueError(f"来源清单已改变：{source}")
    mapping = json.loads((directory / "class_to_idx.json").read_text(encoding="utf-8"))
    rows = {part: read_rows(directory / f"{part}.csv") for part in ("train", "validation", "test")}
    path_sets = {part: {row["relative_path"] for row in items} for part, items in rows.items()}
    hash_sets = {part: {row["sha256"] for row in items} for part, items in rows.items()}
    for part, items in rows.items():
        if len(items) != len(path_sets[part]) or len(items) != len(hash_sets[part]):
            raise ValueError(f"{part} 内部有重复")
        if {row["class_name"] for row in items} != set(mapping):
            raise ValueError(f"{part} 类别不完整")
        for row in items:
            if int(row["class_idx"]) != mapping[row["class_name"]]:
                raise ValueError("标签编号不匹配")
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{left}_{right}"] = {"paths": len(path_sets[left] & path_sets[right]),
                                     "sha256": len(hash_sets[left] & hash_sets[right])}
    if any(value for pair in overlaps.values() for value in pair.values()):
        raise ValueError("跨集合路径或 SHA256 重复")
    smoke_hashes = set()
    for path in config["smoke_manifests"]:
        smoke_hashes.update(row["sha256"] for row in read_rows(path))
    if not smoke_hashes.issubset(hash_sets["train"]) or smoke_hashes & (hash_sets["validation"] | hash_sets["test"]):
        raise ValueError("smoke 样本未被全部锁入训练集")
    if sum(len(items) for items in rows.values()) != config["unique_representatives"]:
        raise ValueError("划分总数不正确")
    return {"split_id": lock["split_id"], "counts": {part: len(items) for part, items in rows.items()},
            "overlaps": overlaps, "smoke_unique_hashes": len(smoke_hashes),
            "all_smoke_hashes_train_only": True, "all_classes_in_each_split": True,
            "test_images_opened": 0, "limitation": LIMITATION}


def create_split():
    if SPLIT.exists():
        raise FileExistsError("固定划分目录已存在；只能校验，不能自动重新生成")
    images = read_rows(BASELINE / "image_manifest.csv")
    hashes = read_rows(AUDIT / "sha256_manifest.csv")
    mapping = json.loads((BASELINE / "class_to_idx.json").read_text(encoding="utf-8"))
    if len(mapping) != 39 or sorted(mapping.values()) != list(range(39)):
        raise ValueError("必须保留原 39 类映射")
    image_map = {row["relative_path"]: row for row in images}
    if len(images) != len(hashes) or len(image_map) != 61486:
        raise ValueError("已有清单数量不匹配")
    representatives = {}
    for row in sorted(hashes, key=lambda item: item["relative_path"]):
        original = image_map[row["relative_path"]]
        if row["error"] or original["status"] != "valid" or row["class_name"] != original["class_name"]:
            raise ValueError("已有清单中有错误或类别不匹配")
        candidate = {"relative_path": row["relative_path"], "class_name": row["class_name"],
                     "class_idx": mapping[row["class_name"]], "sha256": row["sha256"]}
        previous = representatives.setdefault(row["sha256"], candidate)
        if previous["class_name"] != row["class_name"]:
            raise ValueError("跨类别相同哈希冲突")
    smoke_paths = sorted(path for folder in (ROOT / "reports").glob("smoke_*") if folder.is_dir()
                         for path in (folder / "smoke_train.csv", folder / "smoke_validation.csv") if path.is_file())
    if not smoke_paths:
        raise ValueError("未找到 smoke 清单")
    smoke_rows = [row for path in smoke_paths for row in read_rows(path)]
    smoke_hashes = {row["sha256"] for row in smoke_rows}
    if not smoke_hashes.issubset(representatives):
        raise ValueError("smoke 哈希在现有清单中缺失")
    buckets = defaultdict(list)
    for row in representatives.values():
        buckets[row["class_name"]].append(row)
    sets = {part: [] for part in ("train", "validation", "test")}
    rng = random.Random(42)
    per_class = []
    for name in sorted(mapping, key=mapping.get):
        items = sorted(buckets[name], key=lambda item: item["relative_path"])
        forced = [row for row in items if row["sha256"] in smoke_hashes]
        remaining = [row for row in items if row["sha256"] not in smoke_hashes]
        rng.shuffle(remaining)
        n = len(items)
        n_val = n_test = (n * 15 + 50) // 100
        n_train = n - n_val - n_test
        if n_train < len(forced) or min(n_val, n_test) < 1:
            raise ValueError("类别数量不满足强制归属与比例约束")
        extra = n_train - len(forced)
        sets["train"].extend(forced + remaining[:extra])
        sets["validation"].extend(remaining[extra:extra + n_val])
        sets["test"].extend(remaining[extra + n_val:])
        per_class.append({"class_name": name, "class_idx": mapping[name], "unique_images": n,
                          "train": n_train, "validation": n_val, "test": n_test, "forced_smoke_train": len(forced)})
    sources = [BASELINE / "image_manifest.csv", BASELINE / "class_to_idx.json",
               AUDIT / "sha256_manifest.csv", ROOT / "dataset_config.json", *smoke_paths]
    data_config = json.loads((ROOT / "dataset_config.json").read_text(encoding="utf-8-sig"))
    config = {"version": "formal_sha256_seed42_v1", "seed": 42,
              "dataset_root": str((ROOT / data_config["dataset_root"]).resolve()), "class_count": 39,
              "original_files": len(hashes), "unique_representatives": len(representatives),
              "excluded_byte_copies": len(hashes) - len(representatives),
              "representative_rule": "每种 SHA256 取相对路径字典序最小者",
              "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
              "rounding": "每类 val=test=整数四舍五入(15%×该类哈希数)，其余训练",
              "smoke_rule": "所有 smoke 使用过的哈希（含其副本）仅允许归属正式训练集",
              "smoke_manifests": [str(path) for path in smoke_paths],
              "smoke_rows": len(smoke_rows), "smoke_unique_hashes": len(smoke_hashes),
              "source_sha256": {str(path): digest(path) for path in sources},
              "limitation": LIMITATION, "test_policy": "仅保存和核对元数据；不读测试图片、不选模、不调参、不挑演示图"}
    SPLIT.mkdir(parents=True, exist_ok=False)
    fields = ["relative_path", "class_name", "class_idx", "sha256"]
    for part, rows in sets.items():
        write_rows(SPLIT / f"{part}.csv", sorted(rows, key=lambda row: row["relative_path"]), fields)
    write_rows(SPLIT / "class_counts.csv", per_class, list(per_class[0]))
    write_json(SPLIT / "class_to_idx.json", mapping)
    write_json(SPLIT / "split_config.json", config)
    artifacts = {name: digest(SPLIT / name) for name in
                 ("train.csv", "validation.csv", "test.csv", "class_counts.csv", "class_to_idx.json", "split_config.json")}
    split_id = hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest()
    write_json(SPLIT / "split_lock.json", {"split_id": split_id, "artifact_sha256": artifacts,
                                          "policy": "拒绝覆盖；训练前校验文件和来源哈希；不静默重建"})
    result = verify_split()
    write_json(SPLIT / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true", help="仅首次显式创建，已有目录则拒绝")
    args = parser.parse_args()
    print(json.dumps(create_split() if args.create else verify_split(), ensure_ascii=False, indent=2))
