"""校验已锁定的均衡训练视图；仅读取清单，不访问图片或建立新划分。"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def file_sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_balanced_training(manifest, formal_train, validation_rows, mapping, split_id):
    manifest = Path(manifest).resolve()
    lock_path = manifest.parent / "candidate_balance_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest_hash = file_sha256(manifest)
    if manifest_hash != lock["manifest_sha256"]:
        raise ValueError("均衡候选清单SHA256与锁不符，拒绝训练")
    if lock["source_formal_split_id"] != split_id or file_sha256(formal_train) != lock["source_train_sha256"]:
        raise ValueError("均衡视图来源不是当前固定正式训练清单")
    fields = ("relative_path", "class_name", "class_idx", "sha256")
    with Path(formal_train).open(encoding="utf-8-sig", newline="") as file:
        original_rows = list(csv.DictReader(file))
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    originals = {tuple(row[field] for field in fields) for row in original_rows}
    if not rows or any(tuple(row[field] for field in fields) not in originals for row in rows):
        raise ValueError("均衡清单含有不属于正式训练集的记录")
    for row in rows:
        if row["class_name"] not in mapping or int(row["class_idx"]) != mapping[row["class_name"]]:
            raise ValueError("均衡清单类别或编号不符")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("均衡清单出现不安全的图片相对路径")
    paths, hashes = {row["relative_path"] for row in rows}, {row["sha256"] for row in rows}
    if len(paths) != len(rows) or len(hashes) != len(rows):
        raise ValueError("均衡清单内部存在重复路径或SHA256")
    counts = Counter(row["class_name"] for row in rows)
    if (set(counts) != set(mapping) or set(counts.values()) != {lock["per_class"]}
            or len(rows) != lock["total"] or len(mapping) != lock["classes"]):
        raise ValueError("均衡清单未保持全部类别等量或总数不符")
    if paths & {row["relative_path"] for row in validation_rows} or hashes & {row["sha256"] for row in validation_rows}:
        raise ValueError("均衡训练视图与正式验证集存在交集")
    check = {"manifest":str(manifest), "manifest_sha256":manifest_hash,
             "lock":str(lock_path), "lock_sha256":file_sha256(lock_path),
             "source_formal_train_sha256":lock["source_train_sha256"], "split_id":split_id,
             "train_count":len(rows), "classes":len(mapping), "per_class":lock["per_class"],
             "all_rows_subset_of_formal_train":True, "path_and_hash_unique":True,
             "validation_path_and_hash_overlap":0, "formal_split_replaced":False,
             "test_images_opened":0, "class_counts":dict(counts)}
    return rows, check
