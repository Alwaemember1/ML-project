"""固定类别的单标签多分类指标；不依赖 sklearn，不训练模型。"""

import math


def classification_metrics(matrix, class_names):
    n = len(class_names)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("混淆矩阵形状错误")
    if any(not isinstance(value, int) or value < 0 for row in matrix for value in row):
        raise ValueError("混淆计数必须为非负整数")
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise ValueError("没有评价样本")
    classes = []
    for index, name in enumerate(class_names):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        fp, fn = predicted - tp, support - tp
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        classes.append({"class_idx": index, "class_name": name, "precision": precision,
                        "recall": recall, "f1": f1, "support": support, "predicted_count": predicted,
                        "tp": tp, "fp": fp, "fn": fn, "no_predicted_samples": predicted == 0})
    correct = sum(matrix[index][index] for index in range(n))
    return {"total": total, "correct": correct, "errors": total - correct,
            "accuracy": correct / total,
            "macro_precision": math.fsum(row["precision"] for row in classes) / n,
            "macro_recall": math.fsum(row["recall"] for row in classes) / n,
            "macro_f1": math.fsum(row["f1"] for row in classes) / n,
            "classes": classes,
            "zero_division_policy": "分母为0的precision/recall/F1记0，固定全部39类参与macro平均；不删类。"}


def confusion_pairs(matrix, class_names):
    pairs = []
    for true_idx, row in enumerate(matrix):
        support, errors = sum(row), sum(row) - row[true_idx]
        for pred_idx, count in enumerate(row):
            if true_idx != pred_idx and count:
                pairs.append({"true_idx": true_idx, "true_class": class_names[true_idx],
                              "pred_idx": pred_idx, "pred_class": class_names[pred_idx], "count": count,
                              "true_support": support, "fraction_of_true_class": count / support,
                              "fraction_of_source_errors": count / errors})
    return sorted(pairs, key=lambda row: (-row["count"], row["true_idx"], row["pred_idx"]))


def audit_records(records, class_names, matrix, metrics):
    """只重读预测记录复核，不再次运行模型。"""
    rebuilt = [[0] * len(class_names) for _ in class_names]
    paths, hashes, losses = set(), set(), []
    for record in records:
        true_idx, pred_idx = record["true_idx"], record["pred_idx"]
        if record["true_class"] != class_names[true_idx] or record["pred_class"] != class_names[pred_idx]:
            raise AssertionError("逐图记录的编号与类别不一致")
        if len(record["probabilities"]) != len(class_names) or len(record["logits"]) != len(class_names):
            raise AssertionError("逐图分数维度错误")
        if not math.isclose(math.fsum(record["probabilities"]), 1, abs_tol=1e-5):
            raise AssertionError("概率和不是1")
        ordered = sorted(range(len(class_names)), key=lambda idx: -record["logits"][idx])
        if ordered[0] != pred_idx or [item["class_idx"] for item in record["top3"]] != ordered[:3]:
            raise AssertionError("Top3 与原始分数不匹配")
        maximum = max(record["logits"])
        recomputed_loss = maximum + math.log(math.fsum(math.exp(value - maximum) for value in record["logits"])) - record["logits"][true_idx]
        if not math.isclose(record["loss"], recomputed_loss, abs_tol=1e-5, rel_tol=1e-5):
            raise AssertionError("逐图交叉熵与原始分数不符")
        rebuilt[true_idx][pred_idx] += 1
        paths.add(record["relative_path"])
        hashes.add(record["sha256"])
        losses.append(record["loss"])
    if rebuilt != matrix or len(paths) != len(records) or len(hashes) != len(records):
        raise AssertionError("预测记录重建矩阵或唯一性核对失败")
    recomputed = classification_metrics(rebuilt, class_names)
    for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if not math.isclose(recomputed[name], metrics[name], abs_tol=1e-12):
            raise AssertionError(f"指标复核失败：{name}")
    if not math.isclose(math.fsum(losses) / len(losses), metrics["loss"], abs_tol=1e-12):
        raise AssertionError("平均loss复核失败")
    return {"records_replayed": len(records), "matrix_rebuilt_identical": True,
            "metrics_recomputed_identical": True, "loss_verified_from_logits": True,
            "new_model_inference_performed": False}
