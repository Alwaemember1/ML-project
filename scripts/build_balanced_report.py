"""读取已完成均衡实验的真实记录，生成课程对照附录；不加载模型或图片。"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from balanced_manifest import file_sha256
from evaluation_metrics import classification_metrics


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    summary = read_json(run / "training_summary.json")
    comparison = read_json(run / "validation_comparison/comparison.json")
    reload = read_json(run / "best_validation_prediction.json")
    before = read_json(run / "frozen_web_before.json")
    after = read_json(run / "frozen_web_after.json")
    config = summary["training_config"]
    balance = config["balanced_training_view"]
    if summary["purpose"] != "balanced_subset_head_comparison" or not reload["reload_prediction_matches"]:
        raise AssertionError("实验类型或独立重载验收不符")
    if any(comparison[key] for key in ("test_evaluated", "web_model_replaced")):
        raise AssertionError("实验超出验证对照范围")
    if any(before[key] != after[key] for key in before):
        raise AssertionError("冻结网页模型变化")
    root = Path(__file__).resolve().parents[1]
    if file_sha256(root / "FINAL_MODEL.json") != before["record_sha256"]:
        raise AssertionError("冻结记录变化")
    if file_sha256(before["checkpoint"]) != before["checkpoint_sha256"]:
        raise AssertionError("原网页检查点变化")
    for row in summary["history"]:
        epoch = read_json(run / f"epoch_{row['epoch']:03}.json")
        frozen = epoch["frozen_check"]
        if (not frozen["backbone_parameters_and_buffers_unchanged"]
                or not frozen["backbone_gradients_absent"]
                or frozen["checked_state_tensors"] != 120
                or frozen["batchnorm_buffer_tensors"] != 60):
            raise AssertionError("逐轮冻结核对未通过")
    if not summary["head_updated"]:
        raise AssertionError("分类头未更新")
    for name in ("baseline", "balanced"):
        if file_sha256(comparison[f"{name}_checkpoint"]) != comparison[f"{name}_sha256"]:
            raise AssertionError("对照模型哈希变化")
    classes = comparison["metrics"]["baseline"]["classes"]
    names = [row["class_name"] for row in classes]
    matrices = {name: [[0] * 39 for _ in range(39)] for name in ("baseline", "balanced")}
    losses = {name: [] for name in matrices}
    paths, hashes = set(), set()
    with (run / "validation_comparison/predictions.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            label = int(row["true_idx"])
            if names[label] != row["true_class"]:
                raise AssertionError("逐图真实类别编号不符")
            if row["relative_path"] in paths or row["sha256"] in hashes:
                raise AssertionError("验证记录路径/哈希重复")
            paths.add(row["relative_path"])
            hashes.add(row["sha256"])
            for name in matrices:
                predicted = int(row[f"{name}_pred_idx"])
                if names[predicted] != row[f"{name}_pred_class"]:
                    raise AssertionError("逐图预测类别编号不符")
                matrices[name][label][predicted] += 1
                losses[name].append(float(row[f"{name}_loss"]))
    if len(paths) != comparison["samples"] or len(paths) != 9192:
        raise AssertionError("验证记录数不符")
    for name in matrices:
        rebuilt = classification_metrics(matrices[name], names)
        if any(rebuilt[key] != comparison["metrics"][name][key] for key in rebuilt):
            raise AssertionError("逐图记录重建指标不符")
        if not math.isclose(math.fsum(losses[name]) / len(paths), comparison["metrics"][name]["loss"], abs_tol=1e-12):
            raise AssertionError("逐图记录重建loss不符")
    audit = {"records": len(paths), "unique_paths_and_hashes": True,
             "both_models_metrics_recomputed_from_predictions": True,
             "no_new_image_read_or_model_inference": True}
    with (run / "validation_comparison/records_audit.json").open("x", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)
    old, new = (comparison["metrics"][name] for name in ("baseline", "balanced"))
    history = summary["history"]
    total_epoch_seconds = sum(row["train_seconds"] + row["validation_seconds"] for row in history)
    best = history[summary["best_epoch"] - 1]
    delta_acc = (new["accuracy"] - old["accuracy"]) * 100
    delta_f1 = (new["macro_f1"] - old["macro_f1"]) * 100
    lines = [
        "# 类别均衡下采样训练对照：课程报告补充", "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。全部数值来自实际训练和固定验证记录。", "",
        "## 实验结论与适用范围", "",
        f"新均衡模型在原固定9192张验证集上的accuracy为{pct(new['accuracy'])}，macro F1为{pct(new['macro_f1'])}；"
        f"原模型同一验证集为{pct(old['accuracy'])}、{pct(old['macro_f1'])}。变化分别为{delta_acc:+.2f}、{delta_f1:+.2f}个百分点。", "",
        "这里只报告验证集对照。原网页继续使用FINAL_MODEL.json锁定的模型；没有新最终测试成绩，原测试成绩不能转用于新模型。"
        "本实验结束后不继续加轮次、调参或自动替换网页模型。", "",
        "## 数据与预先固定的方案", "",
        f"- 原正式训练集42890张；均衡视图39类各{balance['per_class']}张，共{balance['train_count']}张。只抽取已锁定正式训练集的子集，原图未移动或修改。",
        "- 验证集9192张、39类及其原分布不变；验证路径与SHA256同训练均衡视图的交集为0。测试图片读取0张。",
        f"- 固定划分ID：`{comparison['split_id']}`。",
        f"- 训练视图SHA256：`{balance['manifest_sha256']}`；候选锁SHA256：`{balance['lock_sha256']}`。",
        "- 初始化：原有缓存ImageNet ResNet18权重＋新初始化39类分类头，不续训原最佳模型。",
        "- seed42、batch_size32、num_workers0、Adam、CrossEntropyLoss、lr0.001。训练随机裁剪/水平翻转；验证确定性预处理与原模型完全一致。",
        "- 主干及BatchNorm运行统计固定，仅训练分类头。最多10轮，以最低验证loss选模，连续3轮未改善早停。",
        "- 第1轮检查分类头更新和主干冻结通过后，在同一进程及优化器继续训练。",
        f"- 实际设备：{config['gpu']}；Python {config['python'].split()[0]}，torch {config['torch']}，torchvision {config['torchvision']}，未重装依赖。", "",
        "每轮更新次数从原1341降至710，总更新次数还受实际轮数影响。少数类别没有增加新信息，多数类别减少图片。"
        "因此本实验不能把差异单独归因于均衡，也不能证明差异有统计显著性。只运行预先约定的一组配置。", "",
        "## 真实逐轮训练日志", "",
        "| 轮次 | 训练loss | 训练accuracy | 验证loss | 验证accuracy | 训练秒 | 验证秒 | 学习率 | 验证loss改善 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in history:
        lines.append(f"| {row['epoch']} | {row['train_loss']:.6f} | {pct(row['train_accuracy'])} | {row['validation_loss']:.6f} | "
                     f"{pct(row['validation_accuracy'])} | {row['train_seconds']:.2f} | {row['validation_seconds']:.2f} | "
                     f"{row['learning_rate']} | {'是' if row['improved'] else '否'} |")
    lines += ["", f"实际{len(history)}轮，停止原因：`{summary['stopped_reason']}`；最佳为第{summary['best_epoch']}轮，验证loss={best['validation_loss']:.6f}。",
              f"逐轮训练与验证计时合计{total_epoch_seconds:.2f}秒；含检查点等开销、扣除第1轮人工核对等待的训练墙钟时间"
              f"{summary['training_validation_wall_seconds_excluding_gate']:.2f}秒；首次核对等待{summary['first_epoch_review_wait_seconds']:.2f}秒。",
              f"两模型同批验证对照耗时{comparison['seconds']:.2f}秒，独立单图重载验收耗时{reload['seconds']:.2f}秒。", "",
              "![真实训练与验证曲线](training_curves.png)", "",
              "训练使用随机增强，验证采用确定性预处理，两条曲线的输入难度不同，不能只按差距判断模型。"]
    if len(history) > summary["best_epoch"] and history[-1]["train_loss"] < best["train_loss"] and history[-1]["validation_loss"] > best["validation_loss"]:
        lines += ["观察到：最佳轮次之后，最后一轮训练loss更低、验证loss更高，存在继续贴合训练数据而验证收益消失的迹象。按既定规则保留最佳轮次，未扩大训练。"]
    else:
        lines += ["观察到：当前日志不足以单独认定明显过拟合或欠拟合。以实际loss和准确率曲线描述现象，不把未达满分当作程序错误。"]
    lines += ["", "## 相同验证集的完整指标", "",
              "| 指标 | 原冻结模型 | 均衡下采样模型 | 变化（百分点） |", "| --- | --- | --- | --- |"]
    for key, label in (("accuracy", "Accuracy"), ("macro_precision", "Macro precision"), ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")):
        lines.append(f"| {label} | {pct(old[key])} | {pct(new[key])} | {(new[key]-old[key])*100:+.2f} |")
    lines += [f"| Loss | {old['loss']:.6f} | {new['loss']:.6f} | 不适用 |",
              f"| 正确数/总数 | {old['correct']}/{old['total']} | {new['correct']}/{new['total']} | 不适用 |", "",
              "Accuracy按全部图片计算；macro指标先在39类分别计算，再等权平均，每类一票。类别没有预测样本时precision记0，不删除差类别。",
              "均衡改变的是训练分布，验证仍保留原分布。macro指标与accuracy应同时查看。", "",
              "### 新模型recall最低的5类（未按好看结果选择）", "",
              "| 类别 | 验证support | 原recall | 均衡recall |", "| --- | --- | --- | --- |"]
    for entry in sorted(new["classes"], key=lambda row: (row["recall"], row["class_idx"]))[:5]:
        baseline = old["classes"][entry["class_idx"]]
        lines.append(f"| {entry['class_name']} | {entry['support']} | {pct(baseline['recall'])} | {pct(entry['recall'])} |")
    lines += ["", "### 全部39类对照", "", "所有类别均报告，未只列改善类别。support是两模型相同的验证图片数，训练均衡视图中每类均为582张。", "",
              "| 类别 | support | 原precision | 新precision | 原recall | 新recall | 原F1 | 新F1 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for baseline, balanced in zip(old["classes"], new["classes"]):
        lines.append(f"| {baseline['class_name']} | {baseline['support']} | {pct(baseline['precision'])} | {pct(balanced['precision'])} | "
                     f"{pct(baseline['recall'])} | {pct(balanced['recall'])} | {pct(baseline['f1'])} | {pct(balanced['f1'])} |")
    lines += ["", "## 保存、重载与数据记录验收", "",
              "- 每轮主干120个状态张量及60个BatchNorm缓冲均保持固定，主干无梯度，分类头确实更新。",
              f"- 独立进程重载输出1×39，类别映射、输入张量一致；logits最大绝对差{reload['logits_max_abs_difference']}，概率最大绝对差{reload['probability_max_abs_difference']}。",
              "- 新旧模型重新加载后的验证accuracy均与各自最佳轮次的训练日志一致。",
              "- 逐图预测记录重新构造混淆计数与39类指标、重算平均loss均一致；未再读取图片。",
              "- 对照前后模型的所有状态张量不变；原网页冻结记录及检查点SHA256前后不变。", "",
              f"最佳模型：`{summary['best_checkpoint']}`。", "", f"SHA256：`{comparison['balanced_sha256']}`。", "",
              f"最后一轮：`{summary['last_checkpoint']}`。", "",
              "结果文件：`metrics.csv`、`training.log`、`training_summary.json`、`training_config.json`、`training_curves.png`、"
              "`best_validation_prediction.json`、`validation_comparison/comparison.json`、`validation_comparison/predictions.csv`、"
              "`validation_comparison/per_class_comparison.csv`、`validation_comparison/records_audit.json`。", "",
              "## 必须保留的局限与后续边界", "",
              "本划分已排除已知SHA256字节重复，但未排除增强同源关系，测试成绩可能偏乐观，不能直接代表独立叶片或真实拍摄场景的泛化能力。", "",
              "原验证集已用于两次选模，本对照不是新的独立最终评价。原固定测试集已经查看，不能通过反复在它上面比较来选择新模型。"
              "网页保留原已冻结模型及其原实验成绩，新模型只作课程均衡对照，不以验证结果自动切换。", "",
              "本阶段完成训练层面的均衡对照；图像严重程度仍没有可靠标注与模型，网页风险仍为未田间验证的类别规则参考。"
              "进一步正式评价需要独立采集、可追溯来源和专业标注，而不是继续增加本数据集训练轮数。", "",
              "## 对初学者的说明", "",
              "类别均衡：让训练中每类拥有相同数量的图片。本次从图片多的类别抽取一部分，称下采样；它减少数量，不能创造新病害特征。", "",
              "验证对照：在同一批验证图片上比较，便于观察数值变化；它已经参与选模，所以只能当开发期证据。"
              "已冻结的网页模型和本实验模型是两个对象，文件和成绩都分开保存。", "",
              "## 复现与查看", "", "实际命令见同目录 `commands.ps1`；重新训练时必须改用不存在的新输出目录，脚本拒绝覆盖旧实验。",
              "读取本报告不会训练、推理或重算最终测试。", ""]
    with (run / "COMPARISON_REPORT.md").open("x", encoding="utf-8") as file:
        file.write("\n".join(lines))
    completion = {"status": "completed", "purpose": summary["purpose"], "run_dir": str(run),
                  "best_epoch": summary["best_epoch"], "epochs_completed": len(history),
                  "best_checkpoint": summary["best_checkpoint"], "best_checkpoint_sha256": comparison["balanced_sha256"],
                  "last_checkpoint": summary["last_checkpoint"], "last_checkpoint_sha256": file_sha256(summary["last_checkpoint"]),
                  "report": str(run / "COMPARISON_REPORT.md"), "independent_reload_passed": True,
                  "validation_records_audited": True, "baseline_accuracy": old["accuracy"], "balanced_accuracy": new["accuracy"],
                  "baseline_macro_f1": old["macro_f1"], "balanced_macro_f1": new["macro_f1"],
                  "web_model_replaced": False, "test_evaluated": False, "test_images_opened": 0,
                  "frozen_web_record_unchanged": True, "frozen_web_checkpoint_unchanged": True,
                  "finished_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    with (run / "completion.json").open("x", encoding="utf-8") as file:
        json.dump(completion, file, ensure_ascii=False, indent=2)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
