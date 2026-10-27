"""复用既有清单/日志/指标生成课程报告与候选均衡权重；不解码图片、不训练。"""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from class_labels import category_kind, display_name
from plant_care import (CHECKED_DATE, REFERENCE_VERSION, RISK_SCOPE, SEVERITY_SCOPE,
                        SOURCES, care_for_class, validate_coverage)

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, text):
    with Path(path).open("x", encoding="utf-8", newline="") as file:
        file.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COURSE_REPORT.md").exists():
        raise FileExistsError("拒绝覆盖已有课程报告")
    record = read_json(ROOT / "FINAL_MODEL.json")
    training_dir = Path(record["training_run"])
    summary = read_json(training_dir / "training_summary.json")
    completion = read_json(training_dir / "completion.json")
    metrics = read_json(record["evaluation_metrics"])
    mapping = record["class_to_idx"]
    validate_coverage(mapping)
    with (Path(record["split_directory"]) / "class_counts.csv").open(encoding="utf-8-sig", newline="") as file:
        counts = list(csv.DictReader(file))
    assert len(counts) == 39
    for row in counts:
        assert mapping[row["class_name"]] == int(row["class_idx"])
    train_total = sum(int(row["train"]) for row in counts)
    validation_total = sum(int(row["validation"]) for row in counts)
    test_total = sum(int(row["test"]) for row in counts)
    assert (train_total, validation_total, test_total) == (42890, 9192, 9192)
    with Path(record["checkpoint"]).open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    assert digest == record["checkpoint_sha256"]

    # 只按训练计数导出候选每样本权重；不使用验证/测试表现确定权重。
    weights = [{"class_idx":int(row["class_idx"]), "class_name":row["class_name"],
                "train_count":int(row["train"]),
                "candidate_per_sample_weight":train_total / (39 * int(row["train"])),
                "expected_class_probability":1 / 39}
               for row in counts]
    with (out / "candidate_balancing_weights.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(weights[0]))
        writer.writeheader()
        writer.writerows(weights)
    train_min = min(int(row["train"]) for row in counts)
    train_max = max(int(row["train"]) for row in counts)
    # 另外交付真正等量的候选训练视图；只选已有正式训练清单中的路径，不碰原图。
    train_manifest = Path(record["split_directory"]) / "train.csv"
    with train_manifest.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        train_rows = list(reader)
    groups = defaultdict(list)
    for row in train_rows:
        groups[row["class_name"]].append(row)
    assert len(train_rows) == train_total and set(groups) == set(mapping)
    selected = []
    rng = random.Random(42)
    for name, _ in sorted(mapping.items(), key=lambda item:item[1]):
        selected.extend(sorted(rng.sample(sorted(groups[name], key=lambda row:row["relative_path"]), train_min),
                               key=lambda row:row["relative_path"]))
    assert len(selected) == 39 * train_min
    assert len({row["sha256"] for row in selected}) == len(selected)
    candidate_manifest = out / "candidate_balanced_train_seed42.csv"
    with candidate_manifest.open("x",encoding="utf-8-sig",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=fields)
        writer.writeheader(); writer.writerows(selected)
    with candidate_manifest.open("rb") as file:
        candidate_hash = hashlib.file_digest(file,"sha256").hexdigest()
    with train_manifest.open("rb") as file:
        source_hash = hashlib.file_digest(file,"sha256").hexdigest()
    write_new(out / "candidate_balance_lock.json",json.dumps({
        "purpose":"candidate_governance_view_not_used_for_frozen_model", "seed":42,
        "source_formal_split_id":record["split_id"],"source_train_sha256":source_hash,
        "manifest_sha256":candidate_hash,"classes":39,"per_class":train_min,"total":len(selected),
        "all_rows_from_formal_train":True,"formal_split_replaced":False,
        "validation_test_resampled":False,"trained":False},ensure_ascii=False,indent=2))
    balance = {"training_images":train_total,"classes":39,"min_train_count":train_min,
               "max_train_count":train_max,"max_min_ratio":train_max/train_min,
               "formula":"N_train / (39 * n_train_class)","seed_for_future_sampling":42,
               "applied_to_frozen_model":False,"status":"candidate_only_not_used_for_training",
               "test_images_opened":0,"sampling_probabilities_sum":sum(row["expected_class_probability"] for row in weights),
               "notes":"候选权重仅供未来另立训练实验。验证/测试不重采样，正式划分、原图及当前模型不变。"}
    balance["balanced_candidate_manifest"] = {"file":candidate_manifest.name,"total":len(selected),
                                              "per_class":train_min,"sha256":candidate_hash,"used_for_training":False}
    write_new(out / "class_balance_analysis.json", json.dumps(balance, ensure_ascii=False, indent=2))

    rows = ["| 编号 | 中文类别 | 英文类别 | 训练 | 验证 | 测试 | 参考风险 |", "| ---: | --- | --- | ---: | ---: | ---: | --- |"]
    for row in counts:
        name = row["class_name"]
        rows.append(f"| {row['class_idx']} | {display_name(name)} | `{name}` | {row['train']} | {row['validation']} | {row['test']} | {care_for_class(name)['risk_level']} |")
    history = ["| 轮次 | 训练loss | 训练accuracy | 验证loss | 验证accuracy | 训练秒 | 验证秒 |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["history"]:
        history.append(f"| {row['epoch']} | {row['train_loss']:.6f} | {row['train_accuracy']:.2%} | {row['validation_loss']:.6f} | {row['validation_accuracy']:.2%} | {row['train_seconds']:.3f} | {row['validation_seconds']:.3f} |")
    low_recall = sorted(metrics["classes"], key=lambda value:value["recall"])[:5]
    low_rows = ["| 类别 | recall | support |", "| --- | ---: | ---: |"]
    low_rows += [f"| `{row['class_name']}` | {row['recall']:.2%} | {row['support']} |" for row in low_recall]
    care_lines = []
    for name, _ in sorted(mapping.items(), key=lambda item:item[1]):
        care = care_for_class(name)
        care_lines += [f"### {display_name(name)}", "", f"原始类别：`{name}`；参考风险：{care['risk_level']}。", care["reason"],
                       f"复核：{care['signs_to_check']}", ""]
        care_lines += [f"- {text}" for text in care["actions"]]
        care_lines += [f"- 来源：[{source['title']}]({source['url']})" for source in care["sources"]]
        care_lines += [""]
    write_new(out / "CARE_REFERENCE.md", "# 39类复核与防治资料参考\n\n" + RISK_SCOPE + "\n\n" + SEVERITY_SCOPE + "\n\n" + "\n".join(care_lines))

    curve_link = os.path.relpath(training_dir / "training_curves.png", out).replace("\\", "/")
    confusion_link = os.path.relpath(Path(record["evaluation_dir"]) / "confusion_row_normalized.png", out).replace("\\", "/")
    report = f"""# 基于图像数据的农作物病虫害识别与防治决策：课程项目报告

报告日期：{CHECKED_DATE}。本报告复用已有真实结果，新增类别规则参考和可下载单图报告，不重新训练或评价测试集。

## 1. 任务、范围与实现状态

面向课程演示，采用现有39类数据构建叶片分类与植保资料展示原型。作物和病害是联合标签；另含健康与背景，不能写成39种病害。

| 任务要求 | 本项目交付 | 实际边界 |
| --- | --- | --- |
| 图像治理：清洗、去重、类别均衡、增强 | 已有全量可读性检查、SHA256代表、逐类统计、锁定划分和训练在线增强；本轮导出等量候选清单和均衡权重 | 候选视图未替换正式划分，冻结模型没有启用均衡训练；增强同源未排除 |
| 作物-病害多分类模型 | 冻结 ResNet18 / 39 类完整检查点 | 联合作物病害标签，无目标定位/病斑分割 |
| 严重程度 / 风险等级 | 实现类别规则参考风险高/中/低及背景不适用 | 采用任务中的“风险等级”路线；未做图像严重程度，不是田间风险预测模型 |
| 上传识别可视化 | Streamlit中文页面、11种后缀、Top3、可折叠可复制类别表 | 仅本机运行，图片默认内存处理 |
| 报告与针对性防治方案 | 本综合报告、39类资料附录、按单次真实预测下载Markdown报告 | 资料有来源，须复核后使用，未验证田间防效或生成精准施药处方 |

## 2. 数据治理与固定划分

已完成61486张图片可读性检查，损坏0张。既有SHA256审计识别61274个唯一字节代表，212份冗余副本保持原位。冻结划分训练{train_total:,}、验证{validation_total:,}、测试{test_total:,}，全部39类，seed=42。1170个smoke样本哈希只进入训练集，跨集合已知字节哈希与路径交集为0。

固定划分标识：`{record['split_id']}`。本轮只读取已有逐类计数，不重新解码或扫描原图。

训练集最少{train_min}张/类，最多{train_max}张/类，最大/最小比{train_max/train_min:.3f}。已导出 `candidate_balancing_weights.csv`：每个样本候选权重=N_train/(39×该类训练数)，用于未来带放回的加权采样时可使各类期望采样概率均为1/39。**这些权重没有用于当前冻结模型，没有重采样验证/测试，也没有改动正式清单。** 若课程严格要求均衡训练及对照结果，此项仍需另立实验并重新规划独立评价，不能拿本次已查看测试集反复选模。

另已实际生成 `candidate_balanced_train_seed42.csv`：固定种子42，从正式训练清单按类随机选择{train_min}张，共{len(selected):,}行；每类数量完全相同、哈希不重复，全部是正式训练集子集，验证/测试完全未改。`candidate_balance_lock.json` 记录源清单与新视图哈希。此文件是数据治理备选成果，**没有用它重新训练，也不能把旧模型成绩写成均衡训练成绩**。等量下采样会丢弃可用训练样本，因此同时提供候选权重供未来另立实验比较，未根据最终测试成绩选择方案。

原始图片增强同源关系不可可靠追溯；SHA256只能排除字节相同，旋转、翻转等同源样本可能仍跨集合。

## 3. 模型与训练

ResNet18从ImageNet缓存预训练权重开始，重新初始化39类分类头；冻结主干和BatchNorm状态。训练使用适度随机裁剪和水平翻转、CrossEntropyLoss与Adam(lr=0.001)，batch_size=32，num_workers=0，最多10轮，验证loss最小保存，patience=3。实际完成{summary['epochs_completed']}轮，最佳第{summary['best_epoch']}轮，之后早停。

{"\n".join(history)}

![真实训练曲线]({curve_link})

第3轮训练accuracy=95.75%，选模验证accuracy=93.30%；后续训练指标继续改善、验证loss未改善，存在过拟合迹象，因此保留第3轮并停止。不能因测试或外部示例效果不理想继续无上限训练。

模型完整路径：`{record['checkpoint']}`。

SHA256：`{digest}`。验证预处理：EXIF方向处理、RGB、短边256、中心裁剪224、ImageNet归一化；推理eval与inference_mode。

## 4. 既有真实最终测试（本轮未重跑）

| 指标 | 真实结果 |
| --- | ---: |
| Accuracy | {metrics['accuracy']:.2%} |
| 正确/总数 | {metrics['correct']}/{metrics['total']} |
| Macro precision | {metrics['macro_precision']:.2%} |
| Macro recall | {metrics['macro_recall']:.2%} |
| Macro F1 | {metrics['macro_f1']:.2%} |

指标来自冻结记录指向的现有 `metrics.json`；完整测试报告、原始/归一化混淆矩阵、逐图分数与错误案例均保持原位。Accuracy按图片总体计数，macro F1对各类等权平均，不代表每类同样好。

{"\n".join(low_rows)}

![既有归一化混淆矩阵]({confusion_link})

混淆矩阵行是真实类别、列是预测类别；非对角线显示错分。网上图片与训练分布不同，可能出现背景、光照、裁剪和未收录类别问题；没有核实的网上搜索标签不能直接视为真值。

## 5. 风险与防治决策路线

本项目选择“参考风险等级”，没有使用预测概率充当严重程度。高：对黄龙病、两种晚疫病及两种番茄病毒病优先复核与管理传播；中：其他已收录病害/虫害开展核查和针对性管理；低：预测健康，仍不能排除其他风险；背景：不适用。

这些等级是本项目透明、固定的演示规则，依据资料中的传播/管理特点设置，不是来源机构给出的统一官方风险评级。相同类别的风险不会因模型分数0.2或0.99而改变。当前没有严重程度标签、像素级病斑标注、地块发病率、天气或产量损失输入，因此不能报告叶片病斑百分比、全田风险或自动施药阈值。

每类给出复核点、具体管理要点及政府/大学来源；黄龙病强调健康种苗和木虱管理，花叶病毒强调接触卫生，黄化曲叶病毒强调粉虱与种苗，叶螨强调活螨核查和天敌保护，晚疫病强调尽快核查传播和寄主来源。完整资料见 [39类资料附录](CARE_REFERENCE.md)。不从国外页面直接移植商品药剂、剂量或间隔期；需要用药时由当地植保人员依据当地登记标签确定。

规则版本：`{REFERENCE_VERSION}`；来源核对日期：{CHECKED_DATE}；尚无专家规则一致性或田间风险/防效验证结果。

## 6. 系统流程与使用

浏览器上传 → 内存安全解码与确定性预处理 → 冻结模型输出39类分数 → 按原类别映射显示Top3 → 对Top1查询资料 → 显示规则参考风险 → 点击下载单次报告。类别表为真正HTML文字，可拖选并Ctrl+C复制；保持默认收起。

模型缓存加载，上传不默认落盘。损坏、伪装后缀、动画/多页、超过10MiB或2000万像素的图片会拒绝。支持CPU/GPU，不下载权重。网页与命令行调用相同预测模块。单次报告包含原始类别、Top3分数、模型哈希、规则版本、参考措施和来源，不自动保存到服务器。

启动命令在根目录README；仅监听127.0.0.1，不含登录、数据库、聊天或公网部署。

## 7. 局限及后续工作

1. 本划分已排除已知SHA256字节重复，但未排除增强同源关系，测试成绩可能偏乐观，不能直接代表独立叶片或真实拍摄场景的泛化能力。
2. 当前锁定模型没有均衡训练；本轮只生成候选权重，不能捏造均衡对照成绩。
3. 参考风险是规则展示，严重程度、田间风险和防效未专项验证；真实诊断需植保复核。
4. 39类之外仍会被分到已有类别，背景类不是通用域外检测器，高分不保证诊断可靠。
5. 若要求像素级严重程度或真正精准施药，需要额外标注与验证数据；若要求重新均衡训练，须另开模型实验，保留当前冻结对象与旧成绩。

## 附录A：全部类别、计数和参考规则

{"\n".join(rows)}

本表的参考风险不属于原图标签或模型监督标签。
"""
    write_new(out / "COURSE_REPORT.md", report)
    audit = {"status":"generated","test_images_opened":0,"source_images_decoded":0,"training_started":False,
             "checkpoint_sha256":digest,"candidate_balancing_applied":False,
             "class_count":39,"care_disease_count":26,"training_completion_source":str(training_dir / "completion.json"),
             "balanced_candidate_rows":len(selected),"balanced_candidate_per_class":train_min,
             "training_completion":completion,"metrics_source":record["evaluation_metrics"]}
    write_new(out / "report_provenance.json", json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({key:value for key,value in audit.items() if key != "training_completion"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
