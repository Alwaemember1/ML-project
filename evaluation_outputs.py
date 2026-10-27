"""将固定测试预测记录转成可复核的指标、热图和固定种子案例图。"""

import csv
import math
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def heatmap(matrix, names, output, normalized=False):
    cell, left, top = 23, 80, 110
    grid = cell * len(names)
    image = Image.new("RGB", (1840, 1110), "white")
    draw = ImageDraw.Draw(image)
    title = "Confusion matrix: row-normalized (%)" if normalized else "Confusion matrix: raw counts"
    draw.text((left, 22), title, fill="black", font=font(26))
    draw.text((left, 62), "Rows = TRUE class; columns = PREDICTED class. Class ID legend on the right.", fill="black", font=font(18))
    maximum = 1 if normalized else max(max(row) for row in matrix)
    for true_idx, row in enumerate(matrix):
        for pred_idx, count in enumerate(row):
            value = count / sum(row) if normalized and sum(row) else (0 if normalized else count)
            strength = math.sqrt(value / maximum) if maximum else 0
            color = tuple(round(a + (b - a) * strength) for a, b in zip((247, 251, 255), (8, 48, 107)))
            x, y = left + pred_idx * cell, top + true_idx * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=color, outline="#eeeeee")
            if count:
                label = f"{value * 100:.0f}" if normalized else str(count)
                bbox = draw.textbbox((0, 0), label, font=font(10))
                draw.text((x + (cell - bbox[2]) / 2, y + 5), label,
                          fill="white" if strength > 0.6 else "black", font=font(10))
    for idx in range(len(names)):
        draw.text((left - 26, top + idx * cell + 4), str(idx), fill="black", font=font(12))
        draw.text((left + idx * cell + 3, top - 22), str(idx), fill="black", font=font(12))
        draw.text((1010, top + idx * cell), f"{idx:02d}  {names[idx]}", fill="black", font=font(16))
    draw.text((left, top + grid + 18), "Cell annotations are rounded for display; CSV retains full precision.", fill="#444444", font=font(17))
    draw.text((left, top + grid + 48), f"Color scale: 0 to {'100%' if normalized else maximum}. Test report only; not for further tuning.", fill="#444444", font=font(17))
    if output.exists():
        raise FileExistsError(output)
    image.save(output)
    image.close()


def sample_cases(records, data_root, output):
    selections = []
    for kind, condition in (("errors", False), ("correct", True)):
        pool = sorted((row for row in records if (row["true_idx"] == row["pred_idx"]) == condition),
                      key=lambda row: row["relative_path"])
        chosen = random.Random(42).sample(pool, min(12, len(pool)))
        for index, row in enumerate(chosen):
            selections.append({"case_type": kind, "case_number": index + 1,
                               **{key: row[key] for key in ("relative_path", "sha256", "true_idx", "true_class", "pred_idx", "pred_class", "pred_probability")},
                               "seed": 42, "selection_rule": "sort paths then uniform sample without replacement, within errors/correct separately"})
        if not chosen:
            continue
        width, height, columns = 390, 375, 4
        sheet = Image.new("RGB", (columns * width, 80 + math.ceil(len(chosen) / columns) * height), "#f4f6f8")
        draw = ImageDraw.Draw(sheet)
        draw.text((16, 12), f"{kind.upper()} CASES | seed=42 | uniform random sample; not hand-picked", fill="black", font=font(24))
        draw.text((16, 45), "Previously held-out test images, now viewed report examples; not independent evaluation material again.", fill="#444444", font=font(16))
        for idx, row in enumerate(chosen):
            x, y = (idx % columns) * width, 80 + (idx // columns) * height
            with Image.open(data_root / row["relative_path"]) as source:
                preview = ImageOps.exif_transpose(source).convert("RGB")
                preview.thumbnail((370, 228))
                sheet.paste(preview, (x + (width - preview.width) // 2, y + 4))
                preview.close()
            true_label = "\n".join(textwrap.wrap(f"T[{row['true_idx']}]: {row['true_class']}", width=44))
            pred_label = "\n".join(textwrap.wrap(f"P[{row['pred_idx']}]: {row['pred_class']}", width=44))
            draw.multiline_text((x + 8, y + 238), true_label, fill="#172a3a", font=font(14), spacing=2)
            draw.multiline_text((x + 8, y + 277), pred_label, fill="#b12723" if kind == "errors" else "#1d6d3c", font=font(14), spacing=2)
            draw.text((x + 8, y + 319), f"P(predicted)={row['pred_probability']:.4f} | case {idx + 1}", fill="black", font=font(14))
            draw.text((x + 8, y + 342), Path(row["relative_path"]).name, fill="#444444", font=font(14))
        path = output / f"{kind}_seed42.jpg"
        if path.exists():
            raise FileExistsError(path)
        sheet.save(path, quality=94)
        sheet.close()
    fields = list(selections[0]) if selections else ["case_type", "relative_path"]
    write_csv(output / "report_case_sources.csv", selections, fields)
    return selections


def make_report(output, metrics, identity, training, low_recall, pairs, selections, limitation):
    pct = lambda value: f"{value * 100:.2f}%"
    best = next(row for row in training["history"] if row["epoch"] == identity["epoch"])
    lines = ["# 最终测试与错误分析（2026-10-17）", "", "> " + limitation, "",
             "## 课程报告可直接引用的摘要", "",
             f"采用现有 39 类植物叶片数据集，按固定 SHA256 代表文件划分，训练 42,890 张、验证 9,192 张、测试 9,192 张。模型为 ImageNet 预训练 ResNet18，冻结主干，仅训练 39 类分类头；依据验证 loss 选定第 3 轮，模型方案确定后对保留测试集进行本次最终评价。",
             f"最终测试共 {metrics['total']} 张，正确 {metrics['correct']} 张，错误 {metrics['errors']} 张；accuracy={pct(metrics['accuracy'])}，macro precision={pct(metrics['macro_precision'])}，macro recall={pct(metrics['macro_recall'])}，macro F1={pct(metrics['macro_f1'])}，平均交叉熵 loss={metrics['loss']:.6f}。本轮未重新训练、微调、调整阈值或预处理，测试完成后模型保持锁定。", "",
             "| 阶段 | loss | accuracy | 说明 |", "| --- | ---: | ---: | --- |",
             f"| 第 3 轮训练过程 | {best['train_loss']:.6f} | {pct(best['train_accuracy'])} | 在线训练指标，含随机增强，不是事后重新评价训练集 |",
             f"| 第 3 轮验证（选模） | {best['validation_loss']:.6f} | {pct(best['validation_accuracy'])} | 93.30% 是验证成绩 |",
             f"| 本次最终保留测试 | {metrics['loss']:.6f} | {pct(metrics['accuracy'])} | {metrics['correct']}/{metrics['total']}，未参与本次选模 |", "",
             "## 评价对象与完整性", "", f"最终模型：`{identity['checkpoint']}`", "",
             f"模型文件 SHA256：`{identity['checkpoint_sha256']}`", "", f"划分标识：`{identity['split_id']}`", "",
             f"实际命令：\n\n```powershell\n{identity['command']}\n```", "",
             f"使用 `eval()` 和 `torch.inference_mode()`，批次 32、num_workers=0，预处理直接来自检查点。逐张测试输入与已有 SHA256 核验通过。全部模型状态张量和模型文件 SHA256 在测试前后相同。测试推理主循环耗时 {metrics['inference_seconds']:.3f} 秒（包含数据读取、输入哈希核对、预处理、GPU 计算和逐图记录写入，不含后续绘图）。", "",
             "## 指标定义", "",
             "- accuracy = 全部正确图片数 / 全部测试图片数，样本多的类别贡献更大。",
             "- 每类 precision = TP/(TP+FP)：预测为该类的图片中有多少是真的。",
             "- 每类 recall = TP/(TP+FN)：真实属于该类的图片中找回了多少。support 为该真实类别样本数。",
             "- 每类 F1 = 2TP/(2TP+FP+FN)；macro 指标分别对固定全部 39 类的 precision、recall、F1 做等权平均。macro F1 不是先平均 precision/recall 再算 F1。",
             "- 分母为 0 的指标记 0，并保留该类参与宏平均；没有预测样本的类别也不删除。",
             "- loss 是逐图 CrossEntropyLoss 的样本平均。Top3 概率来自未经额外校准的 Softmax，不是经过验证的疾病发生概率。", "",
             "无预测样本类别：" + ("、".join(row["class_name"] for row in metrics["classes"] if row["no_predicted_samples"]) or "无"), "",
             "## 全部 39 类指标", "", "| 编号 | 类别 | precision | recall | F1 | support | 预测数 |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    lines.extend(f"| {row['class_idx']} | {row['class_name']} | {pct(row['precision'])} | {pct(row['recall'])} | {pct(row['f1'])} | {row['support']} | {row['predicted_count']} |" for row in metrics["classes"])
    lines += ["", "## recall 最低的类别（不隐藏低分）", "", "| 类别 | recall | support | 漏识别数 |", "| --- | ---: | ---: | ---: |"]
    lines.extend(f"| {row['class_name']} | {pct(row['recall'])} | {row['support']} | {row['fn']} |" for row in low_recall)
    for title, ranked in (("错分数量最多的真实→预测类别对", pairs[:10]),
                          ("按真实类别内错分比例排序", sorted(pairs, key=lambda row: (-row["fraction_of_true_class"], -row["count"], row["true_idx"], row["pred_idx"]))[:10])):
        lines += ["", "## " + title, "", "| 真实类别 | 预测类别 | 错分数 | 真实类support | 占该真实类 |", "| --- | --- | ---: | ---: | ---: |"]
        lines.extend(f"| {row['true_class']} | {row['pred_class']} | {row['count']} | {row['true_support']} | {pct(row['fraction_of_true_class'])} |" for row in ranked)
    lines += ["", "这些是有方向的错分对。数量榜会受大类别影响，因此同时报告真实类内比例，并提供比例榜；比例榜也保留数量和 support，避免忽视样本数。", "",
              "## 混淆矩阵怎么看", "", "行是真实类别，列是预测类别；对角线为正确，非对角线为误判。按行归一化后每一行合计 100%，可比较不同规模类别的召回情况；对角线百分比就是该类 recall。轴上编号与右侧完整类别图例一一对应，图中文字为显示而取整，CSV 保留原始计数与完整归一化数值。", "",
              "![原始计数](confusion_counts.png)", "", "![按真实类别归一化](confusion_row_normalized.png)", "",
              "## 固定种子报告案例", "", "分别对按路径排序的全部误判、全部正确样本，用独立 Random(42) 均匀无放回抽取最多 12 张；不按置信度或视觉效果挑选，样本分布不保证覆盖全部类别。图中 T 是真实类、P 是预测类，概率是预测类别的 Softmax 值。来源及抽样规则保存在 report_case_sources.csv。", ""]
    if any(row["case_type"] == "errors" for row in selections):
        lines += ["![随机误判案例](errors_seed42.jpg)", ""]
    else:
        lines += ["本次没有误判，不生成虚构的错误案例。", ""]
    if any(row["case_type"] == "correct" for row in selections):
        lines += ["![随机正确案例](correct_seed42.jpg)", ""]
    lines += ["## 观察、原因假设与限制", "",
              "已经观察到的事实是上表所列的低召回类别和有方向的混淆次数/比例；这些来自全部测试预测，不是从挑选的图片推断。",
              "外观相似、光照或背景差异、裁剪导致局部信息缺失、冻结特征能力有限，都只是可能原因，尚未通过对照实验验证。不能仅凭某张误判图片就断言标签错误；本阶段不据测试结果调整模型、阈值或预处理。",
              limitation, "",
              "除已有 SHA256 去重以外，增强来源独立性、跨真实叶片独立性、真实拍摄场景性能、域外图片识别、概率校准和实地诊断能力均未验证。本次查看的测试案例已成为报告材料，不再作为新的独立评测材料。",
              "测试集不能用于反复调参：每次依据测试分数改方案，都会把测试信息引入模型选择，成绩会越来越偏向这份测试集，失去对未见数据的独立评价作用。后续网页应复用锁定模型和同一预处理，不再用本次测试成绩选择其他模型。", "",
              "## 结果文件", "",
              "- metrics.json：总体、全部类别、耗时和冻结检查。",
              "- per_class_metrics.csv：每类 precision/recall/F1/support、TP/FP/FN 和预测数。",
              "- confusion_counts.csv / confusion_row_normalized.csv：原始计数及按真实类归一化矩阵。",
              "- predictions.csv：每张测试图的真实/预测类别、loss、Top3 编号/类别/概率。",
              "- predictions.jsonl：同一批图片的全部 39 维 logits/概率及逐图 loss，可复核指标。",
              "- audit.json：从保存的预测记录重建矩阵、指标和交叉熵的核对，不再次运行模型。",
              "- model_identity.json / evaluation_config.json：模型完整路径、SHA256、划分及实际命令。",
              "- evaluation.log / completion.json：真实过程与最终验收状态。", ""]
    with (output / "REPORT.md").open("x", encoding="utf-8") as file:
        file.write("\n".join(lines))
