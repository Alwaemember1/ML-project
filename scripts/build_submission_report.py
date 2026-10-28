"""汇总已完成课程成果，生成可编辑和可打印报告；不训练、不加载模型或原图。"""

import argparse
import base64
import csv
import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from class_labels import category_kind, display_name
from plant_care import RISK_SCOPE, SEVERITY_SCOPE, USE_SCOPE, care_for_class, validate_coverage

ROOT = Path(__file__).resolve().parents[1]
BALANCED = ROOT / "reports/balanced_head_20261018_200208_701758"
COURSE = ROOT / "reports/course_completion_20261018_193414_578564"
GOVERNANCE = ROOT / "reports/check_20261016_133031_170321"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_new(path, content):
    with Path(path).open("x", encoding="utf-8", newline="") as file:
        file.write(content)


def percent(value):
    return f"{value:.2%}"


def inline(value):
    text = html.escape(str(value))
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


CSS = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#edf0f2;color:#17212b;font-family:'Microsoft YaHei','Noto Sans CJK SC',sans-serif;line-height:1.8;font-size:15px}
main{max-width:1040px;margin:30px auto;padding:54px 64px;background:white;box-shadow:0 6px 24px #182d4010}h1,h2,h3{color:#000;line-height:1.45;break-after:avoid}h1{font-size:29px;margin:0 0 20px}h2{font-size:23px;margin:40px 0 16px}h3{font-size:18px;margin:26px 0 12px}p{margin:0 0 13px}a{color:#184e71;text-underline-offset:3px}nav{margin:24px 0 30px}nav a{display:inline-block;margin-right:18px}table{border-collapse:collapse;width:100%;margin:16px 0 24px;font-size:13px;table-layout:fixed}thead{display:table-header-group}th,td{border:1px solid #d9d9d9;padding:9px 10px;vertical-align:middle;overflow-wrap:anywhere}th{background:#21465f;color:#fff;text-align:left}tr:nth-child(even) td{background:#f5f7f9}td:not(:first-child){font-variant-numeric:tabular-nums}tr{break-inside:avoid}code{font-family:Consolas,monospace;font-size:.9em;overflow-wrap:anywhere}pre{background:#f4f5f6;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.6}figure{margin:20px 0 26px;break-inside:avoid}figure img{display:block;max-width:100%;height:auto}figcaption{font-size:12px;color:#536371;margin-top:8px}section{scroll-margin-top:24px}.meta{font-size:13px;color:#536371}.table-wrap{overflow-x:auto}.toolbar{max-width:1040px;margin:20px auto;padding:0 12px;font-size:13px}li{margin-bottom:7px}
@media(max-width:700px){main{margin:0;padding:28px 20px}h1{font-size:24px}h2{font-size:21px}table{font-size:12px}th,td{padding:7px 5px}nav a{display:block}body{font-size:14px}}
@page{size:A4;margin:18mm 16mm 18mm}
@media print{body{background:white;font-size:10pt;line-height:1.65}main{max-width:none;margin:0;padding:0;box-shadow:none}.toolbar,nav{display:none}h1{font-size:21pt}h2{font-size:16pt;margin-top:22pt}h3{font-size:12pt}table{font-size:8pt;margin:10pt 0 16pt}th,td{padding:5pt 6pt}th{background:#e8edf1;color:black}figure img{max-height:210mm;object-fit:contain}pre{font-size:8pt}a{color:inherit}section{scroll-margin-top:0}.table-wrap{overflow:visible}}
"""


def render(blocks, out, title):
    markdown, web = [f"# {title}", ""], [f"<h1>{html.escape(title)}</h1>"]
    for block in blocks:
        kind = block[0]
        if kind == "heading":
            _, level, ident, text = block
            markdown += [f"{'#' * level} {text}", ""]
            web.append(f'<h{level} id="{ident}">{html.escape(text)}</h{level}>')
        elif kind == "paragraph":
            markdown += [block[1], ""]
            web.append(f"<p>{inline(block[1])}</p>")
        elif kind == "list":
            markdown += [*(f"- {item}" for item in block[1]), ""]
            web.append("<ul>" + "".join(f"<li>{inline(item)}</li>" for item in block[1]) + "</ul>")
        elif kind == "table":
            _, headers, rows = block
            markdown += ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
            markdown += ["| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows]
            markdown += [""]
            widths = ({7: [5, 20, 35, 10, 10, 10, 10], 6: [6, 34, 15, 15, 15, 15],
                       5: [31, 15, 14, 14, 26], 3: [28, 36, 36]}.get(len(headers))
                      or ([34, 34, 12, 20] if headers[1] == "误判类别" else [48, 16, 18, 18]))
            columns = '<colgroup>' + ''.join(f'<col style="width:{width}%">' for width in widths) + '</colgroup>'
            web.append('<div class="table-wrap"><table>' + columns + '<thead><tr>' + "".join(f'<th scope="col">{inline(cell)}</th>' for cell in headers) + "</tr></thead><tbody>" +
                       "".join("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>" for row in rows) + "</tbody></table></div>")
        elif kind == "image":
            _, relative, caption = block
            image = out / relative
            mime = "image/png" if image.suffix == ".png" else "image/jpeg"
            data = base64.b64encode(image.read_bytes()).decode("ascii")
            markdown += [f"![{caption}]({relative})", "", caption, ""]
            web.append(f'<figure><img alt="{html.escape(caption, quote=True)}" src="data:{mime};base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        elif kind == "code":
            markdown += [f"```{block[1]}", block[2], "```", ""]
            web.append(f"<pre><code>{html.escape(block[2])}</code></pre>")
        elif kind == "links":
            markdown += [*(f"- [{text}]({url})" for text, url in block[1]), ""]
            web.append("<ul>" + "".join(f'<li><a href="{html.escape(url, quote=True)}">{html.escape(text)}</a></li>' for text, url in block[1]) + "</ul>")
        else:
            raise ValueError(f"未知报告块：{kind}")
    navigation = "".join(f'<a href="#{block[2]}">{html.escape(block[3])}</a>' for block in blocks if block[0] == "heading" and block[1] == 2)
    document = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body>'
                '<div class="toolbar">本地报告 · 浏览器 Ctrl+P 可打印或另存为PDF · 图片已嵌入HTML，证据链接需保留同目录文件</div>'
                f'<main><nav aria-label="章节导航">{navigation}</nav>{"".join(web)}</main></body></html>')
    write_new(out / "COURSE_REPORT_FINAL.md", "\n".join(markdown))
    write_new(out / "COURSE_REPORT_FINAL.html", document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "reports"):
        raise ValueError("交付目录必须位于项目reports内")
    out.mkdir(parents=True, exist_ok=False)
    (out / "assets").mkdir()
    (out / "evidence").mkdir()
    sources = {}

    def retain(source, relative):
        source = Path(source)
        data = source.read_bytes()
        with (out / relative).open("xb") as file:
            file.write(data)
        sources[str(source.resolve())] = {"sha256": hashlib.sha256(data).hexdigest(), "copy": relative}

    record = read_json(ROOT / "FINAL_MODEL.json")
    record_before = digest(ROOT / "FINAL_MODEL.json")
    if digest(record["checkpoint"]) != record["checkpoint_sha256"]:
        raise ValueError("原冻结模型哈希不符")
    baseline_dir = Path(record["training_run"])
    test_dir = Path(record["evaluation_dir"])
    baseline = read_json(baseline_dir / "training_summary.json")
    test = read_json(record["evaluation_metrics"])
    test_completion = read_json(test_dir / "completion.json")
    balanced = read_json(BALANCED / "training_summary.json")
    comparison = read_json(BALANCED / "validation_comparison/comparison.json")
    balanced_done = read_json(BALANCED / "completion.json")
    if balanced_done["status"] != "completed" or balanced_done["test_evaluated"] or balanced_done["web_model_replaced"]:
        raise ValueError("均衡实验的完成记录或评价范围不符")
    if not test_completion["test_evaluated"] or test_completion["checkpoint_sha256"] != record["checkpoint_sha256"]:
        raise ValueError("原最终测试归属不符")
    if comparison["baseline_sha256"] != record["checkpoint_sha256"] or digest(balanced_done["best_checkpoint"]) != balanced_done["best_checkpoint_sha256"]:
        raise ValueError("对照模型身份不符")
    if len({record["split_id"], baseline["training_config"]["split_id"], balanced["training_config"]["split_id"], comparison["split_id"]}) != 1:
        raise ValueError("实验划分标识不一致")
    mapping = record["class_to_idx"]
    validate_coverage(mapping)
    ordered = sorted(mapping, key=mapping.get)
    counts = read_csv(Path(record["split_directory"]) / "class_counts.csv")
    assert len(ordered) == len(counts) == 39
    assert [row["class_name"] for row in counts] == ordered
    assert (sum(int(row['train']) for row in counts), sum(int(row['validation']) for row in counts), sum(int(row['test']) for row in counts)) == (42890, 9192, 9192)
    assert test["total"] == comparison["samples"] == 9192
    assert [row["class_name"] for row in test["classes"]] == ordered
    best_row = baseline["history"][record["epoch"] - 1]
    assert best_row["validation_accuracy"] == comparison["metrics"]["baseline"]["accuracy"]
    governance = read_json(GOVERNANCE / "summary.json")
    old_web = read_json(COURSE / "verification.json")
    error = read_json(test_dir / "error_analysis.json")
    pairs = read_csv(test_dir / "confusion_pairs.csv")
    original_curve = baseline_dir / "training_curves_reviewed.png"
    retained = [
        (ROOT / "FINAL_MODEL.json", "evidence/FINAL_MODEL.json"),
        (baseline_dir / "training_summary.json", "evidence/baseline_training_summary.json"),
        (baseline_dir / "metrics.csv", "evidence/baseline_training_metrics.csv"),
        (test_dir / "metrics.json", "evidence/original_final_test_metrics.json"),
        (test_dir / "completion.json", "evidence/original_final_test_completion.json"),
        (test_dir / "confusion_pairs.csv", "evidence/original_test_confusion_pairs.csv"),
        (test_dir / "confusion_counts.csv", "evidence/original_test_confusion_counts.csv"),
        (test_dir / "confusion_row_normalized.csv", "evidence/original_test_confusion_row_normalized.csv"),
        (test_dir / "per_class_metrics.csv", "evidence/original_test_per_class_metrics.csv"),
        (test_dir / "report_case_sources.csv", "evidence/viewed_test_case_sources.csv"),
        (Path(record["split_directory"]) / "class_counts.csv", "evidence/formal_class_counts.csv"),
        (BALANCED / "completion.json", "evidence/balanced_completion.json"),
        (BALANCED / "validation_comparison/comparison.json", "evidence/validation_comparison.json"),
        (BALANCED / "validation_comparison/per_class_comparison.csv", "evidence/validation_per_class_comparison.csv"),
        (BALANCED / "training_config.json", "evidence/balanced_training_config.json"),
        (BALANCED / "metrics.csv", "evidence/balanced_training_metrics.csv"),
        (BALANCED / "validation_comparison/records_audit.json", "evidence/validation_records_audit.json"),
        (GOVERNANCE / "summary.json", "evidence/image_readability_summary.json"),
        (COURSE / "verification.json", "evidence/web_verification_20261018.json"),
        (COURSE / "report_v2/CARE_REFERENCE.md", "CARE_REFERENCE.md"),
        (original_curve, "assets/original_training_curves.png"),
        (BALANCED / "training_curves.png", "assets/balanced_training_curves.png"),
        (COURSE / "prediction_with_care.png", "assets/web_demo_20261018.png"),
        (test_dir / "confusion_row_normalized.png", "assets/original_test_confusion.png"),
        (test_dir / "errors_seed42.jpg", "assets/viewed_test_errors_seed42.jpg"),
        (test_dir / "correct_seed42.jpg", "assets/viewed_test_correct_seed42.jpg"),
    ]
    for source, relative in retained:
        retain(source, relative)
    for name in ("README.md", "FINAL_MODEL.json"):
        retain(ROOT / name, f"evidence/before_delivery_{name}")
    retain(Path(__file__), "evidence/build_submission_report.py")
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    blocks = []
    p = lambda text: blocks.append(("paragraph", text))
    h = lambda level, ident, text: blocks.append(("heading", level, ident, text))
    table = lambda headers, rows: blocks.append(("table", headers, rows))
    image = lambda path, caption: blocks.append(("image", path, caption))
    links = lambda rows: blocks.append(("links", rows))
    p("课程项目3　基于图像数据的农作物病虫害识别与防治决策")
    p(f"整理日期 {timestamp[:10]}。作者姓名、学号、班级与指导教师请按真实信息补充，本文不代填身份信息。")
    h(2, "abstract", "摘要与成果范围")
    p(f"项目使用现有39类植物叶片数据，完成图片可读性检查、SHA256字节去重、固定划分、ResNet18迁移学习、最终测试与中文上传识别系统。原冻结模型在9192张保留测试图片上正确识别{test['correct']}张，accuracy为{percent(test['accuracy'])}，macro F1为{percent(test['macro_f1'])}。网页依据预测类别提供透明的参考风险与防治资料。")
    p(f"独立均衡下采样实验从原训练集按39类各582张构建22698张视图，在相同验证集上获得accuracy {percent(comparison['metrics']['balanced']['accuracy'])}、macro F1 {percent(comparison['metrics']['balanced']['macro_f1'])}。这是验证对照，未形成新模型最终测试成绩，也未替换网页冻结模型。")
    p(record["limitation"])
    h(2, "requirements", "一 课程要求与完成情况")
    table(["任务", "实际交付", "验收范围"], [
        ["图像数据治理", "清洗检查、字节去重、固定划分、在线增强与均衡训练对照", "未排除增强同源关系"],
        ["作物病害多分类", "ResNet18输出39类，包括病害虫害、健康与背景", "保留原类别映射"],
        ["严重程度或风险等级", "采用类别规则参考风险，高、中、低及不适用", "严重程度未评估，风险未田间验证"],
        ["上传识别可视化", "中文Streamlit页面、Top3、可复制类别表、报告下载", "本机演示，不公网部署"],
        ["报告与针对性防治方案", "实验报告、39类来源附录、识别后的分类别复核与管理资料", "不是精准施药处方或防效证明"],
    ])
    h(2, "data", "二 数据治理与固定划分")
    p(f"原始数据为Plant_leave_diseases_dataset_with_augmentation，按类别文件夹组织。已完成的可读性报告记录{governance['total_images']:,}张、{governance['class_count']}类，损坏{governance['corrupted_images']}张。本次整理仅复用该报告，没有重新全量解码图片。39类包括26类病害或虫害、12类健康与1类无叶片背景，不等于39种病害。")
    p("按已有SHA256结果选择稳定代表后，61274张参与正式实验，212个字节重复副本保留原位。SHA256只能发现字节完全相同的文件，不能排除旋转、翻转、裁剪等增强同源图片。来源关系无法可靠追溯，课程要求继续使用现有数据，因此保留这一评价局限。")
    table(["集合", "图片数", "用途"], [["训练", "42890", "参数更新；smoke使用过的已知哈希仅在此集合"], ["验证", "9192", "选择轮次及均衡实验开发期比较"], ["测试", "9192", "原方案确定后已完成一次最终评价；现已查看"], ["均衡训练视图", "22698", "原训练集子集，每类582张；未替换正式划分"]])
    p(f"固定划分目录为splits/formal_sha256_seed42_v1，seed42，划分ID为`{record['split_id']}`。已有核验记录确认跨集合路径及SHA256交集均为0。均衡视图同样与验证集无已知字节重复；它不改变验证、测试清单。")
    p("原始训练类别数为582至3855张，存在数量差异。均衡下采样减少多数类样本，不生成新的独立叶片。原训练和均衡训练均使用适度随机裁剪、水平翻转作为在线增强；原始图片保持只读，变换只作用于内存。")
    h(2, "model", "三 模型与训练方法")
    p("模型使用ImageNet预训练ResNet18提取图像特征，将最后分类头改为39维输出。分类头可理解为根据特征给每个类别打分的最后一层。冻结主干只训练分类头，并让主干及BatchNorm保持eval状态，避免参数及运行统计改变。")
    p("CrossEntropyLoss比较模型输出与真实类别，反向传播计算分类头梯度，Adam据此更新参数。前向测试通过只说明能输出结果；实际训练还需验证loss、梯度、参数更新、验证流程及检查点重载。本项目已完成这些闭环检查。")
    table(["项目", "原冻结模型", "均衡对照模型"], [["训练图片", "42890", "22698"], ["初始化", "ImageNet缓存及新39类分类头", "相同缓存及新分类头，不续训原模型"], ["优化器与学习率", "Adam，0.001", "Adam，0.001"], ["批次与读取进程", "32，num_workers0", "32，num_workers0"], ["训练上限与早停", "最多10轮，patience3；实际6轮", "相同规则；实际10轮"], ["选模依据", "验证loss最低，第3轮", "验证loss最低，第10轮"], ["每轮优化器更新次数", "1341", "710"]])
    p("推理及验证严格复用确定性预处理：EXIF方向修正、RGB、短边256、中心裁剪224×224、双线性插值，按ImageNet均值[0.485,0.456,0.406]和标准差[0.229,0.224,0.225]归一化。训练额外使用RandomResizedCrop，scale为0.8至1.0、ratio为0.9至1.1，水平翻转概率0.5。")
    p("实际训练环境为Python 3.14.6、torch 2.13.0+cu130、torchvision 0.28.0+cu130，GPU为RTX 4060 Laptop。本轮报告整理没有安装依赖、加载模型、训练或推理。")
    image("assets/original_training_curves.png", "图1 原模型真实训练与验证曲线 以第3轮验证loss选择冻结模型")
    h(2, "results", "四 训练 验证与最终测试结果")
    table(["对象与阶段", "Accuracy", "Macro F1", "Loss", "解释"], [
        ["原模型第3轮训练过程", percent(best_row['train_accuracy']), "未单独统计", f"{best_row['train_loss']:.6f}", "含在线随机增强"],
        ["原模型第3轮验证", percent(best_row['validation_accuracy']), percent(comparison['metrics']['baseline']['macro_f1']), f"{best_row['validation_loss']:.6f}", "用于选模"],
        ["原冻结模型最终测试", percent(test['accuracy']), percent(test['macro_f1']), f"{test['loss']:.6f}", f"{test['correct']}/{test['total']}，原测试记录"],
        ["均衡模型第10轮验证", percent(comparison['metrics']['balanced']['accuracy']), percent(comparison['metrics']['balanced']['macro_f1']), f"{comparison['metrics']['balanced']['loss']:.6f}", "开发期验证，不是最终测试"],
    ])
    p("Accuracy是所有图片中预测正确的比例，样本多的类别影响更大。Macro F1先在每个类别计算F1，再让39类等权平均，有助于关注小类别。每类precision=TP/(TP+FP)，recall=TP/(TP+FN)，F1=2TP/(2TP+FP+FN)；分母为0时记0，保留该类，不删除低分。")
    p(f"原最终测试共有{test['errors']}张错分，macro precision为{percent(test['macro_precision'])}、macro recall为{percent(test['macro_recall'])}。这些数值只归属原第3轮模型，不转用于均衡模型。原模型测试之后未依据测试结果调整参数或阈值。")
    h(3, "errors", "原模型最终测试的薄弱类别")
    table(["真实类别", "Support", "Recall", "漏识别数"], [[display_name(row['class_name']), row['support'], percent(row['recall']), row['fn']] for row in error['lowest_recall'][:5]])
    p("例如马铃薯晚疫病recall仅54.67%，150张中漏识别68张；整体准确率不能掩盖这一问题。以下类别对同时列出错分数和占真实类比例，避免仅由大类别主导结论。")
    table(["真实类别", "误判类别", "数量", "占该真实类别"], [[display_name(row['true_class']), display_name(row['pred_class']), row['count'], percent(float(row['fraction_of_true_class']))] for row in pairs[:5]])
    p("混淆矩阵按行放真实类别、按列放预测类别；对角线为正确识别，非对角线为错分。按真实类归一化后，每行总和为1，便于比较不同样本量类别的漏识别比例。已有混淆计数及归一化文件随报告保留，未重新运行测试。")
    links([("查看原测试归一化混淆矩阵高清图", "assets/original_test_confusion.png"), ("原始混淆计数CSV", "evidence/original_test_confusion_counts.csv"), ("逐真实类归一化CSV", "evidence/original_test_confusion_row_normalized.csv"), ("已有seed42误判拼图", "assets/viewed_test_errors_seed42.jpg"), ("已有seed42正确拼图", "assets/viewed_test_correct_seed42.jpg"), ("已有案例来源与标记", "evidence/viewed_test_case_sources.csv")])
    p("案例来自先前测试阶段按seed42抽取的已查看报告材料，本次没有重新挑图。观察到的是类别间错分；症状相似、背景或尺度差异只是可能解释，不能凭单图断言标签错误。上述图片已用于报告，不再视为独立评价材料。")
    h(2, "balance", "五 类别均衡训练对照")
    old, new = comparison['metrics']['baseline'], comparison['metrics']['balanced']
    p(f"均衡实验实际完成10轮，最佳第10轮，训练及验证墙钟{balanced['training_validation_wall_seconds_excluding_gate']:.2f}秒，首轮核对等待另计。在同一9192张验证集上，accuracy相对原模型提高{(new['accuracy']-old['accuracy'])*100:.2f}个百分点，macro F1提高{(new['macro_f1']-old['macro_f1'])*100:.2f}个百分点。")
    table(["指标", "原模型验证", "均衡模型验证"], [[key, f"{old[key]:.6f}" if key == 'loss' else percent(old[key]), f"{new[key]:.6f}" if key == 'loss' else percent(new[key])] for key in ('accuracy','macro_precision','macro_recall','macro_f1','loss')])
    image("assets/balanced_training_curves.png", "图2 均衡实验真实曲线 第8和9轮回落后第10轮改善 达到预定上限停止")
    p("均衡实验逐轮确认主干120个状态张量及60个BatchNorm缓冲固定、分类头更新，独立进程重载时同一输入的logits和概率差为0。保存的9192条验证预测记录已重新构造指标核对。")
    p("均衡不代表所有类别都改善。均衡模型验证中，番茄二斑叶螨危害recall为58.96%，原模型同一验证集为59.76%；番茄早疫病recall由83.33%降到78.00%。这些均是验证结果，不与原测试的同类指标混用。")
    p("下采样同时减少图片与每轮更新次数，实际训练轮次也不同，因此不能把差异全部归因于均衡。只有一个种子的一组实验，未证明统计显著性。原验证集重复用于选模，这一对照没有新的独立最终成绩。")
    h(2, "decision", "六 参考风险与针对性防治资料")
    p("课程任务允许严重程度或风险等级，本项目采用类别规则参考风险路线。预测Top1类别用于查询资料，分类概率不参与高、中、低分级。高风险参考包含柑橘黄龙病、马铃薯晚疫病、番茄晚疫病及两类番茄病毒病；其余病害虫害为中、健康为低、背景为不适用。这是项目展示规则。")
    p(RISK_SCOPE)
    p(SEVERITY_SCOPE)
    p(USE_SCOPE)
    for raw in ('Apple___Apple_scab', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Potato___Late_blight'):
        care = care_for_class(raw)
        h(3, f"care-{mapping[raw]}", display_name(raw))
        p(f"参考风险 {care['risk_level']}。复核要点：{care['signs_to_check']}")
        blocks.append(("list", care['actions']))
        links([(source['title'] + ' 原资料核对日期' + source['checked_date'], source['url']) for source in care['sources']])
    links([("全部39类参考风险 复核要点 管理措施和原文来源", "CARE_REFERENCE.md")])
    p("本报告沿用2026年10月18日已经核对的政府和大学推广资料，没有将本次整理日期当成新的来源核对日期。背景类不提供病害处置判断；健康预测不能排除未收录病害。")
    h(2, "system", "七 中文可视化系统与演示")
    p("系统过程为：浏览器页面上传图片 → app.py接收内存字节 → inference.py核对冻结记录和模型SHA256、解码及确定性预处理 → ResNet18在eval与inference_mode下输出39类分数 → 页面显示Top1、Top3及对应参考资料。模型加载被缓存，页面操作不反复下载权重。")
    p("上传支持JPG、JPEG、JPE、JFIF、PNG、WebP、BMP、TIF、TIFF、静态GIF和AVIF，共11个后缀；HEIC、RAW、动画和多页图片不支持。限制10MiB及2000万像素，同时检查后缀和真实内容。图片只在内存处理，换图或移除时清除旧结果，不设置未经验证的拒识阈值。")
    image("assets/web_demo_20261018.png", "图3 2026年10月18日实际浏览器验收截图 使用既定验证图 不是本阶段重新点击验收")
    p("既有AppTest和真实Chrome验收覆盖上传、预览、连续预测、换图及移除清理、类别折叠与文字选择、资料展示和报告下载；CPU单图推理另有先前记录。文字可以选中，但系统剪贴板未读取、人工粘贴未宣称已验收。本阶段仅复用证据，不把旧服务PID或旧截图当成今天仍在运行的证明。")
    blocks.append(("code", "powershell", r"""Set-Location 'D:\PythonProject3'
$env:TEMP = 'D:\PythonProject3\.venv\smoke_tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:XDG_CACHE_HOME = $env:TEMP
& '.\.venv\Scripts\python.exe' -m streamlit run .\app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false"""))
    p("访问http://127.0.0.1:8501。前台终端按Ctrl+C停止。若端口占用，先确认是否已有本项目服务；旧记录中的PID可能失效，不要直接按过期PID停止进程。逐步演示流程和现场验收项见同目录DEMO_AND_DEFENSE.md。")
    h(2, "limits", "八 局限与交付结论")
    blocks.append(("list", [record['limitation'], "风险模块是有来源的类别规则原型，没有田间调查或防效数据验证，不具备自动施药决策能力。", "分类模型不能计算病斑面积或真实严重程度；需要专业标注及独立验证，不能以Softmax概率冒充严重程度。", "39类之外的输入仍会落入现有类别；背景类不等于通用非叶片或域外检测。真实拍摄、复杂背景及跨设备表现尚未验证。", "原测试集已经查看，不能反复用于调参；均衡模型仅验证，不替代原网页模型，不借用原测试成绩。", "本阶段不训练、不重新测试、不替换模型、不发布网站。后续补验证需要来源可追溯的独立材料和专业复核。 "]))
    p("课程原型已覆盖数据治理、多分类、类别参考风险、上传可视化和防治参考报告。正式田间应用的证据仍不足。系统适合用于说明完整实验流程、复核模型输出与学习评价方法。")
    h(2, "appendix-data", "附录一 全部类别与固定样本数")
    table(["编号", "中文名称", "原英文类别", "训练", "验证", "测试", "均衡训练"], [[row['class_idx'], display_name(row['class_name']), row['class_name'], row['train'], row['validation'], row['test'], '582'] for row in counts])
    h(2, "appendix-test", "附录二 原冻结模型全部测试类别指标")
    p("下表只属于原第3轮模型9192张测试记录；没有筛掉低分或无预测类别。完整英文映射见附录一。")
    table(["编号", "中文名称", "Precision", "Recall", "F1", "Support"], [[row['class_idx'], display_name(row['class_name']), percent(row['precision']), percent(row['recall']), percent(row['f1']), row['support']] for row in test['classes']])
    h(2, "identity", "附录三 模型身份与证据索引")
    p(f"网页模型路径：`{record['checkpoint']}`")
    p(f"网页模型SHA256：`{record['checkpoint_sha256']}`")
    p(f"均衡实验模型路径：`{balanced_done['best_checkpoint']}`")
    p(f"均衡实验模型SHA256：`{balanced_done['best_checkpoint_sha256']}`")
    p("HTML内嵌正文图片，可以单独阅读与打印；证据链接和附录资料需要保持整个交付目录结构。sources.json记录本次引用及复制文件的原路径与SHA256，delivery_manifest.json记录交付文件完整性。")
    links([("原最终测试指标", "evidence/original_final_test_metrics.json"), ("原最终测试完成记录", "evidence/original_final_test_completion.json"), ("均衡实验完成记录", "evidence/balanced_completion.json"), ("同一验证集模型对照", "evidence/validation_comparison.json"), ("全部39类验证对照CSV", "evidence/validation_per_class_comparison.csv"), ("逐图验证记录复核", "evidence/validation_records_audit.json"), ("既有网页验收证据", "evidence/web_verification_20261018.json"), ("演示流程 答辩说明与现场验收", "DEMO_AND_DEFENSE.md"), ("五项任务验收对照", "ACCEPTANCE.md")])
    render(blocks, out, "植物叶片病害识别与防治参考系统课程报告")
    write_new(out / "report_blocks.json", json.dumps(blocks, ensure_ascii=False, indent=2))
    write_new(out / "sources.json", json.dumps({"created_at": timestamp, "sources": sources,
              "frozen_record_sha256": record_before, "web_checkpoint_sha256": record['checkpoint_sha256'],
              "balanced_checkpoint_sha256": balanced_done['best_checkpoint_sha256'],
              "test_images_opened_this_stage": 0, "raw_images_decoded_this_stage": 0,
              "training_started_this_stage": False, "model_loaded_this_stage": False,
              "web_model_replaced": False, "word_export": "not_created_missing_managed_runtime_and_renderer"}, ensure_ascii=False, indent=2))
    assert digest(ROOT / 'FINAL_MODEL.json') == record_before
    for source, entry in sources.items():
        assert digest(source) == digest(out / entry['copy']) == entry['sha256']
    print(json.dumps({"output": str(out), "report_created": True, "test_images_opened": 0,
                      "training_started": False, "model_loaded": False, "frozen_record_unchanged": True,
                      "sources_verified": len(sources), "reused_old_web_verification": old_web['status']}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
