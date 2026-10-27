"""均衡对照：只在原固定验证集上一次性比较新旧模型，不读取测试图片。"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

from smoke_utils import (ROOT, build_transform, model_for_classes, save_json, seed_torch,
                         sha256_file, tensor_digest)
from train_smoke import SmokeImages
from prepare_formal_split import SPLIT, read_rows, verify_split
from evaluation_metrics import classification_metrics

import torch
from torch.utils.data import DataLoader


def state_digests(model):
    return {name:tensor_digest(value) for name,value in model.state_dict().items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True,type=Path)
    args=parser.parse_args()
    run=args.run_dir.resolve()
    output=run / "validation_comparison"
    output.mkdir(exist_ok=False)
    summary=json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
    if summary["purpose"] != "balanced_subset_head_comparison":
        raise ValueError("此脚本仅用于均衡对照实验")
    record_path=ROOT / "FINAL_MODEL.json"
    record=json.loads(record_path.read_text(encoding="utf-8"))
    record_hash=sha256_file(record_path)
    baseline_path=Path(record["checkpoint"])
    baseline_hash=sha256_file(baseline_path)
    if baseline_hash != record["checkpoint_sha256"]:
        raise ValueError("原冻结模型哈希不符")
    balanced_path=Path(summary["best_checkpoint"])
    balanced_hash=sha256_file(balanced_path)
    split=verify_split()  # 测试仅清单锁元数据；没有测试Dataset/DataLoader。
    if split["split_id"] != record["split_id"] or split["split_id"] != summary["training_config"]["split_id"]:
        raise ValueError("新旧模型不是同一固定划分")
    if not torch.cuda.is_available():
        raise RuntimeError("已验证GPU不可用，停止对照，不自动切换设备")
    seed_torch(42)
    checkpoints={"baseline":torch.load(baseline_path,map_location="cpu",weights_only=True),
                 "balanced":torch.load(balanced_path,map_location="cpu",weights_only=True)}
    mapping=record["class_to_idx"]
    info=record["preprocessing"]
    models={}
    for name,checkpoint in checkpoints.items():
        if checkpoint["class_to_idx"] != mapping or checkpoint["preprocessing"] != info:
            raise ValueError("新旧类别映射或验证预处理不一致")
        model=model_for_classes(39)
        model.load_state_dict(checkpoint["state_dict"],strict=True)
        models[name]=model.requires_grad_(False).eval().to("cuda:0")
    before={name:state_digests(model) for name,model in models.items()}
    config=json.loads((SPLIT / "split_config.json").read_text(encoding="utf-8"))
    rows=read_rows(SPLIT / "validation.csv")
    for row in rows:row["class_idx"]=int(row["class_idx"])
    if len(rows) != 9192:
        raise ValueError("原固定验证集样本数改变")
    loader=DataLoader(SmokeImages(Path(config["dataset_root"]),rows,build_transform(info)),
                      batch_size=32,shuffle=False,num_workers=0)
    criterion=torch.nn.CrossEntropyLoss(reduction="none")
    matrices={name:[[0]*39 for _ in range(39)] for name in models}
    loss_sums={name:0.0 for name in models}
    names=[name for name,_ in sorted(mapping.items(),key=lambda item:item[1])]
    fieldnames=["relative_path","sha256","true_idx","true_class","baseline_pred_idx","baseline_pred_class",
                "baseline_top1_probability","baseline_loss","balanced_pred_idx","balanced_pred_class",
                "balanced_top1_probability","balanced_loss"]
    started=time.perf_counter()
    cursor=0
    with (output / "predictions.csv").open("x",encoding="utf-8-sig",newline="") as file, torch.inference_mode():
        writer=csv.DictWriter(file,fieldnames=fieldnames);writer.writeheader()
        for step,(inputs,labels) in enumerate(loader,1):
            inputs,labels=inputs.to("cuda:0"),labels.to("cuda:0")
            true=labels.cpu().tolist()
            batch={}
            for name,model in models.items():
                logits=model(inputs)
                if logits.shape!=(len(labels),39) or not torch.isfinite(logits).all().item():
                    raise ValueError("对照输出形状或有限性异常")
                losses=criterion(logits,labels).cpu().tolist()
                probabilities=logits.softmax(dim=1)
                scores,predictions=probabilities.max(dim=1)
                batch[name]={"prediction":predictions.cpu().tolist(),"score":scores.cpu().tolist(),"loss":losses}
                loss_sums[name]+=math.fsum(losses)
                for label,prediction in zip(true,batch[name]["prediction"]):matrices[name][label][prediction]+=1
            for index,label in enumerate(true):
                source=rows[cursor+index]
                item={"relative_path":source["relative_path"],"sha256":source["sha256"],
                      "true_idx":label,"true_class":names[label]}
                for name in models:
                    prediction=batch[name]["prediction"][index]
                    item.update({f"{name}_pred_idx":prediction,f"{name}_pred_class":names[prediction],
                                 f"{name}_top1_probability":batch[name]["score"][index],f"{name}_loss":batch[name]["loss"][index]})
                writer.writerow(item)
            cursor+=len(labels)
            if step%100==0 or step==len(loader):print(f"validation comparison batch {step}/{len(loader)}, samples={cursor}",flush=True)
    torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    metrics={name:{**classification_metrics(matrix,names),"loss":loss_sums[name]/cursor}
             for name,matrix in matrices.items()}
    if cursor!=9192 or any(state_digests(model)!=before[name] for name,model in models.items()):
        raise AssertionError("验证样本数不符或推理时模型状态改变")
    if sha256_file(record_path)!=record_hash or sha256_file(baseline_path)!=baseline_hash or sha256_file(balanced_path)!=balanced_hash:
        raise AssertionError("检查点或冻结记录在对照期间改变")
    baseline_summary=json.loads((Path(record["training_run"]) / "training_summary.json").read_text(encoding="utf-8"))
    baseline_best=next(item for item in baseline_summary["history"] if item["epoch"]==record["epoch"])
    if abs(metrics["baseline"]["accuracy"]-baseline_best["validation_accuracy"])>1e-12:
        raise AssertionError("原模型验证准确率与原记录不符")
    if abs(metrics["balanced"]["accuracy"]-summary["history"][summary["best_epoch"]-1]["validation_accuracy"])>1e-12:
        raise AssertionError("均衡模型重载验证准确率与训练日志不符")
    result={"purpose":"validation_only_balanced_comparison","samples":cursor,"device":"cuda:0",
            "baseline_checkpoint":str(baseline_path),"baseline_sha256":baseline_hash,
            "balanced_checkpoint":str(balanced_path),"balanced_sha256":balanced_hash,
            "split_id":split["split_id"],"baseline_best_epoch":record["epoch"],"balanced_best_epoch":summary["best_epoch"],
            "metrics":metrics,"seconds":seconds,"test_images_opened":0,"test_evaluated":False,
            "web_model_replaced":False,"all_model_state_tensors_unchanged":True,
            "limitation":"只在已用于选模的验证集比较；下采样改变训练规模和更新次数，不能独立归因于均衡；增强同源未排除，原测试不重用选模型。"}
    save_json(output / "comparison.json",result)
    with (output / "per_class_comparison.csv").open("x",encoding="utf-8-sig",newline="") as file:
        fields=["class_idx","class_name","support","baseline_precision","baseline_recall","baseline_f1",
                "balanced_precision","balanced_recall","balanced_f1","recall_change","f1_change"]
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for baseline,balanced in zip(metrics["baseline"]["classes"],metrics["balanced"]["classes"]):
            item={key:baseline[key] for key in ("class_idx","class_name","support")}
            item.update({f"{part}_{key}":entry[key] for part,entry in (("baseline",baseline),("balanced",balanced))
                         for key in ("precision","recall","f1")})
            item.update(recall_change=balanced["recall"]-baseline["recall"],f1_change=balanced["f1"]-baseline["f1"])
            writer.writerow(item)
    (output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key:value for key,value in result.items() if key!="metrics"},ensure_ascii=False,indent=2),flush=True)
    print(json.dumps({name:{key:row[key] for key in ("loss","accuracy","macro_precision","macro_recall","macro_f1")}
                      for name,row in metrics.items()},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
