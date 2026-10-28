# 数据治理与实验复现

本文面向需要检查数据或复现实验的开发者。所有命令从项目根目录执行。只运行识别页面请先阅读根目录 [README](../README.md)，不需要执行本文训练与评价命令。

## 数据治理与清单核对

### 已有数据记录

原数据61486张、39类，可读性损坏0张；按SHA256选稳定代表后61274张。正式训练42890、验证9192、测试9192。原图、重复副本和旧报告仍在原位，未因整理仓库删除。

数据上游来源、增强同源链路和授权未完整确认，因此本仓库不提供原图下载，不推断可靠原图分组。详见 [数据说明](DATA_CARD.md)。

### 对自己已有数据执行检查

`dataset_config.json`的dataset_root可填绝对路径，或相对于项目根目录的路径。直接包含类别文件夹的一层才是数据根目录。

```powershell
# 仅在确实需要新的全量可读性检查时执行，会读取图片并新建报告。
& '.\.venv\Scripts\python.exe' -m scripts.check_dataset --dataset-root 'D:\data\your_dataset' --output-root .\reports
```

该工具只读原图并生成新报告，不移动、覆盖或删除原始图片。已有可读性报告时无需为了启动网页再扫六万张图片。

### 只核对仓库内清单

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.prepare_formal_split --metadata-only
```

此模式校验文件哈希锁、类别、数量及跨集合路径/SHA256交集，不访问原图，输出`source_provenance_checked: false`、smoke来源检查为null。**它不等同完整来源审计，也不证明没有增强同源泄漏。**

不带选项的严格校验还需要历史来源及smoke清单：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.prepare_formal_split
```

缺少原reports工件或其哈希不符时应当失败，不能改锁、跳过严格检查来冒充复现。`--create`仅适用于首次显式创建；本仓库已有固定划分，不要使用它重建或覆盖。

<a id="training"></a>
## 原实验复现与训练命令

**演示网页不需要执行本节。仅克隆仓库不足以重现原实验。** 本项目原训练流程依赖已完成的历史报告、SHA256清单、smoke清单、固定划分、预训练缓存和原图，很多路径记录在锁定配置中。

### 必要工件

| 工件 | 原项目位置/要求 |
| --- | --- |
| 原始图片 | dataset_config.json及固定split_config记录的数据根目录 |
| 可读性清单 | reports/check_20261016_133031_170321/ |
| 字节去重来源清单 | reports/pretrain_20261016_230415/sha256_manifest.csv |
| 预训练缓存 | reports/pretrain_20261016_230415/torch_cache/checkpoints/resnet18-f37072fd.pth |
| smoke抽样记录 | 固定split_config.json中的smoke_manifests |
| 固定划分与锁 | splits/formal_sha256_seed42_v1/，保持字节不变 |
| 原模型与冻结记录 | 只作对照或推理，不接续smoke检查点训练 |

没有这些工件时先阅读CLI的`--help`，不要以“重新扫描一次并改配置”声称复现旧结果。新数据应另立完整实验和记录；本项目未将原实验自动泛化成任意新数据的一键训练框架。

### 环境诊断 不训练

```powershell
New-Item -ItemType Directory -Force .\reports | Out-Null
& '.\.venv\Scripts\python.exe' -m scripts.check_training_environment --output .\reports\environment_new.json --gpu
```

此工具记录环境并执行GPU张量和随机权重ResNet18前向；随机前向通过不表示迁移学习权重已就绪。`--pretrained`会额外下载并测试ImageNet预训练权重，只有明确需要且允许下载时才使用；本轮整理没有调用它。

### 启动新的分类头实验

满足全部原工件和严格来源检查后，可在**全新输出目录**运行：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.train_v1 --batch-size 32 --output-dir .\reports\user_experiment_new
```

主干和BatchNorm固定，只训练新39类分类头；CrossEntropyLoss、Adam、lr0.001、最多10轮、patience3。训练随机裁剪/翻转；验证确定性预处理。数据加载num_workers0，Windows保留main入口保护。当前训练入口要求CUDA，不自动用CPU冒充原GPU实验。

第1轮结束会等待检查，最长30分钟。请在另一个终端查看`first_epoch_ready.json`，确认head_updated为true，主干参数/缓冲不变且无主干梯度；通过后才创建继续标记：

```powershell
$run = Join-Path $PWD 'reports\user_experiment_new'
$ready = Get-Content -LiteralPath (Join-Path $run 'first_epoch_ready.json') -Raw | ConvertFrom-Json
if (-not $ready.head_updated -or -not $ready.frozen_check.backbone_parameters_and_buffers_unchanged -or -not $ready.frozen_check.backbone_gradients_absent) {
    throw '第1轮核对未通过，不应继续。'
}
$gate = Join-Path $run 'continue_after_epoch1.json'
if (Test-Path -LiteralPath $gate) { throw '继续标记已存在，不覆盖。' }
$text = @{split_id=$ready.split_id; checked_epoch=1} | ConvertTo-Json
[System.IO.File]::WriteAllText($gate, $text, [System.Text.UTF8Encoding]::new($false))
```

实际进度查看：

```powershell
Get-Content .\reports\user_experiment_new\training.log -Tail 20
Get-Content .\reports\user_experiment_new\run_state.json -Raw
```

不要因新实验结果回落自行增加轮数或把测试集用于调参。最佳及最后轮检查点、配置、逐轮指标、曲线与源码快照均保存在独立目录。

### 均衡对照及独立重载

均衡入口使用已锁定、来自原训练集的候选清单。原项目候选位置为`reports/course_completion_20261018_193414_578564/report_v2/candidate_balanced_train_seed42.csv`，以及同目录candidate_balance_lock.json；未随仓库上传。

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.train_v1 --balanced-manifest .\reports\course_completion_20261018_193414_578564\report_v2\candidate_balanced_train_seed42.csv --output-dir .\reports\balanced_experiment_new
```

同样执行首轮检查。完成后从training_summary.json读取实际最佳路径，不凭文件名猜测：

```powershell
$run = Join-Path $PWD 'reports\user_experiment_new'
$summary = Get-Content -LiteralPath (Join-Path $run 'training_summary.json') -Raw | ConvertFrom-Json
& '.\.venv\Scripts\python.exe' -m scripts.predict_smoke --checkpoint $summary.best_checkpoint --reference (Join-Path $run 'best_validation_reference.json') --output (Join-Path $run 'reload_new.json')
```

历史训练记录的源码哈希对应当时布局。当前目录重整后，旧评价入口可能拒绝源码锁；应查看当时source_snapshot，不得为重测分数而删除保护。完整历史流程说明与能力边界也保留在PROJECT_STATUS.md中，其中报告链接仅在原完整项目内存在。

<a id="evaluation"></a>
## 验证与最终测试边界

```powershell
# 仅对已完成的均衡实验，在原验证集比较；需要完整原数据和工件。
& '.\.venv\Scripts\python.exe' -m scripts.compare_balanced_validation --run-dir .\reports\balanced_experiment_new
# 仅读取已有日志和对照记录生成报告，不重新推理。
& '.\.venv\Scripts\python.exe' -m scripts.build_balanced_report --run-dir .\reports\balanced_experiment_new
```

第二条还需要该实验的独立重载结果`best_validation_prediction.json`，请按该实验保存规则命名；脚本拒绝覆盖已有报告。上例重载示范的reload_new.json用于说明参数，生成均衡报告时须使用实际约定文件名。

原最终测试已经完成且在final_test_registry.json登记，正常部署**不执行**评价脚本。查看接口：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.evaluate_final --help
```

该入口绑定原冻结实验、轮次和来源锁，不是任意新模型的通用测试器。只有真实实现错误需要重评时才应说明影响、另存结果并使用相应理由；模型分数不理想不是程序错误。不能删除登记来反复比较新模型。

