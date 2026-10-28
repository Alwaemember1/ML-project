# 农作物病虫害识别与防治参考系统

基于 **PyTorch、ResNet18 迁移学习和 Streamlit** 的机器学习课程项目。用户上传单张叶片图片后，系统输出作物与病害类别、Top3 分类分数、类别参考风险和对应防治资料，并支持下载单次识别报告。

本项目对应《机器学习期末课程答辩》实施方案中的 **项目3 基于图像数据的农作物病虫害识别与防治决策**，用于课程演示、实验审查和学习。网页使用固定的39类模型，包含26类病害或虫害、12类健康和1类无叶片背景。

> 当前风险等级由预测类别匹配规则生成，不是从图片测得的病情轻重。分类概率不等于诊断可靠性；系统不提供自动施药处方。

## 目录

- [项目3要求与实现](#项目3要求与实现)
- [主要功能](#主要功能)
- [实验结果](#实验结果)
- [目录结构](#目录结构)
- [环境与依赖](#环境与依赖)
- [下载并导入编辑器](#下载并导入编辑器)
- [安装与配置](#安装与配置)
- [启动与使用](#启动与使用)
- [检查与测试](#检查与测试)
- [数据与训练复现](#数据与训练复现)
- [提交范围与打包](#提交范围与打包)
- [常见问题](#常见问题)
- [局限与许可](#局限与许可)

## 项目3要求与实现

课程背景是面向农业植保部门与种植基地，利用田间或叶片图像辅助病虫害识别与防治决策。项目按课程的五项任务组织：

| 课程要求 | 本项目实现 | 结果与适用范围 |
| --- | --- | --- |
| 图像数据治理 清洗 去重 类别均衡 数据增强 | 图片可读性检查、SHA256字节去重、固定划分、每类等量下采样对照、训练随机裁剪与翻转 | 原始61486张，字节去重后61274张；增强同源关系未排除 |
| 作物与病害多分类识别模型 | ImageNet预训练ResNet18，冻结主干及BatchNorm统计，训练39类分类头 | 网页使用原第3轮冻结模型，已保存训练、验证和最终测试记录 |
| 输出严重程度或风险等级 | 按Top1类别输出高、中、低参考风险；背景类不适用，并展示依据 | 采用课程中的风险等级路线；类别规则原型，未完成图像严重程度或田间风险验证 |
| 支持图片上传的识别可视化系统 | Streamlit中文页面，上传预览、Top1与Top3、中英文类别表 | 本机运行，支持11种图片后缀，不需要原始数据集 |
| 报告与针对性防治方案 | 既有实验指标、曲线、弱类分析；按预测类别展示复核要点、管理资料和来源，下载Markdown报告 | 防治内容供复核参考；未完成田间防效或精准施药验证 |

课程方案给出的参考数据入口为 [TensorFlow Datasets PlantVillage](https://tensorflow.google.cn/datasets/catalog/plant_village)。现有实验使用本地 `Plant_leave_diseases_dataset_with_augmentation` 的39类材料，其上游来源及增强链路尚未完整核实，不能将其直接等同于课程链接中的数据版本。原始数据不随仓库分发，类别标签不包含病斑掩码或严重程度。

## 主要功能

- 上传、预览单张图片，显示Top1及Top3类别和分数。
- 展开39类中英文类别表，支持复制类别文字。
- 展示与Top1对应的参考风险、症状复核要点、防治管理资料及来源。
- 下载当前图片的识别与防治参考报告，换图后清除旧结果。
- 网页与命令行共用模型加载、哈希校验、预处理和推理逻辑。
- 支持CPU推理；匹配的CUDA环境可使用GPU。图片在内存中处理，不自动保存。

处理流程：图片上传 → 格式与大小检查 → EXIF方向修正与RGB转换 → 图像预处理 → ResNet18分类 → Top3结果 → 类别风险与防治参考。

## 实验结果

| 模型与阶段 | Accuracy | Macro F1 | 说明 |
| --- | ---: | ---: | --- |
| 原模型第3轮验证 | 93.30% | 92.20% | 用于选模 |
| 原冻结模型最终测试 | **93.20%** | **92.24%** | 网页所用模型，8567张正确，共9192张 |
| 均衡模型第10轮验证 | 94.92% | 94.21% | 独立开发期对照，未最终测试，未替换网页模型 |

正式划分为训练42890张、验证9192张、测试9192张。原模型实际训练6轮，按验证损失选择第3轮；均衡实验39类各582张，共22698张，训练10轮。两组均冻结主干及BatchNorm统计，只更新分类头，使用Adam、学习率0.001、batch size 32、seed 42。

均衡实验同时改变训练规模及更新次数，不能将差异全部归因于均衡。原模型马铃薯晚疫病测试召回率54.67%，番茄二斑叶螨危害为60.56%，整体准确率不能替代逐类分析。

完整结果保留在 `docs/results/`，主要文件如下。这些是已有实验记录，不会随上传图片改变。

| 文件 | 内容 |
| --- | --- |
| [original_test_metrics.json](docs/results/original_test_metrics.json) | 原模型最终测试完整指标 |
| [original_test_per_class.csv](docs/results/original_test_per_class.csv) | 全部39类的precision、recall及F1 |
| [original_test_confusion_pairs.csv](docs/results/original_test_confusion_pairs.csv) | 错分类别对及比例 |
| [original_training_metrics.csv](docs/results/original_training_metrics.csv) | 原模型逐轮训练记录 |
| [balanced_training_metrics.csv](docs/results/balanced_training_metrics.csv) | 均衡实验逐轮训练记录 |
| [validation_per_class_comparison.csv](docs/results/validation_per_class_comparison.csv) | 两模型在相同验证集的逐类对照 |

![原模型训练曲线](docs/figures/original_training_curves.png)

![均衡模型训练曲线](docs/figures/balanced_training_curves.png)

模型推理使用EXIF方向修正、RGB、短边缩放256、中心裁剪224、双线性插值及ImageNet均值与标准差。中文名称仅用于展示，不修改类别编号、分类分数或拒识阈值。

## 目录结构

```text
ML-project/
├── README.md                     # 项目3说明、安装配置和操作指南
├── app.py                        # Streamlit网页入口
├── predict_image.py              # 命令行单图识别入口
├── inference.py                  # 模型校验、图片解码与推理
├── project_config.py             # 默认配置选择、路径与缓存
├── smoke_utils.py                # 共用模型构造和预处理
├── class_labels.py               # 中英文类别展示
├── plant_care.py                 # 类别风险、防治参考与报告
├── balanced_manifest.py          # 均衡训练视图校验
├── evaluation_metrics.py         # 评价指标计算
├── evaluation_outputs.py         # 评价结果输出
├── models/final_resnet18.pt       # 已发布的冻结模型 约44.87 MB
├── MODEL_CONFIG.json             # 默认相对路径推理配置
├── FINAL_MODEL.json              # 原实验身份与历史位置记录
├── dataset_config.json           # 数据治理使用的数据根目录配置
├── final_test_registry.json      # 原最终测试登记
├── scripts/                      # 数据、训练、评价、打包和诊断脚本
├── tests/                        # 配置、格式、指标、导出等测试
├── splits/                       # 固定训练验证测试清单及哈希锁
├── docs/
│   ├── results/                  # 真实实验指标、逐类CSV和验证记录
│   └── figures/                  # 原模型与均衡实验训练曲线
├── requirements*.txt             # 依赖版本清单 不包含安装目录
├── .github/workflows/tests.yml   # Windows CPU自动化检查
├── .streamlit/config.toml        # 页面主题、端口和上传限制
└── 启动网页.bat                  # Windows启动入口
```

`smoke_utils.py` 同时被网页和实验脚本使用，不能按名称删除。所有 `scripts/` 命令统一从根目录以 `python -m scripts.<模块>` 执行。

## 环境与依赖

建议首次演示采用 **Windows x64、Python 3.14 和CPU环境**。Windows + Python 3.14的CPU安装、默认配置校验及真实模型合成图片上传测试已在GitHub Actions通过；Linux、macOS及另一台实体电脑的完整流程未验收。

| 组件 | 固定版本或要求 | 用途 |
| --- | --- | --- |
| Python | 3.14 系列 | 运行程序及创建虚拟环境 |
| torch | 2.13.0+cpu 或 2.13.0+cu130 | 模型推理与训练 |
| torchvision | 0.28.0+cpu 或 0.28.0+cu130 | ResNet18与图像变换 |
| Pillow | 12.3.0 | 图片读取及格式检查 |
| Streamlit | 1.66.0 | 中文识别页面 |
| PyCharm或VS Code | 任选 支持Python | 编辑与运行源码 |
| Git | 可选 | 克隆及版本管理；下载ZIP无需Git |

原实验环境记录为Windows、Python3.14.6和CUDA构建；CI使用Python3.14系列CPU构建。课程方案中的“Python3.7及以上”是通用建议，本仓库使用上述固定依赖，不保证这些依赖支持Python3.7。

依赖清单不会随源码附带安装好的Python、PyTorch、Streamlit或虚拟环境。首次安装需要联网及足够磁盘空间；完成安装后，正常推理不下载模型。无需额外安装Anaconda、数据库、Node.js，也无需配置API密钥。

## 下载并导入编辑器

任选一种方式取得源码：

```powershell
git clone https://github.com/Alwaemember1/ML-project.git
Set-Location .\ML-project
```

或在GitHub选择 **Code → Download ZIP**，完整解压后进入含 `app.py` 的根目录。完整仓库已包含冻结模型，不能只下载几个Python文件。

**PyCharm**：选择Open打开项目根目录，完成下面的虚拟环境创建后，在项目Python Interpreter中选择 `.venv\Scripts\python.exe`。

**VS Code**：打开项目文件夹，安装Python扩展，使用“Python: Select Interpreter”选择 `.venv\Scripts\python.exe`。

编辑器负责打开源码，不会自动安装程序依赖。以下命令均在项目根目录的终端中执行，Windows示例使用PowerShell。

## 安装与配置

### 1 安装Python并创建环境

安装Windows x64 Python3.14后执行：

```powershell
py -3.14 --version
py -3.14 -m venv .venv
& '.\.venv\Scripts\python.exe' --version
```

如果 `py` 不存在，确认Python已安装，或将它替换为Python3.14解释器的完整路径。不要在已经可用的项目环境中重复创建或升级依赖。后续直接调用环境解释器，无需激活环境或修改PowerShell执行策略。

### 2 安装依赖

默认CPU方案，适用于没有NVIDIA显卡的电脑：

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-cpu.txt
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-web.txt
& '.\.venv\Scripts\python.exe' -m pip check
```

若明确需要GPU，且显卡、驱动支持CUDA13.0构建，将第一条中的 `requirements-cpu.txt` 替换为 `requirements-gpu.txt`，其余不变。CPU和GPU方案二选一。显式选择GPU前可运行 `python -m scripts.doctor --gpu` 查询状态。

**只安装 `requirements.txt` 不够**：该文件只列基础图片依赖，无法独立运行识别系统。PyTorch使用专用索引，网页依赖单独从PyPI安装；不要把两种索引混在一条安装命令里。

### 3 确认模型与配置

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.doctor
```

默认使用 `MODEL_CONFIG.json`，模型路径为 `models/final_resnet18.pt`，指标路径为 `docs/results/original_test_metrics.json`；正常部署无需修改D盘路径。`ok: true` 表示依赖版本可查询、模型哈希与指标基本检查通过，仍应实际上传图片完成运行验收。

冻结模型SHA256：

```text
1c464873a01b10e1dce89211859a0a9bfe1448a765ecfaef4ed57089436c6b2f
```

仅在模型存放位置变化时生成本地配置：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.configure_model --checkpoint 'D:\models\final_resnet18.pt'
```

该命令创建 `FINAL_MODEL.local.json` 并拒绝覆盖已有文件。配置选择顺序为 `LEAF_MODEL_RECORD` 环境变量 → `FINAL_MODEL.local.json` → `MODEL_CONFIG.json` → 历史 `FINAL_MODEL.json`。模型哈希不符时应取得正确文件，不能删除校验。

## 启动与使用

Windows完成安装后，可双击 `启动网页.bat`；也可以在终端直接启动：

```powershell
& '.\.venv\Scripts\python.exe' -m streamlit run .\app.py --server.address 127.0.0.1 --server.port 8601 --server.headless true --browser.gatherUsageStats false
```

浏览器打开 **http://127.0.0.1:8601**。启动窗口保持运行，使用Ctrl+C停止。端口占用时可把命令中的8601改成8602，再访问对应地址；不要结束不明进程。

1. 展开类别表，了解模型支持的作物及病害范围。
2. 上传单张图片，检查预览。
3. 点击“开始识别”，查看Top1、Top3及分数。
4. 阅读对应类别的参考风险、复核要点和防治资料。
5. 按需下载当前识别报告；更换或移除图片后旧结果清除。

支持 `.jpg`、`.jpeg`、`.jpe`、`.jfif`、`.png`、`.webp`、`.bmp`、`.tif`、`.tiff`、`.gif`、`.avif`，共11种后缀。文件不超过10 MiB，图像不超过2000万像素；只接受单帧静态图片，真实格式须与后缀一致，不支持HEIC或RAW。

也可使用命令行预测：

```powershell
& '.\.venv\Scripts\python.exe' .\predict_image.py 'D:\photos\leaf.jpg' --device cpu
& '.\.venv\Scripts\python.exe' .\predict_image.py 'D:\photos\leaf.jpg' --device cpu --output .\prediction.json
```

将示例路径替换为实际图片；输出JSON默认仅打印，指定 `--output` 时拒绝覆盖已有文件。正常识别无需下载数据集、执行训练或启动数据库。

## 检查与测试

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' -m scripts.prepare_formal_split --metadata-only
```

默认单元测试不需要原图，真实模型网页测试默认跳过。需要显式检查已附模型的上传推理时：

```powershell
$env:LEAF_RUN_MODEL_TESTS = '1'
& '.\.venv\Scripts\python.exe' -m unittest tests.test_image_formats.ImageFormatTests.test_web_upload_preview_and_real_prediction -v
Remove-Item Env:\LEAF_RUN_MODEL_TESTS
```

CI执行依赖安装、默认模型与配置检查、默认测试、11种后缀的真实模型合成图片上传推理和划分元数据检查。合成图检查证明程序流程，不计算识别准确率，也不代表真实场景泛化。运行记录见仓库 **Actions**。

## 数据与训练复现

完整训练源码、固定划分及结果记录均已保留，但原图、预训练缓存、历史审计报告和均衡候选清单没有随仓库分发。`dataset_config.json` 用于数据治理工具；仅改它不能迁移全部历史训练流程。

复训需要恢复原始图片及固定配置引用的历史工件、准备ImageNet缓存和可用CUDA环境，再执行严格划分校验。必要工件和开发命令见下方折叠说明。缺少工件时程序应明确失败，不能改锁绕过检查来宣称复现。

原最终测试已登记，不作为任意新模型的反复调参工具。正式清单、原模型、历史成绩及来源锁保持不变。若新增数据或模型，应另建独立实验。


<details>
<summary>开发者可展开查看数据检查与训练复现步骤</summary>

### 数据检查

`dataset_config.json` 的 `dataset_root` 应指向直接包含类别子文件夹的数据根目录。下面的可读性检查只读原图并生成新报告，不移动或删除原始图片：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.check_dataset --dataset-root 'D:\data\your_dataset' --output-root .\reports
```

`python -m scripts.prepare_formal_split --metadata-only` 只检查仓库清单的哈希、数量、类别及跨集合路径和字节哈希交集，不读取原图，不证明排除了增强同源泄漏。不带该选项的严格检查还需恢复历史来源与smoke清单；已有划分不要使用 `--create` 重建或覆盖。

#### 必要工件

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

#### 环境诊断 不训练

```powershell
New-Item -ItemType Directory -Force .\reports | Out-Null
& '.\.venv\Scripts\python.exe' -m scripts.check_training_environment --output .\reports\environment_new.json --gpu
```

此工具记录环境并执行GPU张量和随机权重ResNet18前向；随机前向通过不表示迁移学习权重已就绪。`--pretrained`会额外下载并测试ImageNet预训练权重，只有明确需要且允许下载时才使用；本轮整理没有调用它。

#### 启动新的分类头实验

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

#### 均衡对照及独立重载

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

历史训练记录的源码哈希对应当时布局。当前目录重整后，旧评价入口可能拒绝源码锁；应查看当时source_snapshot，不得为重测分数而删除保护。原开发历史可从Git提交记录追溯。


### 验证与最终测试

完成独立均衡实验后，可用 `python -m scripts.compare_balanced_validation --run-dir <实验目录>` 比较原验证集；这不构成新的最终测试。

`python -m scripts.evaluate_final --help` 可查看原最终测试入口。该入口绑定原实验与来源锁，不是任意新模型的通用测试器；执行时还必须用 `--project-status` 提供Git历史保留的原验收说明快照，原路径核对与重复测试登记保护仍然有效。缺少历史工件、来源锁不符或只是希望提高分数时，不应重新评价。

</details>

## 提交范围与打包

项目只保留这一份README作为说明入口；实验指标和曲线在docs内，开发过程可从Git历史追溯。完整仓库或GitHub Download ZIP包含源码、依赖清单、默认配置、唯一冻结模型、固定划分、测试和必要实验结果。模型约44.87 MB是演示所需权重，继续保留；无需打包已安装的依赖。

不上传 `.venv/`、Python安装目录、依赖安装目录、原始数据、其他权重、缓存、上传图片、`reports/`、`deliverables/`、密钥或编辑器个人配置。Git忽略这些目录不等于删除本地原件。

需要不含模型的纯源码审查包时，可在Git克隆目录执行：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.export_source --output .\deliverables\source_review.zip
```

该命令需要Git元数据，自动跳过唯一已发布模型，并在清单中注明不含权重。纯源码审查包不能直接推理；下载ZIP且没有 `.git` 时，应直接使用原完整ZIP，或先通过Git克隆再导出。

`build_teacher_package` 和 `build_teacher_offline` 是依赖本地历史报告或环境工件的专用打包工具，不是首次运行入口；仅克隆仓库不能直接生成完整离线环境包。GitHub仓库也不包含免安装Python环境。

本README覆盖项目3的源码操作与实现说明。课程实施方案另外要求小组总结报告、汇报PPT、个人记录与总结等材料，应按实际分工另行准备，不能以源码仓库替代全部课程材料。

## 常见问题

| 现象 | 处理方法 |
| --- | --- |
| 找不到torch、streamlit或PIL | 在编辑器选择项目 `.venv`，按CPU或GPU清单和web清单两步安装 |
| 双击启动器提示没有Python环境 | 先创建 `.venv` 并安装依赖；启动器不负责安装 |
| 安装失败或没有兼容wheel | 核对Python3.14、Windows x64、下载索引和完整pip错误，不随意升级固定版本 |
| 模型路径指向旧D盘目录 | 检查是否存在旧本地配置或LEAF_MODEL_RECORD覆盖；正常默认应使用MODEL_CONFIG.json |
| 模型缺失或SHA256不符 | 重新取得完整仓库及原冻结模型，不跳过校验 |
| 网页无法访问 | 保持启动终端运行，确认地址与端口；端口占用时改用8602 |
| 图片被拒绝 | 检查后缀、真实格式、文件大小、像素及是否动画或多页 |
| 预测分数很高但病名不对 | 可能超出39类范围或拍摄分布，分数不能保证诊断正确 |
| 严格划分检查缺少历史reports | 仅核对仓库清单使用metadata-only；完整复训需要恢复历史工件 |
| 默认测试有skipped | 真实模型测试默认显式跳过，按上方命令单独开启 |

## 局限与许可

本模型不覆盖全部农业病虫害；背景类也不是经过验证的通用非叶片检测器。已排除已知字节重复，尚未排除增强同源泄漏；真实拍摄、跨设备和田间泛化未专项验证。

防治资料按预测类别匹配，来源保存在 `plant_care.py`；不是对当前图片进行联网确诊。健康预测不能排除其他问题，风险规则未经田间验证，实际处置需结合现场情况与植保复核。

原创代码采用 [MIT许可证](LICENSE)。许可证仅覆盖维护者有权授权的原创代码与说明，不授予第三方数据、预训练权重、机构商标或资料的新权利。

| 内容 | 许可与来源说明 |
| --- | --- |
| PyTorch、torchvision、Streamlit、Pillow及其依赖 | 通过依赖清单安装，遵循各上游项目许可 |
| ImageNet预训练ResNet18及本项目微调模型 | 已附唯一冻结检查点；不自动纳入原创代码MIT许可，分发时需分别核查上游权利 |
| 本地增强叶片数据集 | 上游来源、增强链路及分发许可尚未完整核实，原图不上传 |
| 大学推广与农业机构防治资料 | 来源保存在plant_care.py，不表示来源机构为预测结果背书 |
| 实验曲线与指标 | 保留真实记录及评价局限，不保证生产场景效果 |

上游项目：[PyTorch](https://github.com/pytorch/pytorch)、[torchvision](https://github.com/pytorch/vision)、[Streamlit](https://github.com/streamlit/streamlit)、[Pillow](https://github.com/python-pillow/Pillow)。

项目仓库：https://github.com/Alwaemember1/ML-project
