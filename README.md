# 植物叶片病害识别与防治参考系统

**老师免安装演示版**：本机交付文件为`deliverables/teacher_offline_20261028_v1.zip`，包含独立Windows x64 Python、依赖和原冻结模型，完整解压后双击`启动网页.bat`。包内`项目说明.html`和`README.md`面向老师介绍项目、推理和真实训练结果。见[离线版说明](docs/TEACHER_OFFLINE_README.md)。该大文件不进入Git，需单独交付；尚未在另一台实体电脑验收。

基于 **PyTorch、ResNet18迁移学习和Streamlit** 的中文单图识别项目。支持固定39类，输出Top1、Top3、可复制的中英文类别，以及按预测类别匹配的参考风险与有来源的管理资料。

**当前实现的是病害分类和类别参考风险，不具备经过验证的病情轻中重诊断或直接施药处方能力。** 分类分数不等于诊断可靠性；真实拍摄泛化尚未专项验证。

原创代码采用 [MIT](LICENSE)。原始数据、预训练/微调权重和第三方资料的授权另行说明，见 [第三方内容](THIRD_PARTY_NOTICES.md)。源码仓库：[Alwaemember1/ML-project](https://github.com/Alwaemember1/ML-project)。模型Release尚未发布，GitHub Actions实际运行状态以仓库页面为准。

## 目录

- [项目功能与当前成果](#features)
- [仓库目录与文件职责](#layout)
- [新用户从零部署](#quickstart)
- [准备并配置冻结权重](#weights)
- [启动网页与操作](#web)
- [命令行单图预测](#cli)
- [检查环境与运行测试](#tests)
- [数据治理与清单核对](#data)
- [原实验复现与训练命令](#training)
- [验证与最终测试边界](#evaluation)
- [打包与准备GitHub发布](#release)
- [常见错误](#faq)
- [结果限制与资料来源](#limitations)

<a id="features"></a>
## 项目功能与当前成果

- 单张图片上传、预览、真实模型推理与Top3百分比条形图。
- 39类：26类病害/虫害、12类健康、1类无叶片背景，不是39种病害。
- 中英文类别表默认折叠，文字可拖选复制；中文展示不改变原类别编号。
- 参考风险、复核要点、管理资料来源与单次Markdown报告下载。
- 图片只在内存处理，换图或移除后清除旧预测、资料与下载内容。
- 优先使用GPU，CUDA不可用时可用CPU。模型eval、inference_mode及SHA256检查保留。
- 数据检查、SHA256去重、固定划分、分类头训练、均衡对照、独立重载和报告整理均有真实记录。

| 对象与评价阶段 | Accuracy | Macro F1 | 用途 |
| --- | --- | --- | --- |
| 原模型第3轮验证 | 93.30% | 92.20% | 用于选择轮次 |
| 原冻结模型最终测试 | 93.20% | 92.24% | 网页使用的模型，8567/9192正确 |
| 均衡实验第10轮验证 | 94.92% | 94.21% | 独立开发期对照，未最终测试、未替换网页 |

这些是已有数据集实验成绩，不会随每次上传图片变化。完整39类指标、薄弱类别、混淆对与真实曲线见 [实验结果](docs/RESULTS.md)、[模型说明](docs/MODEL_CARD.md) 和 [数据说明](docs/DATA_CARD.md)。

<a id="layout"></a>
## 仓库目录与文件职责

```text
.
├── README.md                     # 本文件：安装、配置、运行、实验和发布说明
├── LICENSE                       # 原创代码MIT许可
├── CONTRIBUTING.md / SECURITY.md
├── THIRD_PARTY_NOTICES.md
├── app.py                        # Streamlit中文页面
├── predict_image.py              # 单图命令行入口
├── inference.py                  # 共用加载、校验、安全解码和预测
├── project_config.py             # 路径、本地配置和缓存设置，不依赖torch
├── smoke_utils.py                # 共用ResNet18构造与预处理，名字保留兼容含义
├── class_labels.py / plant_care.py
├── balanced_manifest.py           # 均衡视图校验
├── evaluation_metrics.py / evaluation_outputs.py
├── scripts/                      # 数据、训练、验证、报告与部署工具
├── tests/                        # 合成图片、配置、指标和清单测试
├── docs/                         # 模型/数据说明、真实指标、曲线、课程交付说明
├── splits/formal_sha256_seed42_v1/# 固定清单、39类映射和字节哈希锁
├── dataset_config.json           # 原始数据目录配置
├── FINAL_MODEL.json              # 原冻结实验记录，不要手工改写
├── FINAL_MODEL.local.json        # 首次部署生成；不进入Git
├── final_test_registry.json      # 原最终测试登记，防止静默重测
├── requirements-cpu.txt          # 固定CPU版torch/torchvision
├── requirements-gpu.txt          # 固定CUDA版torch/torchvision
├── requirements-web.txt          # Pillow和Streamlit，通过PyPI安装
├── requirements-training.txt     # 旧实验依赖入口，保留追溯
├── .streamlit/config.toml        # 本机8601端口、主题和上传限制
├── .github/workflows/tests.yml   # 无模型、无数据集的CPU合成测试
└── 启动网页.bat                  # Windows启动器，不负责安装环境
```

以下目录只存在于完整本地项目，不上传Git：`.venv/`、`models/`、原始图片目录、`reports/`、`deliverables/`和缓存。**克隆源码不等于已经取得权重、数据或安装环境。**

不要按名字删除`smoke_utils.py`：网页和实验脚本共用它。原根目录的实验命令已迁移到`scripts/`，统一使用`python -m scripts.<脚本名>`，不要沿用历史报告中的旧路径。

<a id="quickstart"></a>
## 新用户从零部署

### 1 取得源码

在GitHub仓库页面选择Code → Download ZIP，解压后进入含app.py的项目根目录；也可使用下列命令克隆源码。源码仓库不含模型和运行环境，完整老师离线包需单独交付。

也可在**替换示例用户名和仓库名后**使用Git：

```powershell
git clone https://github.com/Alwaemember1/ML-project.git leaf-disease-demo
Set-Location .\leaf-disease-demo
```

所有后续命令均从项目根目录执行。无需把项目放在固定D盘路径；本机已有原项目时使用 `Set-Location 'D:\PythonProject3'` 即可。

### 2 选择Python并建立环境

已实际验证：Windows x64、Python3.14.6、torch2.13.0+cu130、torchvision0.28.0+cu130、Pillow12.3.0、Streamlit1.66.0。CPU单图推理也在该环境验证。其他Python版本和Linux/macOS完整流程未验收。

CPU固定wheel的2.13.0/0.28.0已在PyTorch官方索引查询到，但本轮没有另外建立全新CPU环境安装验收。不要把本文版本写成当前最新版本。

```powershell
py -3.14 --version
py -3.14 -m venv .venv
& '.\.venv\Scripts\python.exe' --version
```

如果`py`不存在，先安装Python并确保可执行；也可用已经确认版本的Python可执行文件替换`py -3.14`。不要在原有项目环境中重复创建或重装。下面始终调用`.venv`的解释器，无需手动激活环境。

### 3 将临时目录与下载缓存放在项目内

```powershell
$env:TEMP = "$PWD\.venv\smoke_tmp"
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:XDG_CACHE_HOME = $env:TEMP
$env:PIP_CACHE_DIR = "$PWD\.venv\pip_cache"
```

原项目位于D盘时，这些文件均保留在D盘。不要复制整份旧`.venv`到另一台电脑当作环境安装。

### 4 安装依赖 二选一

**只做CPU演示或没有NVIDIA显卡：**

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-cpu.txt
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-web.txt
& '.\.venv\Scripts\python.exe' -m pip check
```

**使用匹配驱动的NVIDIA GPU：**

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-gpu.txt
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-web.txt
& '.\.venv\Scripts\python.exe' -m pip check
```

CPU与GPU方案选一个，不必都装。torch使用官方专用索引，网页依赖另从PyPI安装；只装`requirements.txt`中的Pillow不能启动识别。既有环境已可用时跳过安装，不为文档整理升级依赖或修改系统显卡驱动。

如目标平台缺少相应wheel，请记录pip完整错误，并参照 [PyTorch官方安装说明](https://pytorch.org/get-started/locally/)选择兼容组合；更换环境后需另行验证。不要改模型或删哈希检查来解决安装问题。

<a id="weights"></a>
## 准备并配置冻结权重

### 1 自行取得完整检查点

Git中不含权重，当前没有公开的项目权重下载URL。需要向项目维护者取得有权使用的完整检查点，或使用已有老师包中的同一文件；仓库不会自动下载替代数据或模型。

推荐放置：

```text
models/final_resnet18.pt
```

唯一接受的原网页模型SHA256为：

```text
1c464873a01b10e1dce89211859a0a9bfe1448a765ecfaef4ed57089436c6b2f
```

PowerShell可先查看：

```powershell
Get-FileHash .\models\final_resnet18.pt -Algorithm SHA256
```

不要将ImageNet原始1000类权重、均衡实验模型、随机权重或重命名文件冒充该检查点。

### 2 生成自己的本地配置

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.configure_model --checkpoint .\models\final_resnet18.pt
```

此命令只校验文件哈希并创建`FINAL_MODEL.local.json`，不加载模型、不下载、不训练、不修改原`FINAL_MODEL.json`。类别映射、预处理、轮次、划分标识和模型哈希沿用原冻结记录；指标指向仓库内真实结果。

已有本地配置时命令会拒绝覆盖。需要另一个配置可显式给出新文件，并指定其位置：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.configure_model --checkpoint .\models\final_resnet18.pt --output .\FINAL_MODEL.demo.local.json
$env:LEAF_MODEL_RECORD = 'FINAL_MODEL.demo.local.json'
```

配置选择顺序：`LEAF_MODEL_RECORD` → 根目录`FINAL_MODEL.local.json` → 原`FINAL_MODEL.json`。原记录内的D盘路径仅用于历史追溯，不要求新用户创建同样目录。文件路径相对于配置文件所在目录解析；不同盘符时可保留绝对路径。

原项目用户如果已有有效的原记录与权重，可继续使用，不必生成新配置。

<a id="web"></a>
## 启动网页与操作

先检查：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.doctor
# 可选：额外查询实际CUDA状态。
& '.\.venv\Scripts\python.exe' -m scripts.doctor --gpu
```

`ok: true`表示依赖、配置、模型哈希和指标文件基本可用，不代表新环境已完成完整预测验收。缺文件时退出码为1，并说明下一步。

### 最快启动

Windows双击根目录`启动网页.bat`，或者：

```powershell
& '.\启动网页.bat'
```

启动器优先使用包内`.venv`；老师包仍位于原项目`deliverables/包名/`时可回退复用原项目环境。它不安装Python、依赖或权重，找不到环境时按首次部署步骤处理。端口占用时只提示，不结束其他程序。

### 直接启动与换端口

```powershell
& '.\.venv\Scripts\python.exe' -m streamlit run .\app.py --server.address 127.0.0.1 --server.port 8601 --server.headless true --browser.gatherUsageStats false
```

浏览器访问 **http://127.0.0.1:8601**。当前终端保持运行，Ctrl+C停止；批处理询问是否终止时输入Y。需要不同端口时仅将命令中的8601改成8602，并访问相应地址。旧报告里的8501是历史端口。

仅监听本机，不自动打开浏览器。本项目未按公网多用户生产服务验收，不包含登录、数据库或聊天功能。

### 页面使用

1. 展开39类中英文表，查看支持范围，拖选文字后Ctrl+C复制。
2. 上传单张照片并检查预览；支持JPG/JPEG/JPE/JFIF、PNG、WebP、BMP、TIF/TIFF、静态GIF和AVIF，共11个后缀。
3. 点击“开始识别”，查看Top1及Top3分数。首次加载模型较慢，后续交互复用缓存。
4. 查看与Top1对应的参考风险、复核要点、管理资料和来源，按需下载本次Markdown报告。
5. 更换或移除图片，旧预测、旧资料与下载内容会清除。

文件上限10MiB、像素上限2000万，真实格式与后缀须一致；不支持HEIC/RAW、动画和多页图片。不要关闭Pillow安全限制。页面本身为中文，浏览器自动翻译可能改乱病名与句子。

“参考风险：中”是按类别设置的提醒，不是当前叶片病情为中度。“模型与实验结果”显示历史测试成绩，上传另一张图时不会改变。

<a id="cli"></a>
## 命令行单图预测

```powershell
& '.\.venv\Scripts\python.exe' .\predict_image.py 'D:\photos\leaf.jpg'
& '.\.venv\Scripts\python.exe' .\predict_image.py 'D:\photos\leaf.webp' --device cpu --output .\prediction.json
```

路径替换成你有权使用的图片。`--device`支持auto、cpu、cuda:0；显式要求不可用GPU时会报错，不生成假结果。默认只输出JSON，不保存上传图片；`--output`拒绝覆盖已有文件。命令行与网页共用加载、预处理、39类映射及推理逻辑。

<a id="tests"></a>
## 检查环境与运行测试

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' -m scripts.prepare_formal_split --metadata-only
```

默认测试仅用合成图片、临时元数据和仓库内清单，不依赖模型权重或原始数据。网页真实权重集成测试默认跳过，避免CI暗中依赖本机模型。

取得并配置权重后，可显式执行合成图片的真实模型上传检查：

```powershell
$env:LEAF_RUN_MODEL_TESTS = '1'
& '.\.venv\Scripts\python.exe' -m unittest tests.test_image_formats -v
Remove-Item Env:\LEAF_RUN_MODEL_TESTS
```

该检查证明格式和代码流程，不计算数据集准确率，不代表真实拍摄泛化。首次环境请再人工确认一次实际上传、识别、换图清理、资料与报告下载。

GitHub Actions文件已准备为Windows + Python3.14 + CPU依赖，运行默认测试和清单元数据检查；只有实际推送到GitHub并查看运行记录后，才能声称远端CI通过。

本次重整的[本地验证记录](docs/results/opensource_verification.json)：默认24项测试，23通过、1项真实权重测试显式跳过；另行开启的11种后缀真实模型AppTest通过；18个CLI帮助入口通过。在不含权重和数据的全新源码副本中，禁止torch.load后默认测试仍通过，清单元数据核对正常。同一内存合成图片的39类CPU概率与重整前完全一致。没有重新训练、重新评价测试集，未在全新CPU环境或远端Actions安装验收。

<a id="data"></a>
## 数据治理与清单核对

### 已有数据记录

原数据61486张、39类，可读性损坏0张；按SHA256选稳定代表后61274张。正式训练42890、验证9192、测试9192。原图、重复副本和旧报告仍在原位，未因整理仓库删除。

数据上游来源、增强同源链路和授权未完整确认，因此本仓库不提供原图下载，不推断可靠原图分组。详见 [数据说明](docs/DATA_CARD.md)。

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

<a id="release"></a>
## 打包与准备GitHub发布

### 老师演示包

仅在拥有原项目模型和完整课程报告时生成全新目录：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.build_teacher_package --output-dir .\deliverables\teacher_demo_new
```

包含一个原冻结模型、必要网页代码、真实指标、报告和实验源码；不复制数据集或虚拟环境，不覆盖旧ZIP。仅克隆代码缺少原reports时不能执行完整课程打包。老师换新电脑仍需首次安装环境。

### 发布检查

导出不含模型和数据的源码ZIP，适合检查后上传GitHub；输出路径必须是不存在的新文件：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.export_source --output .\deliverables\github_source_new.zip
```

它根据Git候选文件生成包，并拒绝常见的本地工件。仍应检查内容和第三方授权，不会自动创建远端或上传。

```powershell
git status --short
git diff --check
git status --ignored --short
```

只提交源码、测试、配置、固定划分元数据和必要真实结果。不要强行添加models、.venv、原图、reports、deliverables、上传文件、.env、Token、私钥或本机IDE/代理状态。

维护本项目时使用github远端；保留已有origin，不强推或覆盖其他历史。首次在已有本地项目中添加远端：

```powershell
git remote add github https://github.com/Alwaemember1/ML-project.git
git push -u github main
```

如果已经存在github远端，无需重复添加；从GitHub新克隆时默认远端名为origin。MIT只覆盖原创代码；模型和图片公开发布前单独确认分发权限。

参与开发与安全问题见 [贡献说明](CONTRIBUTING.md)、[安全说明](SECURITY.md)、[老师交付建议](docs/TEACHER_HANDOFF.md)。根目录README是当前执行说明；旧报告命令属于当时历史记录。

<a id="faq"></a>
## 常见错误

| 现象 | 原因与处理 |
| --- | --- |
| Project Python / No project Python environment | 包不带环境。先创建.venv并安装依赖；包位于原项目deliverables下时可复用原环境 |
| No module named torch / streamlit / PIL | 使用了全局Python或依赖未安装；始终调用.venv解释器，按两步依赖安装 |
| 模型仍指向D盘不存在路径 | 执行scripts.configure_model生成本地配置，或设置LEAF_MODEL_RECORD；不要改原冻结身份 |
| SHA256不符 | 文件版本错误或损坏；重新取得正确检查点，不跳过校验 |
| 指标文件缺失 | 使用仓库docs/results原最终指标，重新创建一个本地配置；不要捏造页面数字 |
| 8601端口被占用 | 先访问确认已有项目服务，或命令行改8602；不要按历史PID结束今天的进程 |
| 直接运行scripts/某文件提示无法导入模块 | 回到根目录，使用python -m scripts.某文件 |
| 严格划分检查提示历史来源文件缺失 | 仅克隆仓库没有完整reports。只看元数据用--metadata-only，复训需恢复原工件 |
| 正式训练停在第1轮 | 按训练节核对first_epoch_ready，再写继续标记；不能直接伪造检查通过 |
| CPU很慢或CUDA不可用 | CPU允许单图演示；--gpu查询状态，GPU安装按官方兼容方案处理，不修改模型来掩盖设备问题 |
| 测试显示skipped | 默认不运行真实权重网页集成测试；这是显式跳过，不是预测测试已通过 |
| 网页病名或句子古怪 | 关闭浏览器自动翻译；项目已提供中文及原英文类别 |
| 网上图片预测错或高分仍错 | 可能超出类别或拍摄分布，分数不保证可靠；不可据此直接认定标签错误 |

<a id="limitations"></a>
## 结果限制与资料来源

本划分已排除已知SHA256字节重复，但未排除增强同源关系，测试成绩可能偏乐观，不能直接代表独立叶片或真实拍摄场景的泛化能力。

资料来源由开发时整理的标题、URL和病害类别映射组成，位于plant_care.py，不是识别时联网检索或来源机构对上传图片的诊断。核对日期为维护日期，链接未来可能变化。风险规则未田间验证，不是病斑严重程度、发生概率或官方检疫等级。实际处置需专业复核。

原图、模型、旧报告及其哈希保留在本地；仓库忽略这些内容不等于删除。当前模型不支持通用域外检测，也不具备精准施药决策能力。

技术参考：[PyTorch安装](https://pytorch.org/get-started/locally/)、[Streamlit项目配置](https://docs.streamlit.io/develop/api-reference/configuration/config.toml)、[MIT许可说明](https://choosealicense.com/licenses/mit/)。运行版本以本项目实际记录和依赖文件为准，官方网页内容会更新。
