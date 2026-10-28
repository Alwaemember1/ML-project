# 项目状态：课程报告与演示资料已整理，原网页模型继续冻结

## 发布冻结模型 2026年10月28日

按用户明确要求，将原第3轮检查点的只读副本作为models/final_resnet18.pt加入GitHub源码仓库，44,866,251字节，SHA256仍为1c464873a01b10e1dce89211859a0a9bfe1448a765ecfaef4ed57089436c6b2f。新增MODEL_CONFIG.json，相对指向模型和已有真实测试指标；原FINAL_MODEL.json和原报告检查点不改。配置优先级为环境变量、本地覆盖、仓库配置、历史记录。依赖仍需安装，不将源码包称为免安装版。模型及上游内容不自动适用原创代码MIT许可，其他模型、原始数据、环境和老师离线ZIP保持不上传。配置检查及3项配置回归通过；没有训练、调整预处理或重新评价测试集。

## 老师离线运行版 2026年10月28日

新增独立交付目录`deliverables/teacher_offline_20261028_v1`，内含Python3.14.6、现有PyTorch2.13.0+cu130/torchvision0.28.0+cu130/Streamlit1.66.0及依赖、Python扩展需要的DLL、一个原第3轮冻结模型、相对路径配置、原真实指标和训练记录。不是直接复制不可搬迁的.venv；独立解释器采用python314._pth，搜索路径只在包内，不依赖D:/Anaconda或原项目环境。未安装、升级或更改原环境。

双击包内`启动网页.bat`即可运行，默认CPU并隐藏GPU选择，避免要求老师配置CUDA；包内包含原CUDA构建的依赖，所以体积较大。优先127.0.0.1:8601，占用时选择随后9个可用端口；启动后尝试打开默认浏览器，Ctrl+C停止。`启动前检查.bat`核对模型、依赖并用合成图片检查CPU预测。临时目录在包内.runtime，原图和旧交付包保持不动。

老师文档为包内README.md和项目说明.html，介绍课程要求完成程度、39类组成、完整推理流程、数据划分、实际6轮训练/第3轮选模、逐轮真实指标和耗时、原最终测试以及能力限制。附原训练配置、日志、曲线和完成记录；明确均衡实验只提供验证结果、未替换网页模型。HTML仅作静态排版，不宣称已经验证实体打印。

实际验收：独立包内CPU合成图的39维概率与整理前参考完全一致（最大差0）；通过Streamlit AppTest检查初始页面、真实93.20%/92.24%指标、PNG上传预览识别、WebP替换清除旧结果、连续识别、移除图片、损坏文件及伪装后缀；临时记录检查缺失模型和SHA不符错误。验收中阻断非回环连接，sys.path全部位于包内。原始数据集图片读取0张，未训练或重新评价测试集。

已实际运行包内BAT（--no-browser）启动，http://127.0.0.1:8601健康检查200/ok，首页200。页面交互采用AppTest，未声称鼠标实际点击。未在另一台实体电脑验证，自动打开浏览器、其他Windows版本、无NVIDIA驱动的另一台电脑仍需目标机验收；本包默认CPU，不宣称GPU推理已验收。本地普通受限执行环境的回环连接检查曾阻塞，允许本机回环后通过；最初测试阻断全部socket连接误伤Windows asyncio，已将检查限定为阻断非回环连接，并非模型实现错误。

本地记录在包内results/offline_verification.json；最终封包文件清单和SHA在DELIVERY_MANIFEST.json及ZIP旁.sha256。本轮不发布公网，不上传包含环境和权重的包到源码仓库。

封包后实际完整解压到`reports/验收 空格/teacher_offline_20261028_v1`，21,950个清单文件SHA256全部一致；子进程PATH只保留Windows System32，使用新位置包内Python成功完成39类CPU推理，未借用原项目环境。ZIP为2,118,695,437字节，清单文件解压共3,281,963,908字节；ZIP SHA256为`23eddc47da09867a291d91eac2c87ec4736ddac388f55cc43ada4c3d8afe7fef`。完整解压及校验耗时167.29秒，证据为`reports/teacher_offline_20261028/archive_verification.json`。端口占用时8601→8602选择通过，原回归检查24项中23通过、1项按默认配置跳过；另行真实模型AppTest通过。演示验收服务已停止，老师可重新双击启动。原冻结模型、记录、最终测试登记与正式划分未改变。

## GitHub开源版目录与文档整理 2026年10月27日

按用户要求，将项目整理为准备上传GitHub的源码仓库。用户明确选择MIT，仅覆盖原创代码；新增LICENSE、THIRD_PARTY_NOTICES、CONTRIBUTING、SECURITY、模型和数据说明。第三方数据、权重及网页资料不自动继承MIT，原图与模型不进入源码包，本轮未建立或发布GitHub远端。

15个原根目录命令行脚本迁入scripts/，原单元测试迁入tests/，统一使用python -m scripts.<名称>。根目录保留app.py、predict_image.py和共用模块；历史报告中的旧命令仅用于追溯，当前执行说明以根目录README为准。迁移前已提交源码完整备份至reports/github_restructure_20261027/source_before.zip，原冻结记录、模型、最终测试登记和固定划分字节不改。

根目录README详细说明CPU/GPU安装、首次权重配置、启动、预测、测试、元数据核对、原实验工件、训练首轮门控、验证及最终测试边界、打包、GitHub发布和常见错误。CPU版固定torch2.13.0/torchvision0.28.0在官方索引查询可用，未为整理任务重装现有环境。

新增project_config.py及scripts.configure_model、scripts.doctor。新部署以FINAL_MODEL.local.json或LEAF_MODEL_RECORD设置本地路径，保留原FINAL_MODEL.json不覆盖；诊断命令不下载、不训练。scripts.prepare_formal_split的--metadata-only只检查仓库内清单，不声称外部来源及smoke归属已核验，正式训练仍默认严格检查。scripts.export_source导出不含本机工件的源码ZIP，不自动上传。

默认测试24项，其中23通过、1项真实权重集成检查显式跳过；18个CLI模块--help和语法检查通过。另行显式开启真实权重合成图片网页检查，结果见本轮completion及integration记录。整理前后同一内存合成图片的39维CPU概率完全一致，无原始数据图片读取、训练或重新评价测试集。GitHub Actions工作流已添加，尚无远端实际执行结果；全新CPU环境安装及其他平台未验收。

以下为此前阶段历史记录。部分历史工具需要原reports工件；仅克隆代码不具备这些输入，不能因此修改来源锁或登记来冒充复现。

## 交付版启动环境修复 2026年10月27日

用户在teacher_demo_20261027_v1中遇到“Project Python was not found”。原因是交付ZIP不含虚拟环境，而旧启动器仅查找包内.venv，没有复用本机已有的原项目环境。

已修复启动器：优先包内.venv；包位于原项目deliverables/包名/下且上两级有原项目FINAL_MODEL.json时，回退到上两级原项目.venv。缺少两种环境时明确说明首次安装步骤在README，不自动安装或修改系统环境。原项目根目录入口仍优先自己的.venv。已更新原v1运行目录的启动文件和README及对应清单项，旧文件备份在reports/launcher_environment_fix_20261027，原v1 ZIP保留不覆盖，后续交付使用新v2 ZIP。

实际通过CMD执行用户原交付路径：包内Python不存在，成功复用原项目环境，HTTP健康响应200/ok；独立临时目录无环境时正确返回退出码1并提示README安装步骤。未训练、未重新评价测试集、未安装环境或修改模型。记录见reports/launcher_environment_fix_20261027/verification.json。本机网页已启动于http://127.0.0.1:8601，PID会随启动变化；未宣称在老师另一台电脑新安装并验收环境。

## 源码仓库与老师演示包 2026年10月27日

本轮目标仓库为 https://gitee.com/warr46ant/ml-project.git ，只读检查时没有已有远端分支。本地原仓库尚无提交，保留用户现有8601端口配置，准备main源码提交。数据集、权重、.venv、IDE/工具状态、缓存及本地交付ZIP由.gitignore排除；固定划分元数据保留，并在.gitattributes中禁止字节换行转换以保护清单锁。

README改为面向仓库阅读的简明入口，旧README和修改前状态保存在本地reports/release_20261027_gitee。docs/RESULTS.md及少量真实结果记录/曲线可在仓库直接查看；完整历史reports仍仅保留本地。

新增build_teacher_package.py、docs/TEACHER_HANDOFF.md、docs/TEACHER_README.md及test_portable_paths.py。老师包为本地 `deliverables/teacher_demo_20261027_v1.zip`，46,489,545字节；只复制一个原第3轮模型，另有必要网页代码、真实测试指标、课程资料和实验源码供审查，不含原始数据集和环境。原图、旧模型、旧报告不删除、不移动。交付包被Git忽略，需单独交给老师。

共用inference.py仅增加相对冻结记录路径的解析，原项目的绝对路径仍兼容。老师副本内模型/指标采用相对路径，原FINAL_MODEL.json和原权重字节不变，类别、预处理、概率及阈值不改。

15项清单、指标、资料和路径检查通过。两个独立CPU进程仅用同一张内存合成图片核对，原项目与老师副本的39维概率和Top3完全一致；副本AppTest初始页面及93.20%/92.24%真实指标正常。未在另一台机器新建环境安装验收，未重新训练或评价数据集。核对记录见docs/results/teacher_package_verification.json，完整本地日志在reports/release_20261027_gitee。

老师包实验源码目录用于审查；完整复训仍需原始数据和原历史输入工件。演示无需原始数据和划分清单。风险与严重程度、增强同源及真实场景泛化限制保持不变。提交/推送结果以Git提交历史及远端分支记录为准。

## 启动方式简化 2026年10月24日

新增根目录 [启动网页.bat](启动网页.bat)，演示时双击即可，自动定位项目并使用现有`.venv`，临时目录与缓存仍放在D盘，不需要手动输入环境变量或打开PyCharm。启动后手动访问http://127.0.0.1:8501；前台窗口Ctrl+C结束，若询问终止批处理则输入Y。保留仅本机监听及禁止自动打开浏览器设置，不安装依赖、不训练或修改模型。

实际通过CMD从其他工作目录启动，健康响应200/ok，监听PID3572。重复执行检测8501占用后提示先访问，不重复启动、不结束其他程序；复制启动文件到没有`.venv`的临时目录时，明确报错并返回退出码1。初次检查发现批处理中文提示编码和验收脚本的CMD引用问题，已改用ASCII提示、Windows CRLF并修复引用后重新检查通过；网页中文不变。没有声称用鼠标双击验收，验证方式为实际执行同一批处理。模型SHA256与冻结记录一致，未读取原图或评价测试集。

启动验证服务保持运行，实际日志和核对记录：[reports/one_click_20261024_121438_428207/verification.json](reports/one_click_20261024_121438_428207/verification.json)。更新README为双击优先，原PowerShell命令保留用于排查。旧报告和模型不覆盖。

## 最新完成阶段 课程交付整理

2026年10月19日用户要求继续下一阶段，完成课程报告整合、答辩演示资料与验收证据整理。本阶段没有训练、推理、重新测试、读取原始数据图片或切换网页模型；不发布网站。

新目录：[reports/course_submission_20261019_v1](reports/course_submission_20261019_v1/COURSE_REPORT_FINAL.html)。主报告为 `COURSE_REPORT_FINAL.html`（正文图内嵌，可打印）及 `COURSE_REPORT_FINAL.md`（可编辑），配套 `CARE_REFERENCE.md`、`DEMO_AND_DEFENSE.md`、`ACCEPTANCE.md`、`sources.json`、`delivery_manifest.json` 和 `completion.json`；[资料包](reports/course_submission_20261019_v1.zip)包含报告、图和必要证据副本，不包含模型及原始数据集。原报告和修改前根目录文档保留。

报告从实际文件读取并严格区分：原模型第3轮训练accuracy95.75%、验证accuracy93.30%/macro F1 92.20%、原最终测试accuracy93.20%/macro F1 92.24%，以及均衡模型验证accuracy94.92%/macro F1 94.21%。包括全部39类计数及原测试指标、低recall类别、按真实类比例说明的混淆对、两次真实曲线和原网页截图。没有将原测试成绩转给均衡模型；原已查看案例只用于报告。

新增 `build_submission_report.py`，只用标准库及现有类别/参考资料模块生成文档，不导入torch、不加载权重、不安装依赖。实际命令：

```powershell
& '.\.venv\Scripts\python.exe' .\build_submission_report.py --output-dir 'D:\PythonProject3\reports\course_submission_20261019_v1'
```

新目录拒绝覆盖；已有目录不要原地重跑。后续若重新整理须指定全新目录。环境临时目录及浏览器配置均放在D盘。报告核对30个来源副本及两模型SHA256，原冻结记录与模型不变。

本阶段实际用独立Chrome打开本地HTML：16个本地链接、两份39行附录、3张正文图片、关键指标与限制、目录锚点均通过；桌面1440宽和手机390宽无整页横向溢出，打印CSS生效；10张实际截图已逐张查看。此处是报告排版检查，不是今天重新点击Streamlit上传。专用浏览器已关闭，未修改正在使用的网页。验收记录见 `report_layout_verification.json`，截图保存在 `layout_qa` 供追溯，不放入交付资料包。

Word版未生成：documents技能要求通过工作区依赖解析器使用受管运行时，并用render_docx.py渲染逐页检查；本会话没有对应工具及受管渲染环境。本阶段交付HTML和Markdown，未安装全局或项目Word依赖，未伪称DOCX或PDF已验收。HTML可由浏览器打印/另存PDF，但本阶段未验证实际PDF分页或实体打印。今天Streamlit服务在线状态、真人复制粘贴与演示现场交互仍需按 `DEMO_AND_DEFENSE.md` 确认；旧PID不当作今日状态。

风险模块继续为未田间验证的类别参考规则，严重程度明确未评估；增强同源、真实拍摄泛化、真实严重程度与田间防效仍无新证据。课程是否接受该风险原型需由教师验收，不能把缺少专业数据的能力写成已完成。本阶段交付完毕后停止。

以下保留均衡实验和更早阶段记录。

## 上一阶段 独立均衡下采样训练对照

2026-10-18 用户要求“继续下一阶段”，按课程要求完成独立均衡训练对照。训练、重载、验证对照及报告已完成，本阶段停止，**网页仍使用原锁定模型**。

实际实验目录：`D:\PythonProject3\reports\balanced_head_20261018_200208_701758`；训练进程PID23624已退出（退出码0）。训练清单哈希及来源检查通过，39类每类582张，共22698张，全部是原正式训练集的子集；与原验证集9192张的路径/SHA256交集为0。RTX4060 Laptop GPU、batch_size32、num_workers0、seed42、lr0.001、Adam、CrossEntropyLoss，冻结主干及BN，仅训练新分类头；从原ImageNet缓存重新开始，不接续旧最佳模型。未安装依赖或下载数据/权重。

新入口仍为 `train_v1.py`，新增可选 `--balanced-manifest` 和独立输出目录参数，复用原训练循环；`balanced_manifest.py` 校验哈希锁、等量、原训练子集和验证隔离。5项合成清单边界检查及4项既有指标检查通过。实际命令、代码快照和设置保存在本实验目录；第1轮检查冻结与更新后放行，同进程、同优化器继续。

按预先约定最多10轮、patience3、验证loss最小选模，实际完成10轮后停止，最佳第10轮（第8/9轮回落，第10轮改善，未触发早停）。真实逐轮loss、accuracy、耗时、学习率见 [metrics.csv](reports/balanced_head_20261018_200208_701758/metrics.csv)、[完整报告](reports/balanced_head_20261018_200208_701758/COMPARISON_REPORT.md)、[实际训练曲线](reports/balanced_head_20261018_200208_701758/training_curves.png)。训练及验证墙钟1564.18秒，首轮核对等待216.07秒另计；两模型验证对照32.64秒。训练loss总体下降，验证中途波动后改善，不能仅凭第8/9轮断言最终模型明显过拟合，也不因最后一轮改善继续加轮数。

同一固定验证集（非最终测试）对照：

| 指标 | 原冻结模型（第3轮） | 新均衡模型（第10轮） |
| --- | --- | --- |
| Loss | 0.208861 | 0.166577 |
| Accuracy | 93.30%（8576/9192） | 94.92%（8725/9192） |
| Macro precision | 93.08% | 94.22% |
| Macro recall | 92.16% | 94.53% |
| Macro F1 | 92.20% | 94.21% |

accuracy提高1.62个百分点，macro F1提高2.00个百分点。独立进程PID46968重载1张预定验证图，39维输出、编号、预处理输入一致，logits及概率保存前后最大差为0；新旧重载验证accuracy均与各自原日志吻合。每轮120个主干状态张量、其中60个BN缓冲保持固定，主干无梯度、分类头更新。对照推理前后全部模型状态不变；从保存的9192条逐图记录复算混淆计数、全部39类指标及平均loss一致。详见 `validation_comparison/comparison.json`、`predictions.csv`、`per_class_comparison.csv`、`records_audit.json`。

新最佳模型完整路径：`D:\PythonProject3\reports\balanced_head_20261018_200208_701758\checkpoints\best_epoch_010.pt`。SHA256：`259d444771ccab1759abb91da92c9e4ea36dca35b6e16249dba60b5c1b23375f`。最后一轮另存 `last_epoch_010.pt`；本次最佳恰为最后轮，两文件内容哈希相同。验收总记录：[completion.json](reports/balanced_head_20261018_200208_701758/completion.json)。训练摘要中的reload_pending为生成当时状态，完成情况以completion和独立重载文件为准。

实际训练命令（环境变量和全部验收命令见 [commands.ps1](reports/balanced_head_20261018_200208_701758/commands.ps1)，不要原地重跑覆盖）：

```powershell
& '.\.venv\Scripts\python.exe' .\train_v1.py --balanced-manifest '.\reports\course_completion_20261018_193414_578564\report_v2\candidate_balanced_train_seed42.csv' --output-dir 'D:\PythonProject3\reports\balanced_head_20261018_200208_701758'
```

实验计划见 [EXPERIMENT_PLAN.md](reports/balanced_head_20261018_200208_701758/EXPERIMENT_PLAN.md)。本轮仅训练均衡视图、在原验证集对照，测试图片读取0张。原 `FINAL_MODEL.json` 及原第3轮检查点哈希前后不变，网页仍显示原模型的原最终测试成绩。下采样同时减少训练量和每轮更新次数，轮数也与原实验不同，不能把差异全部归因于均衡；验证集已用于选模，不是新独立最终评价；增强同源关系未排除，真实拍摄泛化未验证。图像真实严重程度和田间防效验证仍缺少数据，本轮不虚报这两项。

新增 `balanced_manifest.py`、`test_balanced_manifest.py`、`compare_balanced_validation.py`、`build_balanced_report.py`，只对 `train_v1.py` 增加均衡视图及记录选项，更新README与课程对照；未改网页预测模块、冻结记录、原图和原正式划分，不覆盖旧模型、训练日志或测试报告。

下方为此前各阶段的历史记录；其中“候选未训练”描述的是当时状态，本轮独立对照完成情况以上文为准。

更新日期：2026-10-18。**网页已实际启动在 http://127.0.0.1:8501，仅监听本机，实际监听 PID 41776（虚拟环境启动器 PID 40536），关闭自动打开浏览器。已完成真实 Chrome 交互和 AppTest 验收；未训练、调参、修改预处理、下载权重或重新评价测试集。** 本阶段开发到此停止，保留本地服务供访问，不发布网站。

## 本轮：可复制类别与课程决策参考、综合报告

用户提供了“项目3、基于图像数据的农作物病虫害识别与防治决策”的五项任务，已登记在 [COURSE_REQUIREMENTS.md](COURSE_REQUIREMENTS.md)。当前完成课程原型与资料交付，不虚报均衡训练、图像严重程度或田间应用能力。

阶段一：39类表格改为原生HTML文字，默认收起，可拖选中英文后Ctrl+C复制。中英文、原编号顺序和类型保持一致，动态文字经HTML转义。

阶段二：新增 `plant_care.py`，覆盖26类病害/虫害，健康与背景单独处理。按当前Top1展示类别规则参考风险及有来源的复核/管理建议，可点击下载单次Markdown报告。概率不参与高/中/低分级；等级不是图像严重程度或官方检疫等级；缺少严重程度标注，因此显示未评估。资料来源为农业部门与大学推广机构，保留核对日期和原文链接，未田间验证。

阶段三：`build_course_report.py` 复用已有清单、计数、日志与最终指标，生成综合报告、39类资料附录和均衡治理成果。当前报告：[report_v2/COURSE_REPORT.md](reports/course_completion_20261018_193414_578564/report_v2/COURSE_REPORT.md)，[39类资料](reports/course_completion_20261018_193414_578564/report_v2/CARE_REFERENCE.md)。初稿和更新前文档保留在新目录 `reports/course_completion_20261018_193414_578564/`，旧报告和原图不动。

均衡的真实状态：正式训练仍42890张，每类582～3855，最大/最小比6.624。候选等量视图seed42，每类582张共22698张，哈希不重复，全部为正式训练清单子集；独立哈希锁记录源清单与新视图。另提供N_train/(39×该类训练数)候选权重。均未用于当前冻结模型，未替换正式划分，验证/测试未重采样；若要求均衡训练对照，仍须另立实验，旧成绩不能改名。

4项边界测试覆盖39类资料、健康/背景、未知类别拒绝及概率不用于风险。AppTest与真实Chrome核对39类HTML中英文及顺序、原生文字选择、折叠；一张既定验证图的39维概率与旧参考一致；风险/严重程度说明正常，实际下载报告含正确类别、92.20%示例分数、模型哈希与来源；移除图片后旧风险/下载清除。未读取系统剪贴板，未声称做过人工粘贴验收。测试图片读取0张，无训练。见 [verification.json](reports/course_completion_20261018_193414_578564/verification.json)、新目录日志与截图。

仅改页面及文档，新增 `plant_care.py`、`test_plant_care.py`、`build_course_report.py`、`COURSE_REQUIREMENTS.md`；未新增依赖，未改共享推理、检查点或冻结记录。复用PID41776服务，当前信息见新目录 `server.json`，专用验收浏览器已关闭。旧服务在浏览器关闭时记录过WinError10054，之后健康检查与实际下载正常，不是模型或评价错误。

正式验证边界：没有图像严重程度模型；选择任务中的“风险等级”路线实现类别规则原型，规则一致性/田间防效未验证；均衡候选数据未训练；精准施药处方未实现。需要更强能力时，须补充标注、植保专家和独立实验，不自动扩大训练。

## 此前补充：中英文类别折叠栏

上传区域上方新增“支持识别的类别（中英文，共39类）”，默认收起，展开显示可滚动表格：中文名称、原始英文类别名、类型。类别来自 `FINAL_MODEL.json` 冻结映射，按原编号排序；39类完整保留，含26类病害/虫害、12类健康和1类背景，不把39类写成39种病害。

只改页面 `app.py` 与展示映射 `class_labels.py`，为两项葡萄类别补充中文名称并保留 Esca / Isariopsis 专有名。模型、编号、预处理与阈值均不改。AppTest 逐行核对完整39类中英文、顺序和类型；真实 Chrome 实际验证默认收起、点击展开及再次收起，并已查看截图。此次验收未加载模型权重、未读取数据集图片，无训练或评价。

本次开始时原服务已停止，已在原本机地址启动新版服务。记录与更新前文档快照：[reports/web_categories_20261018_192136_771289/REPORT.md](reports/web_categories_20261018_192136_771289/REPORT.md)。冻结文件 SHA256 与此前相同；专用验收浏览器已关闭。

## 本次补充：支持更多照片格式

新增 WebP、BMP、TIFF（.tif/.tiff）、静态 GIF、AVIF，以及 JPEG 的 .jpe/.jfif 后缀；保留 JPG/JPEG/PNG，共 11 个后缀。`inference.py` 定义统一映射，`app.py` 从同一映射生成上传选择列表，避免网页允许上传但预测函数不接受的情况。大小写后缀可用，真实格式校验、10 MiB/2000 万像素限制、Pillow 安全限制及单帧限制均保留。HEIC/HEIF、RAW 尚未支持。

未安装或升级依赖。只修改 `app.py`、`inference.py`、README 和本状态，新增 `test_image_formats.py`；不改模型、类别、预处理、概率或阈值。4 项检查通过：11 个后缀逐一解码为 RGB、伪装/损坏图片拒绝、动画 WebP/GIF/AVIF 与多页 TIFF 拒绝，以及所有后缀在官方 AppTest 中预览并实际调用冻结模型识别。仅使用内存生成的合成图片，不计算准确率、不读取真实数据集或测试图。

真实 Chrome 另验证新选择列表、WebP 上传、预览、点击识别与 Top3。旧服务在本次开始时已停止，已按原地址重新启动；实际监听 PID 和停止方法已更新。锁定模型 SHA256 仍为 `1c464873a01b10e1dce89211859a0a9bfe1448a765ecfaef4ed57089436c6b2f`。

新增记录：[reports/web_formats_20261017_205817_118815/REPORT.md](reports/web_formats_20261017_205817_118815/REPORT.md)，包含检查日志、浏览器记录、合成夹具、真实截图、服务记录和更新前文档快照。新增格式未逐一做真人拍摄效果评价，原有真实场景与增强同源局限保持不变。

## 本阶段交付与启动

新增 `app.py`、`inference.py`、`class_labels.py`、`predict_image.py`、`verify_web.py`、`requirements-web.txt`、`.streamlit/config.toml`；更新 README 和本状态。原 `smoke_utils.py` 模型构造和确定性预处理直接复用，未复制另一套变换，未修改训练和评价代码。

新结果目录：[reports/web_demo_20261017_123514_064143](reports/web_demo_20261017_123514_064143/REPORT.md)。更新前完整状态和 README 已保存为该目录的 `status_before_web.md`、`readme_before_web.md`，原模型、日志、测试报告均保留。

从项目根目录 PowerShell 启动的准确命令：

```powershell
Set-Location 'D:\PythonProject3'
$env:TEMP = 'D:\PythonProject3\.venv\smoke_tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:XDG_CACHE_HOME = $env:TEMP
& '.\.venv\Scripts\python.exe' -m streamlit run .\app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

本次服务仍在运行时直接打开地址，勿重复启动。前台终端按 Ctrl+C 停止；本次后台服务可用 `Stop-Process -Id 41776` 停止，后续启动 PID 会变化。当前服务配置和 stdout/stderr 见类别补充目录 `reports/web_categories_20261018_192136_771289/` 内的 `server.json`、`server.stdout.log`、`server.stderr.log`；首次交付与格式补充的旧日志继续保留。

## 网页模型、指标与依赖

仍唯一使用 `FINAL_MODEL.json` 指定的 `D:\PythonProject3\reports\v1_head_20261017_102902_302541\checkpoints\best_epoch_003.pt`，实际 SHA256 核对为 `1c464873a01b10e1dce89211859a0a9bfe1448a765ecfaef4ed57089436c6b2f`，验收结束仍相同。加载完整检查点并核对类别、预处理、划分标识及轮次；不读取其中记录的原图/划分路径，也不下载权重。

页面保留全部 39 类及原始英文，明确区分健康、病害/虫害、背景。中文名称只用于展示，39类均有中文辅助名，葡萄两项保留 Esca / Isariopsis 专有名用于对照。背景类不等于通用非叶片检测，未设置概率拒识阈值。上传图片仅内存处理，格式与后缀都核验，限制 10 MiB、2000 万像素，Pillow 默认安全限制仍开启。

折叠区从已有 `reports/final_test_20261017_121954_264063/metrics.json` 读取固定测试集 **accuracy=93.20%、macro F1=92.24%**，标为数据集实验成绩；该文件不是此次重新计算的结果。增强同源关系未完全排除、真实拍摄效果尚未验证的限制在页面保留。概率不代表诊断可靠性。

环境仍是项目 `.venv`：Python 3.14.6、torch 2.13.0+cu130、torchvision 0.28.0+cu130、Pillow 12.3.0；仅补充 Streamlit **1.66.0** 及其必要依赖。官方文档支持 Python 3.14，实际通过 Windows cp314 / 通用 wheel 安装，无重装训练环境、源码编译或驱动修改。`pip check` 通过；实际完整新增依赖、Requires-Python 和来源见 `environment.json`、`pip_install.json`，直接依赖固定在 `requirements-web.txt`。安装缓存/临时目录、本次浏览器配置和日志均在 D 盘。

## 实际验收结果

| 验收项 | 结果及证据 |
| --- | --- |
| 固定验证图与旧逻辑一致 | `Apple___Apple_scab/image (1).JPG`；输入张量哈希相同，网页 GPU 的完整 39 类概率与已有参考差为 0；Top1 苹果黑星病 92.196393% |
| 真实浏览器交互 | Chrome 155：上传、预览、点击识别、Top3 条形图、连续识别、更换清旧结果、删除清旧结果、损坏/伪装后缀提示、展开真实指标均成功 |
| UI 状态和缓存 | AppTest 确认连续识别只加载模型一次，旧结果不残留；模型 eval、全部参数 requires_grad=False、inference_mode |
| CPU 加载/预测 | 实际成功，输出 `[1,39]`，Top3 编号与 GPU 一致；概率最大绝对差 0.00005196035，跨设备容差 rtol=0.001、atol=0.0001；未修改概率 |
| 异常处理 | 真实损坏/伪装图片、超过文件大小、超过像素上限均拒绝；用临时记录和文件检查真实模型缺失及哈希不符，实际模型未动 |
| 独立于整套数据 | 验收先把一张验证图读入内存，随后拦截数据集与划分文件读取；正常页面和推理仍通过 |
| 页面检查 | 已查看真实桌面和 390 像素窄屏截图，无横向溢出；窄屏不是物理手机实测 |
| 数据与模型保护 | 测试图片读取 0 张，无训练/重新评价；冻结模型 SHA256 不变 |

本次 AppTest 最终检查约 2.61 秒（不含模块导入），这不是单次网页延迟或性能基准。[详细记录](reports/web_demo_20261017_123514_064143/verification.json)、[真实浏览器记录](reports/web_demo_20261017_123514_064143/browser_verification.json)、[真实缺失文件补充检查](reports/web_demo_20261017_123514_064143/additional_checks.json)。初次验收采用过严的 CPU/GPU 同设备容差，后改为单独记录跨设备误差；随后修正 AppTest 图片属性、浏览器删除按钮选择器及页面刷新等待，均只修改验收脚本，未改模型或预测值。尝试日志保留。日志中的 SHA256 错误是故意注入的错误模型测试，不是锁定模型损坏。

尚未验证：人工系统文件对话框、拖拽动作、其他浏览器、物理手机、多用户并发、全部格式边界、真实拍摄泛化和概率校准。错误路径包含 AppTest 故障注入，不声称在真实浏览器中实际删除过锁定模型。无剩余程序阻塞。

操作说明、代码作用、浏览器→Python→模型流程及三个理解检查问题见 [README](README.md)。本阶段使用 Sites 技能的本地预览分支，仅保留用户指定的本机 Streamlit 服务，专用验收浏览器已关闭，未调用网站发布。

---

## 上一阶段历史记录：最终测试与错误分析完成

以下是上阶段原记录，时间描述仅适用于上阶段；新的网页交付以上文为准。

更新日期：2026-10-17。**已完成 9,192 张保留测试图片的实际评价，记录和指标复核通过，模型未更新。完成后停止，不训练、微调、调参、更换模型或制作网页。**

## 本次真实最终测试结果

| 指标 | 实际结果 |
| --- | ---: |
| 平均测试 CrossEntropyLoss | 0.209433 |
| Accuracy | **93.20%** |
| 正确 / 总数 | **8567 / 9192** |
| 误判 | 625 |
| Macro precision | **93.12%** |
| Macro recall | **92.17%** |
| Macro F1 | **92.24%** |
| 推理主循环耗时 | 58.143 秒 |

测试耗时包含读取、逐图 SHA256 核对、确定性预处理、GPU 计算及逐图记录写入，不含初始模块导入和后续绘图。固定 39 类均参与指标计算，没有隐藏低分或没有预测样本的类别；本次没有预测数为 0 的类别。

## 区分训练、选模验证和最终测试

- 第 3 轮训练过程：loss=0.145097、accuracy=95.75%，是含随机增强、参数逐步更新过程中的训练指标。
- 第 3 轮验证：loss=0.208861、accuracy=93.30%，用于选择模型；完整训练在第 6 轮按 patience=3 早停。
- **本次最终保留测试：loss=0.209433、accuracy=93.20%、macro F1=92.24%。** 模型方案先确定，之后才读取保留测试图片。

验证与测试准确率数值接近，不能据此证明增强来源独立或真实场景泛化能力。此次分数不会被用于自动改模型、阈值、预处理或追加训练。

## 模型对象、划分与冻结核验

实际最佳模型路径来自训练完成验收文件指向的训练汇总，不靠遍历文件名选择，也没有加载其他实验模型。当前唯一锁定对象：

`D:\PythonProject3\reports\v1_head_20261017_102902_302541\checkpoints\best_epoch_003.pt`

文件 SHA256：

`1c464873a01b10e1dce89211859a0a9bfe1448a765ecfaef4ed57089436c6b2f`

供后续网页读取的冻结记录：[FINAL_MODEL.json](FINAL_MODEL.json)，包含完整路径、SHA256、39 类映射、模型配置、确定性预处理、划分标识和评价位置。本轮没有制作网页。

固定划分：`splits/formal_sha256_seed42_v1`；标识 `553a89fd302dd00cf3abc2e8554151a66a053dcc0edec0d9fc4ee82159a4f8d8`。

训练 42,890、验证 9,192、测试 9,192，总计 61,274 个唯一 SHA256 代表，均保留 39 类。模型内部的划分锁、映射、轮次、配置与磁盘清单一致；smoke 的 1,170 个哈希仍只在训练集，跨集合已知路径及 SHA256 交集为 0。没有重建或更换划分。测试清单 SHA256 为 `14756385e32ee572fe47cbca68e9000bcdb4e8200078327fb30a884aa357e86e`。

使用现有项目 .venv（Python 3.14.6，torch 2.13.0+cu130、torchvision 0.28.0+cu130），eval + inference_mode，batch_size=32、num_workers=0、shuffle=False；直接使用检查点中的验证预处理，未添加随机增强。全部 **122 个模型状态张量**（含分类头和 BatchNorm）以及检查点文件哈希在测试前后完全相同，没有梯度或参数更新。所有测试图片当前字节均与固定清单 SHA256 核验一致，原图未修改。

## 错误分析，不隐藏弱项

recall 最低的 8 类如下；完整 39 类 precision、recall、F1、support 和预测数均在报告及 CSV 中。

| 类别 | recall | support | 漏识别数 |
| --- | ---: | ---: | ---: |
| Potato___Late_blight | 54.67% | 150 | 68 |
| Tomato___Spider_mites Two-spotted_spider_mite | 60.56% | 251 | 99 |
| Tomato___Target_Spot | 71.56% | 211 | 60 |
| Tomato___Early_blight | 78.00% | 150 | 33 |
| Tomato___Leaf_Mold | 79.33% | 150 | 31 |
| Tomato___Septoria_leaf_spot | 83.08% | 266 | 45 |
| Tomato___Bacterial_spot | 89.03% | 319 | 35 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 89.33% | 150 | 16 |

最常见的有方向错分对同时给出数量和真实类内比例：

| 真实 → 预测 | 错分数 / 真实类support | 占该真实类 |
| --- | ---: | ---: |
| Tomato___Spider_mites Two-spotted_spider_mite → Tomato___healthy | 35/251 | 13.94% |
| Tomato___Spider_mites Two-spotted_spider_mite → Tomato___Target_Spot | 33/251 | 13.15% |
| Potato___Late_blight → Tomato___Late_blight | 32/150 | 21.33% |
| Tomato___Target_Spot → Tomato___healthy | 23/211 | 10.90% |
| Corn___Cercospora_leaf_spot Gray_leaf_spot → Corn___Northern_Leaf_Blight | 16/150 | 10.67% |

马铃薯晚疫病→番茄晚疫病虽为 32 张，少于红蜘蛛→健康番茄的 35 张，但类内比例为 21.33%，高于后者 13.94%，不能只看数量。马铃薯晚疫病 precision=100% 也不代表该类识别充分：只找回 82/150 张，漏识别 68 张。

已固定种子 42，分别从按路径排序的全部 625 个误判和 8,567 个正确记录中均匀无放回抽取 12 张，未按置信度、外观或结果好坏筛选。案例来源和抽样规则已保存。已查看的测试图只作为报告材料，不再作为新的独立评测材料。

误判图中可观察到高置信错误、叶片特征不明显等现象，见 [人工观察记录](reports/final_test_20261017_121954_264063/VISUAL_OBSERVATIONS.md)。视觉相似、光照/背景、裁剪和冻结特征能力等只能作为原因假设，未做对照或专家复标，不能单凭图片断言标签错误。没有把模型表现不理想当作实现错误。

## 输出文件与实际命令

新结果目录：`D:\\PythonProject3\\reports\\final_test_20261017_121954_264063`，没有覆盖训练日志或旧报告。

- [课程报告用评价摘要及完整分析](reports/final_test_20261017_121954_264063/REPORT.md)
- [总体及全部类别指标 JSON](reports/final_test_20261017_121954_264063/metrics.json)、[逐类指标 CSV](reports/final_test_20261017_121954_264063/per_class_metrics.csv)
- [原始混淆计数 CSV](reports/final_test_20261017_121954_264063/confusion_counts.csv)、[按真实类归一化 CSV](reports/final_test_20261017_121954_264063/confusion_row_normalized.csv)
- [原始计数图](reports/final_test_20261017_121954_264063/confusion_counts.png)、[归一化混淆矩阵图](reports/final_test_20261017_121954_264063/confusion_row_normalized.png)
- [逐图真实/预测及 Top3 CSV](reports/final_test_20261017_121954_264063/predictions.csv)、[全部 39 维分数及逐图 loss JSONL](reports/final_test_20261017_121954_264063/predictions.jsonl)
- [完整错分类别对 CSV](reports/final_test_20261017_121954_264063/confusion_pairs.csv)、[低 recall 和错分排名](reports/final_test_20261017_121954_264063/error_analysis.json)
- [固定种子误判图](reports/final_test_20261017_121954_264063/errors_seed42.jpg)、[固定种子正确图](reports/final_test_20261017_121954_264063/correct_seed42.jpg)、[案例来源](reports/final_test_20261017_121954_264063/report_case_sources.csv)
- [复核记录](reports/final_test_20261017_121954_264063/audit.json)、[最终完成记录](reports/final_test_20261017_121954_264063/completion.json)、[实际评价配置](reports/final_test_20261017_121954_264063/evaluation_config.json)、[运行日志](reports/final_test_20261017_121954_264063/evaluation.log)

实际执行命令（留作记录，不应日常重复运行）：

```powershell
& 'D:\PythonProject3\.venv\Scripts\python.exe' 'D:\PythonProject3\evaluate_final.py' --training-run 'D:\PythonProject3\reports\v1_head_20261017_102902_302541'
```

新增 `evaluate_final.py`、`evaluation_metrics.py`、`evaluation_outputs.py` 和公式单元测试；复用已有模型构造、RGB 读取、确定性预处理和划分校验。没有新增依赖或修改训练代码。

4 项人工计数单元测试覆盖完全正确、无预测样本、矩阵行列定义和空评价。全量预测记录已从文件重新读取，重建混淆矩阵与指标一致，并从 logits 复核逐图交叉熵；这一步没有再次运行模型。CSV 也独立核对了 9,192 行、8,567 正确、39 类 macro F1。未发现评价实现错误，未重新评价；此前语法检查遇到的缓存写权限问题发生在读取测试图片之前，已通过执行权限处理，与模型或评价数值无关。

根目录 `final_test_registry.json` 登记了本次评价，防止静默重复测试。只有真实实现错误且明确记录原因时，脚本才允许以同一模型、同一划分另存重评；不能用表现不理想作为理由。

## 初学者如何理解结果

Accuracy 把所有图片放在一起算正确比例，样本多的类别权重更大；macro F1 先分别计算每类 F1，再对 39 类等权平均，因此更能暴露较小类别的弱项。本次 93.20% accuracy 不代表每类都达到 93.20%。

混淆矩阵行是真实类、列是预测类；对角线是正确，非对角线是错分。按行归一化可比较类别规模不同的混淆，对角线的比例就是该类 recall；图上的编号和右侧类别图例对应。

反复看测试分数再调参会把测试信息带入选模，使分数逐渐偏向这份数据，失去独立评价意义。此轮模型方案已经确定，测试只用于最终报告，不据此重新优化。

**本划分已排除已知SHA256字节重复，但未排除增强同源关系，测试成绩可能偏乐观，不能直接代表独立叶片或真实拍摄场景的泛化能力。**

尚未验证增强同源独立性、独立真实叶片、真实拍摄场景、域外图片识别、概率校准和实地诊断能力。当前无剩余程序运行阻塞，评价局限保留。模型、阈值和预处理保持锁定，本阶段到此停止。

上一阶段完整状态已保存到 [status_before_final_test.md](reports/final_test_20261017_121954_264063/status_before_final_test.md)，训练阶段原始报告、日志和模型均保留。
