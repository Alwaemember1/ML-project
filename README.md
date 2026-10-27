# 植物叶片病害识别与防治参考系统

PyTorch + ResNet18迁移学习 + Streamlit中文单图网页。支持39类：26类病害/虫害、12类健康、1类无叶片背景。

上传图片后展示Top1、Top3分数、中英文类别及按类别匹配的参考风险和有来源的管理资料。上传默认只在内存处理；更换或移除图片会清除旧结果。

## 快速启动

在已有完整环境和模型的项目中，双击 **启动网页.bat**，或在PowerShell运行：

```powershell
& 'D:\PythonProject3\启动网页.bat'
```

当前地址：**http://127.0.0.1:8601**。仅监听本机，不自动打开浏览器。保持前台窗口运行，Ctrl+C停止；如询问终止批处理则输入Y。端口已有服务时会提示先访问，不重复启动、不结束其他程序。

旧报告中的8501属于历史端口，当前以批处理输出和`.streamlit/config.toml`为准。

## 从仓库获取后需要什么

**仓库只包含源码、固定划分元数据、依赖和少量真实结果，不包含模型权重、原始数据或.venv。克隆后不能直接当成已经部署好的程序。**

老师演示交付包已另行整理在原项目的`deliverables/`，包含一个冻结模型、真实指标、课程报告和必要源码；未上传Git。拿到交付包后，请按包内README建立环境并启动。不要使用不明权重替代锁定模型。

已验证环境：Python3.14.6、torch2.13.0+cu130、torchvision0.28.0+cu130、Pillow12.3.0、Streamlit1.66.0。依赖记录为`requirements-training.txt`和`requirements-web.txt`，原始Pillow依赖在`requirements.txt`；它们需要分步安装，不要只安装Pillow就启动网页。GPU优先，支持CPU推理。

## 实验结果

| 对象与阶段 | Accuracy | Macro F1 |
| --- | --- | --- |
| 原模型验证 | 93.30% | 92.20% |
| 原冻结模型最终测试 | 93.20% | 92.24% |
| 均衡实验模型验证 | 94.92% | 94.21% |

网页仍使用原第3轮模型，测试8567/9192正确。均衡模型未切换到网页，94.92%不是它的最终测试成绩。真实记录、逐类指标和曲线见[实验结果](docs/RESULTS.md)。

## 文件入口

| 文件 | 作用 |
| --- | --- |
| app.py | 网页、上传预览、按钮、结果展示和清理 |
| inference.py | 模型身份与SHA256核对、图片安全解码、共用预测 |
| smoke_utils.py | 共用ResNet18构造与预处理，网页也依赖它 |
| class_labels.py | 明确的中文名和健康/病害虫害/背景类型 |
| plant_care.py | 按类别匹配参考风险、建议、来源和下载报告 |
| predict_image.py | 命令行单图推理 |
| check_dataset.py、pretrain_data_audit.py | 可读性与字节去重检查 |
| prepare_formal_split.py、splits/ | 固定划分与锁定元数据 |
| train_v1.py、train_smoke.py | 共用训练流程和冻结检查；不要按smoke名字直接删文件 |
| compare_balanced_validation.py | 仅验证集的新旧模型对照 |
| build_teacher_package.py | 生成独立老师演示包，拒绝覆盖原输出 |

## 交付与文档

- [老师版保留与排除建议](docs/TEACHER_HANDOFF.md)
- [老师包使用说明模板](docs/TEACHER_README.md)
- [课程要求与实际边界](COURSE_REQUIREMENTS.md)
- [实验结果和薄弱类别](docs/RESULTS.md)
- `PROJECT_STATUS.md`保留本地开发历史，其中reports链接需要原项目的历史工件，不随Git全部发布。

在原完整项目中生成新交付目录：

```powershell
& '.\.venv\Scripts\python.exe' .\build_teacher_package.py --output-dir '.\deliverables\teacher_demo_new'
```

此命令复用现有模型和报告，不训练、不重新测试，也不删除原件。首次运行需要原项目reports中的真实成果；仅克隆源码缺少这些输入时不能直接打包。老师包中包含实验源码供审查，完整重训仍需原始数据、预训练缓存和历史输入工件。

## 能力边界

- 只识别固定39类，背景类不代表通用域外或非叶片检测。
- 分类概率是模型输出分数，不等于诊断可靠性或病情严重程度。
- 风险是按类别设置的参考规则，真实轻中重、田间风险和防效尚未验证；资料不作为直接施药处方。
- 已知SHA256字节重复已排除，增强同源关系未排除，测试成绩可能偏乐观，真实拍摄泛化未验证。
- 原图、旧报告和检查点留在本地；Git忽略它们并不代表已经删除。第三方数据和依赖各自的授权不因本仓库提交而改变。
