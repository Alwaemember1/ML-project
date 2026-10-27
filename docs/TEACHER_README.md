# 植物叶片病害识别系统老师演示版

本目录为独立演示交付副本。网页使用原第3轮ResNet18冻结模型，39类，原固定测试集accuracy93.20%、macro F1 92.24%。均衡实验94.92%是验证成绩，该模型未用于网页。

## 启动

双击 `启动网页.bat`，浏览器访问 **http://127.0.0.1:8601**。启动脚本优先使用本目录 `.venv\Scripts\python.exe`；交付包仍位于原项目 `deliverables/包名/` 下时，也可复用上两级原项目的 `.venv`，不用重复安装。

**交付ZIP没有附带Python环境。**换到没有原项目环境的新电脑，首次仍需按下方命令创建本目录 `.venv` 并安装依赖，之后才能双击启动。把整个包单独移动到其他位置时，原项目环境回退不一定存在；此时按首次安装处理。不要只移动启动文件。

首次建立环境可在本目录PowerShell中执行下列命令。本项目已验证Python3.14.6及依赖文件中的固定版本；其他版本未在本交付中验证。需要联网安装依赖，模型已包含，不重新下载预训练权重。

```powershell
py -3.14 -m venv .venv
$env:TEMP = "$PWD\.venv\smoke_tmp"
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:PIP_CACHE_DIR = "$PWD\.venv\pip_cache"
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-training.txt
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-web.txt
& '.\启动网页.bat'
```

当前依赖使用已验证的CUDA构建，优先GPU，CUDA不可用时预测代码可自动选择CPU；CPU推理流程在项目环境中验证过。首次安装可能需要较大的下载空间，建议答辩前在目标机器完成环境准备。本轮没有在另一台电脑重新安装并验收环境。

前台窗口Ctrl+C停止，若询问是否终止批处理则输入Y。端口已有服务时先确认地址，不重复启动或结束其他程序。

## 文件说明

- 根目录Python文件：网页、共用推理、类别展示和参考资料。
- `models/final_resnet18.pt`：唯一演示模型，SHA256为 `1c464873a01b10e1dce89211859a0a9bfe1448a765ecfaef4ed57089436c6b2f`。
- `FINAL_MODEL.json`：交付副本，模型与指标使用相对路径；类别、预处理、轮次、划分标识及模型SHA与原记录一致。
- `results/metrics.json`：原最终测试的真实指标，页面仅展示，不重新评价数据。
- `docs/COURSE_REPORT_FINAL.html`：整合课程报告，浏览器可打印；`docs/`内保留配套来源与证据。
- `experiment_source/`：完整实验源码和固定划分元数据供审查；复训还需要原始数据及历史工件，不能直接把演示包当成完整训练环境。
- `DELIVERY_MANIFEST.json`：文件清单和SHA256，不包含清单自身哈希。

旧报告和演示说明中的8501为当时的历史端口，本交付当前端口统一以8601为准。报告旧截图没有重新生成或篡改日期。原始数据集、.venv、浏览器缓存和其他实验模型未随包复制；正常上传预测无需原始数据集。

## 能力边界

39类包括26类病害虫害、12类健康和1类背景，并非39种病害。参考风险按类别规则匹配，没有从图片测量真实严重程度，不是经过田间验证的风险模型。来源资料提供管理参考，不是可直接执行的施药处方。

已排除已知SHA256字节重复，但未排除增强同源关系，实验成绩可能偏乐观，不能直接代表真实拍摄泛化。用户图片默认只在内存处理，不自动留存，不设置拒识阈值。
