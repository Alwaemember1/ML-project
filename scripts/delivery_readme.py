"""由统一README的许可说明生成交付包运行指引，不依赖旧老师文档。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def delivery_readme(offline=False):
    if offline:
        startup = """完整解压到Windows x64电脑后，先双击 `启动前检查.bat`，再双击 `启动网页.bat`。
本包已附独立Python运行环境、依赖和原冻结模型，无需再次安装，默认CPU。
首次启动请等待依赖加载；浏览器访问 http://127.0.0.1:8601 。
启动失败时查看 `.runtime/startup.log`，不要只移动启动文件。
"""
    else:
        startup = r"""本包附原冻结模型，不附Python或已安装依赖。Windows x64安装Python3.14后，在包根目录PowerShell执行：

```powershell
py -3.14 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-cpu.txt
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-web.txt
& '.\启动网页.bat'
```

浏览器访问 http://127.0.0.1:8601 ，保留启动窗口；按Ctrl+C停止。
"""
    source = (ROOT / "README.md").read_text(encoding="utf-8")
    license_notice = source.split("## 局限与许可", 1)[1]
    return """# 农作物病虫害识别系统交付包

本包用于课程项目3的单图识别演示。模型为原第3轮ResNet18，39类，原最终测试准确率93.20%、Macro F1为92.24%。均衡实验94.92%为验证成绩，该模型未用于网页。

## 启动

""" + startup + """
## 操作

上传图片后点击“开始识别”，查看Top3、参考风险和管理资料，按需下载本次报告。
图片不超过10MiB或2000万像素，后缀须与真实格式一致，只支持单帧静态图。

模型及指标由包内FINAL_MODEL.json以相对路径定位，正常预测不读取训练数据。
完整源码与项目说明在experiment_source内，课程综合报告在docs内。
参考风险按类别规则匹配，不是图像严重程度或田间风险预测，不作为自动施药依据。

## 局限与许可
""" + license_notice
