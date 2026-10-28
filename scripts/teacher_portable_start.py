"""老师离线包入口；本文件在交付包根目录作为portable_start.py运行。"""

import argparse
import hashlib
import io
import json
import os
import socket
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prepare():
    os.chdir(ROOT)
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR", "XDG_CACHE_HOME", "LEAF_RUNTIME_DIR"):
        os.environ[key] = str(runtime)
    os.environ["LEAF_MODEL_RECORD"] = str(ROOT / "FINAL_MODEL.json")
    # 课程演示默认CPU，既不要求NVIDIA显卡，也不改变模型或预处理。
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    return runtime


def verify(*, predict=False):
    from inference import FROZEN_SHA256, load_predictor, read_experiment_metrics, read_final_record, resolve_record_file
    import torch
    import torchvision
    import streamlit
    from PIL import Image
    record = read_final_record(ROOT / "FINAL_MODEL.json")
    checkpoint = resolve_record_file(record["checkpoint"], ROOT / "FINAL_MODEL.json")
    with checkpoint.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != FROZEN_SHA256:
        raise ValueError("模型文件校验失败，请重新完整解压交付包。")
    metrics, _ = read_experiment_metrics(ROOT / "FINAL_MODEL.json")
    bundled = ROOT / "runtime"
    if not Path(sys.executable).resolve().is_relative_to(bundled):
        raise RuntimeError("请使用包内runtime/python.exe启动，不能使用外部环境。")
    for module in (torch, torchvision, streamlit):
        if not Path(module.__file__).resolve().is_relative_to(bundled):
            raise RuntimeError(f"依赖未从包内加载：{module.__name__}")
    result = {"status": "passed", "python": sys.version.split()[0], "torch": torch.__version__,
              "torchvision": torchvision.__version__, "streamlit": streamlit.__version__,
              "model_sha256": actual, "classes": len(record["class_to_idx"]),
              "device": "cpu", "bundled_dependencies": True,
              "dataset_accuracy": metrics["accuracy"], "dataset_macro_f1": metrics["macro_f1"]}
    if predict:
        predictor = load_predictor(ROOT / "FINAL_MODEL.json", "cpu")
        with Image.new("RGB", (300, 240), (61, 147, 73)) as image:
            prediction = predictor.predict(image)
        if prediction["output_shape"] != [1, 39]:
            raise ValueError("模型输出维度不正确")
        result["synthetic_prediction"] = prediction
        result["note"] = "合成图片仅检查运行，不代表病害识别效果；未评价测试集。"
    return result


def choose_port(preferred):
    for port in range(preferred, min(preferred + 10, 65536)):
        with socket.socket() as connection:
            try:
                connection.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("连续10个本地端口被占用，请关闭旧演示窗口，或用--port指定其他端口。")


def open_when_ready(url):
    for _ in range(120):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(url + "/_stcore/health", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="检查包内依赖、模型及CPU单图推理后退出")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8601, choices=range(1024, 65526), metavar="PORT")
    args = parser.parse_args()
    runtime = prepare()
    try:
        started = time.perf_counter()
        result = verify(predict=args.check)
        result["check_seconds"] = time.perf_counter() - started
        if args.check:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        port = choose_port(args.port)
        url = f"http://127.0.0.1:{port}"
        print(f"\nPlant Leaf Demo | CPU | 39 classes\nOpen: {url}\nStop: Ctrl+C\n", flush=True)
        with (runtime / "startup.log").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"url": url, "startup_check": result}, ensure_ascii=False) + "\n")
        if not args.no_browser:
            threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
        from streamlit.web.cli import main as streamlit_main
        sys.argv = ["streamlit", "run", str(ROOT / "app.py"), "--server.address", "127.0.0.1",
                    "--server.port", str(port), "--server.headless", "true", "--browser.gatherUsageStats", "false",
                    "--server.fileWatcherType", "none"]
        streamlit_main()
    except Exception:
        message = traceback.format_exc()
        with (runtime / "startup.log").open("a", encoding="utf-8") as file:
            file.write(message)
        print(message, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
