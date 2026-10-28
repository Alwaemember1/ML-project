"""在老师包的独立解释器内验收；只用合成图片，不打开原始数据集。"""

import argparse
import io
import json
import os
import socket
import sys
import time
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import portable_start
    portable_start.prepare()
    root = portable_start.ROOT
    began = time.perf_counter()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    original_connect = socket.socket.connect

    def local_connections_only(connection, address):
        # Windows asyncio使用本地socketpair唤醒线程；允许回环，不允许外网。
        if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError(f"验收期间禁止非回环连接：{address}")
        return original_connect(connection, address)

    with patch.object(socket.socket, "connect", local_connections_only):
        check = portable_start.verify(predict=True)
        actual = check.pop("synthetic_prediction")
        difference = max(abs(a - b) for a, b in zip(actual["probabilities"], reference["probabilities"]))
        assert len(actual["probabilities"]) == 39 and difference <= 1e-7, difference
        assert all(Path(path).is_relative_to(root) for path in sys.path if path), sys.path
        from PIL import Image
        from inference import InferenceError, decode_image, load_predictor, model_cache_key
        from streamlit.testing.v1 import AppTest
        buffers = []
        for format_name, color in (("PNG", (61, 147, 73)), ("WEBP", (140, 60, 30))):
            output = io.BytesIO()
            with Image.new("RGB", (300, 240), color) as image:
                image.save(output, format=format_name, **({"lossless": True} if format_name == "WEBP" else {}))
            buffers.append(output.getvalue())
        app = AppTest.from_file(str(root / "app.py"), default_timeout=90).run()
        assert not app.exception and not app.error
        assert [metric.value for metric in app.metric] == ["93.20%", "92.24%"]
        for name, data in zip(("runtime_check.png", "replacement.webp"), buffers):
            app.file_uploader[0].upload(name, data).run()
            assert not app.exception and not app.error and len(app.image) == 1
            assert "prediction" not in app.session_state
            app.button[0].click().run()
            assert not app.exception and not app.error
            prediction = app.session_state["prediction"]
            assert prediction["device"] == "cpu" and prediction["output_shape"] == [1, 39]
            if name.endswith("png"):
                assert prediction["probabilities"] == actual["probabilities"]
        app.file_uploader[0].clear().run()
        assert "prediction" not in app.session_state and len(app.image) == 0
        app.file_uploader[0].upload("broken.png", b"not an image").run()
        assert app.error and "prediction" not in app.session_state and not app.exception
        app.file_uploader[0].upload("disguised.webp", buffers[0]).run()
        assert app.error and "prediction" not in app.session_state and not app.exception
        # 错误处理使用临时记录，不移动或修改真实模型。
        temporary = root / ".runtime/invalid_record.json"
        record = json.loads((root / "FINAL_MODEL.json").read_text(encoding="utf-8"))
        record["checkpoint"] = "missing.pt"
        temporary.write_text(json.dumps(record), encoding="utf-8")
        try:
            model_cache_key(temporary)
        except InferenceError:
            pass
        else:
            raise AssertionError("缺失模型未报错")
        fake = root / ".runtime/wrong_model.pt"
        fake.write_bytes(b"not the frozen model")
        record["checkpoint"] = str(fake)
        temporary.write_text(json.dumps(record), encoding="utf-8")
        try:
            load_predictor(temporary, "cpu")
        except InferenceError:
            pass
        else:
            raise AssertionError("模型哈希不符未报错")
    result = {"status": "passed", "startup_check": check,
              "all39_cpu_probability_max_abs_difference": difference,
              "sys_path_inside_package_only": True, "outbound_connections_blocked_during_check": True,
              "apptest": {"initial_page": True, "real_metrics": True, "png_preview_and_prediction": True,
                          "webp_replacement_clears_old_prediction": True, "repeat_prediction": True,
                          "remove_image_clears_results": True, "corrupt_file": True, "disguised_suffix": True},
              "missing_model_and_wrong_hash_rejected": True,
              "raw_dataset_images_opened": 0, "test_set_reevaluated": False,
              "model_retrained_or_changed": False, "seconds": time.perf_counter() - began,
              "limits": ["不是鼠标点击验收", "未在另一台实体Windows电脑验收", "默认CPU，未在此包验收GPU推理"]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
