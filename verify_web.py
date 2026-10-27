"""网页验收：只读一张既定验证图；不读取测试图，不重新评价数据集。

使用真实冻结模型及 Streamlit AppTest。故障只注入临时记录/文件或 mock，
绝不改动实际冻结记录、原始图片、模型和旧报告。
"""

import argparse
import builtins
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

from inference import (FINAL_RECORD, MAX_FILE_BYTES, ROOT, InferenceError, decode_image,
                       load_predictor, model_cache_key, read_final_record)
from smoke_utils import image_tensor, sha256_file, tensor_digest

import streamlit as st
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "verification.json").exists():
        raise FileExistsError("拒绝覆盖验收记录")
    started = time.perf_counter()
    record = read_final_record()
    training = Path(record["training_run"])
    reference = json.loads((training / "best_validation_reference.json").read_text(encoding="utf-8"))
    assert reference["source_split"] == "validation"
    image_path = ROOT / "Plant_leave_diseases_dataset_with_augmentation" / reference["relative_path"]
    data = image_path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == reference["image_sha256"]
    report = {"purpose": "web_demo_verification", "test_images_opened": 0,
              "validation_image": str(image_path), "reference_source": "existing validation reference",
              "checks": {}, "versions": {"python": sys.version, "streamlit": st.__version__, "torch": torch.__version__}}
    checks = report["checks"]
    original_open, original_io_open = builtins.open, io.open

    def guarded_open(original):
        def checked(file, *arguments, **keywords):
            if isinstance(file, (str, bytes, Path)):
                text = str(file).replace("\\", "/").lower()
                if "plant_leave_diseases_dataset_with_augmentation" in text or "/splits/" in text:
                    raise AssertionError(f"正常网页/推理不应访问数据集或划分：{file}")
            return original(file, *arguments, **keywords)
        return checked

    with patch("builtins.open", guarded_open(original_open)), patch("io.open", guarded_open(original_io_open)):
        cpu = load_predictor(device="cpu")
        image, _ = decode_image(data, image_path.name)
        assert tensor_digest(cpu.transform(image)) == reference["input_tensor_sha256"]
        prediction = cpu.predict(image)
        image.close()
        expected = torch.tensor(reference["probabilities"])
        actual = torch.tensor(prediction["probabilities"])
        # CPU/GPU 使用不同数值内核；单独记录跨设备误差，不能套用同设备的严格容差。
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
        assert [item["class_idx"] for item in prediction["top3"]] == expected.topk(3).indices.tolist()
        checks["cpu_load_and_prediction"] = {"passed": True, "max_probability_difference": (actual - expected).abs().max().item(),
                                                "top3": prediction["top3"], "output_shape": prediction["output_shape"],
                                                "cross_device_tolerance": {"rtol": 1e-3, "atol": 1e-4}}
        assert not cpu.model.training and all(not layer.training for layer in cpu.model.modules())
        assert all(not parameter.requires_grad for parameter in cpu.model.parameters())
        checks["eval_and_frozen_parameters"] = True
        del cpu

        st.cache_resource.clear()
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not app.exception, list(app.exception)
        assert app.button[0].disabled
        assert app.metric[0].value == "93.20%" and app.metric[1].value == "92.24%"
        checks["initial_page_and_real_metrics"] = True
        with patch("inference.load_predictor", wraps=load_predictor) as loader:
            app.file_uploader[0].upload(image_path.name, data, "image/jpeg").run()
            assert not app.exception and not app.button[0].disabled
            assert len(app.image) == 1
            checks["upload_and_preview"] = True
            app.button[0].click().run()
            assert not app.exception and not app.error
            result = app.session_state["prediction"]
            torch.testing.assert_close(torch.tensor(result["probabilities"]), expected, rtol=1e-5, atol=1e-6)
            assert result["output_shape"] == [1, 39]
            assert len(app.get("progress")) == 3
            checks["web_matches_original_prediction"] = {"passed": True, "top3": result["top3"],
                "max_probability_difference": (torch.tensor(result["probabilities"]) - expected).abs().max().item()}
            app.button[0].click().run()
            assert not app.exception and not app.error
            assert loader.call_count == 1, loader.call_count
            checks["consecutive_prediction_and_model_cached"] = True

            # 同一验证图在内存转为 PNG，仅用于验证更换文件后的清理，不计算新评价指标。
            png = io.BytesIO()
            with Image.open(io.BytesIO(data)) as picture:
                picture.save(png, format="PNG")
            app.file_uploader[0].upload("replacement.png", png.getvalue(), "image/png").run()
            assert "prediction" not in app.session_state
            assert len(app.metric) == 2 and len(app.get("progress")) == 0
            checks["replacement_clears_old_result"] = True
            app.button[0].click().run()
            assert not app.exception and not app.error
            app.file_uploader[0].clear().run()
            assert "prediction" not in app.session_state and app.button[0].disabled
            checks["removal_clears_old_result"] = True

        for name, content, message in [
            ("broken.jpg", b"not an image", "损坏"),
            ("disguised.jpg", png.getvalue(), "不一致"),
            ("too_large.jpg", b"0" * (MAX_FILE_BYTES + 1), "过大"),
        ]:
            app.file_uploader[0].upload(name, content, "image/jpeg").run()
            assert not app.exception
            assert any(message in item.value for item in app.error)
            assert "prediction" not in app.session_state and app.button[0].disabled
            checks[name] = True

        with patch("inference.model_cache_key", side_effect=InferenceError("模型文件不存在：验收故障注入")):
            app.run()
            assert any("模型文件不存在" in item.value for item in app.error)
            assert app.button[0].disabled
            checks["missing_model_ui"] = True

        fixtures = output / "fault_fixtures"
        fixtures.mkdir()
        bad_model = fixtures / "not_the_frozen_model.pt"
        bad_model.write_bytes(b"deliberately invalid model for SHA256 rejection")
        bad_record = {**record, "checkpoint": str(bad_model)}
        bad_manifest = fixtures / "bad_model_record.json"
        bad_manifest.write_text(json.dumps(bad_record), encoding="utf-8")
        bad_key = model_cache_key(bad_manifest)
        app.file_uploader[0].upload(image_path.name, data, "image/jpeg")
        with patch("inference.model_cache_key", return_value=bad_key):
            app.run()
            app.button[0].click().run()
            assert not app.exception
            assert any("SHA256 不符" in item.value for item in app.error)
            assert "prediction" not in app.session_state
            checks["hash_mismatch_ui"] = True
        checks["normal_app_without_dataset_or_split_reads"] = True

    # 大像素图片用内存创建；Pillow 自身限制没有被关闭。
    large = io.BytesIO()
    with Image.new("1", (5000, 4001)) as picture:
        picture.save(large, format="PNG")
    try:
        decode_image(large.getvalue(), "large_pixels.png")
        raise AssertionError("过大像素图片没有拒绝")
    except InferenceError as exc:
        assert "像素过大" in str(exc)
    checks["oversized_pixel_count"] = True
    assert Image.MAX_IMAGE_PIXELS is not None
    checks["pillow_safety_limit_retained"] = Image.MAX_IMAGE_PIXELS
    report["final_model_sha256"] = sha256_file(record["checkpoint"])
    assert report["final_model_sha256"] == record["checkpoint_sha256"]
    report["seconds"] = time.perf_counter() - started
    report["status"] = "passed"
    with (output / "verification.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
