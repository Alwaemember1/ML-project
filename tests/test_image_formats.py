"""新增上传格式的真实解码及网页识别检查；只用内存生成的合成图片。"""

import io
import os
import unittest

from inference import ALLOWED_FORMATS, InferenceError, ROOT, decode_image

from PIL import Image
from streamlit.testing.v1 import AppTest


def encoded_image(format_name, *, animated=False):
    output = io.BytesIO()
    with Image.new("RGB", (64, 48), (40, 140, 70)) as image:
        options = {"lossless": True} if format_name == "WEBP" else {}
        if animated:
            with Image.new("RGB", image.size, (200, 60, 20)) as other:
                image.save(output, format=format_name, save_all=True, append_images=[other],
                           duration=100, **options)
        else:
            image.save(output, format=format_name, **options)
    return output.getvalue()


class ImageFormatTests(unittest.TestCase):
    def test_all_extensions_decode_to_rgb(self):
        for suffix, format_name in ALLOWED_FORMATS.items():
            with self.subTest(extension=suffix):
                data = encoded_image(format_name)
                image, info = decode_image(data, "photo" + suffix.upper())
                try:
                    self.assertEqual(info["format"], format_name)
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (64, 48))
                    if format_name in ("PNG", "WEBP", "BMP", "TIFF"):
                        self.assertEqual(image.getpixel((20, 20)), (40, 140, 70))
                finally:
                    image.close()

    def test_real_format_and_corruption_are_checked(self):
        with self.assertRaisesRegex(InferenceError, "不一致"):
            decode_image(encoded_image("PNG"), "disguised.webp")
        with self.assertRaisesRegex(InferenceError, "损坏|无法解码"):
            decode_image(b"not a real WebP", "broken.webp")

    def test_animation_and_multipage_are_rejected(self):
        for format_name, suffix in [("WEBP", ".webp"), ("GIF", ".gif"),
                                    ("TIFF", ".tiff"), ("AVIF", ".avif")]:
            with self.subTest(format=format_name):
                with self.assertRaisesRegex(InferenceError, "单帧静态"):
                    decode_image(encoded_image(format_name, animated=True), "animation" + suffix)

    @unittest.skipUnless(os.environ.get("LEAF_RUN_MODEL_TESTS") == "1",
                         "需显式设置LEAF_RUN_MODEL_TESTS=1且提供冻结权重，默认不运行真实模型测试")
    def test_web_upload_preview_and_real_prediction(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        accepted = {item.lstrip(".").lower() for item in app.file_uploader[0].allowed_type}
        self.assertEqual(accepted, {suffix[1:] for suffix in ALLOWED_FORMATS})
        for suffix, format_name in ALLOWED_FORMATS.items():
            with self.subTest(upload=suffix):
                app.file_uploader[0].upload("photo" + suffix, encoded_image(format_name)).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(len(app.image), 1)
                self.assertNotIn("prediction", app.session_state)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(app.session_state["prediction"]["output_shape"], [1, 39])
                self.assertEqual(len(app.session_state["prediction"]["top3"]), 3)
        app.file_uploader[0].upload("fake.webp", encoded_image("PNG")).run()
        self.assertTrue(any("不一致" in error.value for error in app.error))
        self.assertNotIn("prediction", app.session_state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
