"""用临时生成的图片验证坏图识别与原始数据保护，不使用真实数据集。"""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from check_dataset import run_check


def file_hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class DatasetCheckTests(unittest.TestCase):
    def test_counts_corruption_and_originals_preserved(self):
        with tempfile.TemporaryDirectory(prefix="leaf_check_test_") as temporary:
            base = Path(temporary)
            data = base / "data"
            (data / "healthy").mkdir(parents=True)
            (data / "diseased").mkdir()
            with Image.new("RGB", (32, 24), "green") as image:
                image.save(data / "healthy" / "leaf.JPG")
            with Image.new("RGBA", (40, 30), (20, 100, 20, 128)) as image:
                image.save(data / "diseased" / "leaf.png")
            (data / "diseased" / "broken.jpg").write_bytes(b"not a real image")
            (data / "healthy" / "notes.txt").write_text("keep this", encoding="utf-8")
            before = file_hashes(data)
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = run_check(data, base / "reports", 42)
            output = next((base / "reports").iterdir())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["class_count"], 2)
            self.assertEqual(summary["total_images"], 3)
            self.assertEqual(summary["valid_images"], 2)
            self.assertEqual(summary["corrupted_images"], 1)
            self.assertEqual(summary["ignored_non_image_files"], 1)
            self.assertEqual(file_hashes(data), before)
            with Image.open(output / "samples.jpg") as sample:
                sample.load()
            # 第二次运行必须生成另一份报告，不覆盖前一份。
            with contextlib.redirect_stdout(io.StringIO()):
                run_check(data, base / "reports", 42)
            self.assertEqual(len(list((base / "reports").iterdir())), 2)

    def test_refuse_output_inside_original_dataset(self):
        with tempfile.TemporaryDirectory(prefix="leaf_check_guard_") as temporary:
            data = Path(temporary) / "data"
            (data / "healthy").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "报告目录"):
                run_check(data, data / "reports", 42)
            self.assertFalse((data / "reports").exists())


if __name__ == "__main__":
    unittest.main()
