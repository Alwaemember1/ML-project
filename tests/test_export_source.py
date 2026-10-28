"""纯源码导出在仓库附带模型后的集成回归，不加载模型或原图。"""

import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.export_source import export_source


class ExportSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.write("README.md", b"source")
        self.write("启动网页.bat", b"launcher")
        self.write("models/final_resnet18.pt", b"not-real-weights")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        self.output = self.root / "deliverables/source.zip"
        patcher = patch("scripts.export_source.ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def test_export_skips_only_published_model_and_preserves_sources(self):
        result = export_source(self.output)
        self.assertFalse(result["model_or_data_included"])
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(set(archive.namelist()), {"README.md", "启动网页.bat", "SOURCE_EXPORT.json"})
            self.assertEqual(archive.read("README.md"), b"source")
            self.assertEqual(archive.read("启动网页.bat"), b"launcher")
            manifest = json.loads(archive.read("SOURCE_EXPORT.json"))
            self.assertFalse(manifest["weights_and_data_included"])
            self.assertEqual(manifest["excluded_published_model"], "models/final_resnet18.pt")
        self.assertEqual((self.root / "models/final_resnet18.pt").read_bytes(), b"not-real-weights")

    def test_other_weights_still_fail_without_partial_archive(self):
        self.write("models/other.pt", b"unexpected")
        with self.assertRaisesRegex(ValueError, "models/other.pt"):
            export_source(self.output)
        self.assertFalse(self.output.exists())

    def test_local_config_still_fails_without_partial_archive(self):
        self.write("FINAL_MODEL.local.json", b"{}")
        with self.assertRaisesRegex(ValueError, "本机配置"):
            export_source(self.output)
        self.assertFalse(self.output.exists())

    def test_existing_archive_is_not_overwritten(self):
        self.write("deliverables/source.zip", b"existing")
        with self.assertRaises(FileExistsError):
            export_source(self.output)
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_output_must_stay_in_deliverables(self):
        with self.assertRaises(ValueError):
            export_source(self.root / "source.zip")


if __name__ == "__main__":
    unittest.main()
