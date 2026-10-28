"""文档精简后，交付指引生成和源码打包不依赖已移除的旧文档。"""

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import build_teacher_offline, build_teacher_package, delivery_readme


class DeliveryPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("README.md", "# 项目说明\n\n## 局限与许可\n\n原创代码遵循MIT，模型不自动纳入MIT。\n")
        patcher = patch.object(delivery_readme, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_online_and_offline_instructions_match_runtime(self):
        normal = delivery_readme.delivery_readme()
        self.assertIn("requirements-cpu.txt", normal)
        self.assertIn("不附Python", normal)
        self.assertIn("模型不自动纳入MIT", normal)
        offline = delivery_readme.delivery_readme(offline=True)
        self.assertIn("启动前检查.bat", offline)
        self.assertIn("无需再次安装", offline)
        self.assertNotIn("pip install", offline)
        output = self.root / "offline"
        output.mkdir()
        build_teacher_offline.teacher_documents(output)
        self.assertEqual((output / "README.md").read_text(encoding="utf-8"), offline)
        self.assertIn("启动前检查.bat", (output / "项目说明.html").read_text(encoding="utf-8"))

    def test_teacher_package_builds_without_removed_documents(self):
        for name in ("app.py", "inference.py", "smoke_utils.py", "class_labels.py", "plant_care.py",
                     "predict_image.py", "project_config.py", "requirements.txt", "requirements-web.txt",
                     "requirements-training.txt", "requirements-cpu.txt", "requirements-gpu.txt",
                     "启动网页.bat", ".streamlit/config.toml", "LICENSE", "dataset_config.json",
                     "final_test_registry.json"):
            self.write(name, "fixture")
        model = self.write("models/final_resnet18.pt", "synthetic-package-fixture-not-real-weights")
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        self.write("results.json", "{}")
        record = self.write("MODEL_CONFIG.json", json.dumps({"checkpoint": "models/final_resnet18.pt",
            "checkpoint_sha256": digest, "evaluation_metrics": "results.json"}))
        course = self.root / "reports/course"
        for name in ("ACCEPTANCE.md", "CARE_REFERENCE.md", "COURSE_REPORT_FINAL.md", "COURSE_REPORT_FINAL.html",
                     "DEMO_AND_DEFENSE.md", "report_layout_verification.json", "sources.json"):
            self.write(f"reports/course/{name}", "fixture")
        output = self.root / "deliverables/demo"
        with patch.object(build_teacher_package, "ROOT", self.root), \
             patch.object(build_teacher_package, "COURSE", course), \
             patch("project_config.default_record_path", return_value=record), \
             patch("sys.argv", ["build_teacher_package", "--output-dir", str(output), "--no-archive"]), \
             redirect_stdout(StringIO()):
            build_teacher_package.main()
        self.assertTrue((output / "README.md").is_file())
        self.assertTrue((output / "experiment_source/README.md").is_file())
        self.assertFalse((output / "docs/TEACHER_HANDOFF.md").exists())
        portable = json.loads((output / "FINAL_MODEL.json").read_text(encoding="utf-8"))
        self.assertEqual(portable["checkpoint_sha256"], digest)
        self.assertEqual(hashlib.sha256((output / portable["checkpoint"]).read_bytes()).hexdigest(), digest)
        self.assertTrue((output / portable["evaluation_metrics"]).is_file())
        manifest = json.loads((output / "DELIVERY_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_count"], 1)


if __name__ == "__main__":
    unittest.main()
