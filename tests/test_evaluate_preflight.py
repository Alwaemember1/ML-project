"""最终测试仍须提供原验收快照，错误快照不能触发划分或原图读取。"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "最终测试入口导入需要PyTorch和torchvision")
class EvaluatePreflightTests(unittest.TestCase):
    def test_external_snapshot_is_checked_before_dataset_access(self):
        from scripts import evaluate_final
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "reports/fixture"
            run.mkdir(parents=True)
            checkpoint = run / "checkpoints/best_epoch_003.pt"
            (run / "completion.json").write_text(json.dumps({"status": "completed_and_verified",
                "best_epoch": 3, "training_report": "summary.json"}), encoding="utf-8")
            (run / "training_config.json").write_text("{}", encoding="utf-8")
            (run / "summary.json").write_text(json.dumps({"best_checkpoint": str(checkpoint), "best_epoch": 3}), encoding="utf-8")
            snapshot = root / "external_snapshot.md"
            arguments = ["evaluate_final", "--training-run", str(run), "--project-status", str(snapshot)]
            with patch.object(evaluate_final, "ROOT", root), patch("sys.argv", arguments), \
                 patch.object(evaluate_final, "verify_split", side_effect=RuntimeError("stop-before-data")) as verify:
                with self.assertRaises(FileNotFoundError):
                    evaluate_final.main()
                snapshot.write_text("unrelated experiment", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "原验收说明"):
                    evaluate_final.main()
                verify.assert_not_called()
                snapshot.write_text(checkpoint.relative_to(root).as_posix(), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "stop-before-data"):
                    evaluate_final.main()
                verify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
