"""用临时元数据检查相对路径解析，不使用原图或训练。"""

import json
import tempfile
import unittest
from pathlib import Path

from inference import FROZEN_SHA256, model_cache_key, read_experiment_metrics, resolve_record_file
from smoke_utils import RUNTIME


class PortablePathsTests(unittest.TestCase):
    def test_relative_paths_resolve_from_record_not_cwd(self):
        with tempfile.TemporaryDirectory(dir=RUNTIME) as folder:
            root = Path(folder)
            (root / "models").mkdir()
            (root / "models/final.pt").write_bytes(b"metadata-only-fixture")
            record = {"status": "frozen_final_model", "checkpoint_sha256": FROZEN_SHA256,
                      "checkpoint": "models/final.pt", "evaluation_metrics": "metrics.json"}
            (root / "FINAL_MODEL.json").write_text(json.dumps(record), encoding="utf-8")
            metrics = {"accuracy": .5, "macro_f1": .4, "total": 9192, "classes": [{}] * 39}
            (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            key = model_cache_key(root / "FINAL_MODEL.json")
            self.assertEqual(Path(key[2]), root / "models/final.pt")
            self.assertEqual(read_experiment_metrics(root / "FINAL_MODEL.json")[0], metrics)

    def test_absolute_paths_are_preserved(self):
        path = Path(__file__).resolve()
        self.assertEqual(resolve_record_file(str(path), Path("unused/FINAL_MODEL.json")), path)


if __name__ == "__main__":
    unittest.main()
