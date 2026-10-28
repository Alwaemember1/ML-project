"""本地配置、错误权重和仓库内清单的发布回归检查；不运行模型。"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import project_config
from scripts.configure_model import configure
from scripts.prepare_formal_split import verify_split


class ProjectConfigTests(unittest.TestCase):
    def setUp(self):
        project_config.RUNTIME.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=project_config.RUNTIME)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_local_pointer_precedes_historical_record_and_env_precedes_both(self):
        with patch.object(project_config, "ROOT", self.root), patch.dict(os.environ):
            os.environ.pop("LEAF_MODEL_RECORD", None)
            self.assertEqual(project_config.default_record_path(), self.root / "FINAL_MODEL.json")
            (self.root / "MODEL_CONFIG.json").write_text("{}", encoding="utf-8")
            self.assertEqual(project_config.default_record_path(), self.root / "MODEL_CONFIG.json")
            (self.root / "FINAL_MODEL.local.json").write_text("{}", encoding="utf-8")
            self.assertEqual(project_config.default_record_path(), self.root / "FINAL_MODEL.local.json")
            os.environ["LEAF_MODEL_RECORD"] = "custom.json"
            self.assertEqual(project_config.default_record_path(), self.root / "custom.json")

    def test_setup_checks_bytes_preserves_metadata_and_refuses_overwrite(self):
        model = self.root / "fake-fixture.pt"
        model.write_bytes(b"unit-test-fixture-not-real-weights")
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        reference = self.root / "reference.json"
        metadata = {"status": "frozen_final_model", "checkpoint_sha256": digest,
                    "checkpoint": "historical/path", "class_to_idx": {"example": 0},
                    "preprocessing": {"keep_this": True}, "epoch": 3}
        reference.write_text(json.dumps(metadata), encoding="utf-8")
        metrics = self.root / "metrics.json"
        metrics.write_text(json.dumps({"total": 9192, "classes": [{}] * 39}), encoding="utf-8")
        target = self.root / "local.json"
        with patch("scripts.configure_model.FROZEN_SHA256", digest):
            configure(model, target, reference=reference, metrics=metrics)
            actual = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(actual["preprocessing"], metadata["preprocessing"])
            self.assertEqual(actual["class_to_idx"], metadata["class_to_idx"])
            self.assertEqual(actual["checkpoint"], model.name)
            with self.assertRaises(FileExistsError):
                configure(model, target, reference=reference, metrics=metrics)
            model.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                configure(model, self.root / "other.json", reference=reference, metrics=metrics)
        self.assertEqual(json.loads(reference.read_text(encoding="utf-8")), metadata)

    def test_metadata_only_does_not_claim_external_source_or_smoke_audit(self):
        result = verify_split(verify_sources=False)
        self.assertEqual(result["counts"], {"train": 42890, "validation": 9192, "test": 9192})
        self.assertFalse(result["source_provenance_checked"])
        self.assertIsNone(result["all_smoke_hashes_train_only"])
        self.assertIsNone(result["smoke_unique_hashes"])
        self.assertTrue(all(value == 0 for pair in result["overlaps"].values() for value in pair.values()))


if __name__ == "__main__":
    unittest.main()
