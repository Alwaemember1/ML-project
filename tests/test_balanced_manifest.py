"""训练前清单边界测试，合成元数据，不读取真实图片。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from balanced_manifest import file_sha256, load_balanced_training
from project_config import RUNTIME

ROOT = Path(__file__).resolve().parents[1]


class BalancedManifestTests(unittest.TestCase):
    def setUp(self):
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=RUNTIME)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "train.csv"
        self.manifest = self.root / "candidate.csv"
        self.mapping = {"A":0,"B":1}
        self.a = {"relative_path":"A/a.jpg","class_name":"A","class_idx":"0","sha256":"a"*64}
        self.b = {"relative_path":"B/b.jpg","class_name":"B","class_idx":"1","sha256":"b"*64}
        self.write_rows(self.source,[self.a,self.b])
        self.write_rows(self.manifest,[self.a,self.b])
        self.lock = {"manifest_sha256":file_sha256(self.manifest),"source_formal_split_id":"fixed",
                     "source_train_sha256":file_sha256(self.source),"per_class":1,"total":2,"classes":2}
        self.write_lock()

    def write_rows(self,path,rows):
        with path.open("w",encoding="utf-8",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=self.a)
            writer.writeheader();writer.writerows(rows)

    def write_lock(self):
        (self.root / "candidate_balance_lock.json").write_text(json.dumps(self.lock),encoding="utf-8")

    def relock_candidate(self):
        self.lock["manifest_sha256"] = file_sha256(self.manifest)
        self.write_lock()

    def load(self,validation=()):
        return load_balanced_training(self.manifest,self.source,list(validation),self.mapping,"fixed")

    def test_valid_locked_balanced_subset(self):
        rows,check=self.load()
        self.assertEqual(len(rows),2)
        self.assertEqual(check["class_counts"],{"A":1,"B":1})
        self.assertTrue(check["all_rows_subset_of_formal_train"])

    def test_modified_manifest_rejected(self):
        with self.manifest.open("a",encoding="utf-8") as file:file.write("\n")
        with self.assertRaisesRegex(ValueError,"SHA256"):
            self.load()

    def test_outside_training_record_rejected_even_with_new_candidate_hash(self):
        self.write_rows(self.manifest,[self.a,{**self.b,"relative_path":"B/unseen.jpg"}])
        self.relock_candidate()
        with self.assertRaisesRegex(ValueError,"不属于正式训练集"):
            self.load()

    def test_duplicate_and_unbalanced_candidates_rejected(self):
        self.write_rows(self.manifest,[self.a,self.a]);self.relock_candidate()
        with self.assertRaisesRegex(ValueError,"重复"):
            self.load()
        self.write_rows(self.manifest,[self.a]);self.relock_candidate()
        with self.assertRaisesRegex(ValueError,"等量|总数"):
            self.load()

    def test_validation_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,"验证集存在交集"):
            self.load([self.a])


if __name__ == "__main__":
    unittest.main(verbosity=2)
