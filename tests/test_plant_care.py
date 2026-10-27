"""决策参考边界测试，不评价模型、不读取测试图片。"""
import json
import unittest
from pathlib import Path

from plant_care import (DISEASE_CARE, RISK_SCOPE, SEVERITY_SCOPE, build_prediction_report,
                        care_for_class, validate_coverage)


class PlantCareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((Path(__file__).resolve().parents[1] / "FINAL_MODEL.json").read_text(encoding="utf-8"))

    def test_exact_class_coverage_and_sources(self):
        mapping = self.record["class_to_idx"]
        validate_coverage(mapping)
        self.assertEqual(len(DISEASE_CARE), 26)
        for name in mapping:
            with self.subTest(name=name):
                value = care_for_class(name)
                self.assertEqual(value["class_name"], name)
                self.assertTrue(value["actions"])
                if name != "Background_without_leaves":
                    self.assertTrue(value["sources"])
                    self.assertTrue(all(source["url"].startswith("https://") for source in value["sources"]))

    def test_background_and_health_are_not_disease_diagnoses(self):
        self.assertEqual(care_for_class("Background_without_leaves")["risk_level"], "不适用")
        healthy = care_for_class("Apple___healthy")
        self.assertEqual(healthy["risk_level"], "低")
        self.assertIn("不能排除", healthy["reason"])
        with self.assertRaises(ValueError):
            care_for_class("Unsupported___disease")
        with self.assertRaises(ValueError):
            care_for_class("Unsupported___healthy")

    def test_probability_is_not_risk_or_severity(self):
        # 同类的高、低模型分数只能改变报告里的分数，不能改风险/伪造严重程度。
        name = "Potato___Late_blight"
        for probability in (0.2, 0.99):
            result = {"checkpoint_sha256":self.record["checkpoint_sha256"],
                      "top3":[{"class_name":name,"class_idx":22,"probability":probability}]}
            report = build_prediction_report(result, {"filename":"sample.webp"}, self.record)
            self.assertIn("高（类别规则参考）", report)
            self.assertIn(RISK_SCOPE, report)
            self.assertIn(SEVERITY_SCOPE, report)
            self.assertIn("尚未经过田间效果验证", report)
            self.assertIn(self.record["checkpoint_sha256"], report)

    def test_missing_or_extra_class_cannot_pass_coverage(self):
        mapping = dict(self.record["class_to_idx"])
        mapping["NewCrop___new_disease"] = 39
        with self.assertRaises(ValueError):
            validate_coverage(mapping)


if __name__ == "__main__":
    unittest.main(verbosity=2)
