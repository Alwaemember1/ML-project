"""只用人工计数验证评价公式，完全不读取真实测试图片。"""

import unittest

from evaluation_metrics import classification_metrics, confusion_pairs


class EvaluationMetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        result = classification_metrics([[3, 0], [0, 2]], ["a", "b"])
        self.assertEqual(result["correct"], 5)
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["macro_f1"], 1)

    def test_missing_predictions_not_hidden(self):
        result = classification_metrics([[9, 0], [1, 0]], ["large", "small"])
        self.assertEqual(result["accuracy"], 0.9)
        self.assertEqual(result["classes"][1]["precision"], 0)
        self.assertEqual(result["classes"][1]["recall"], 0)
        self.assertEqual(result["classes"][1]["f1"], 0)
        self.assertTrue(result["classes"][1]["no_predicted_samples"])
        self.assertAlmostEqual(result["macro_f1"], (18 / 19) / 2)

    def test_true_rows_and_predicted_columns(self):
        result = classification_metrics([[8, 2], [1, 9]], ["a", "b"])
        self.assertEqual(result["classes"][0]["support"], 10)
        self.assertEqual(result["classes"][0]["predicted_count"], 9)
        self.assertAlmostEqual(result["classes"][0]["precision"], 8 / 9)
        self.assertEqual(result["classes"][0]["recall"], 0.8)
        pair = confusion_pairs([[8, 2], [1, 9]], ["a", "b"])[0]
        self.assertEqual((pair["true_idx"], pair["pred_idx"], pair["count"]), (0, 1, 2))
        self.assertEqual(pair["fraction_of_true_class"], 0.2)

    def test_empty_evaluation_is_error(self):
        with self.assertRaises(ValueError):
            classification_metrics([[0, 0], [0, 0]], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
