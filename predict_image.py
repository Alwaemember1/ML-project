"""与网页共用冻结模型推理，不读取数据清单或重新评价测试集。"""

import argparse
import json
from pathlib import Path

from inference import FINAL_RECORD, InferenceError, decode_image, load_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    parser.add_argument("--output", type=Path, help="可选：另存 JSON，拒绝覆盖已有文件")
    args = parser.parse_args()
    try:
        if args.output and args.output.exists():
            raise InferenceError(f"拒绝覆盖：{args.output}")
        with args.image.open("rb") as file:
            data = file.read(10 * 1024 * 1024 + 1)
        image, info = decode_image(data, args.image.name)
        try:
            predictor = load_predictor(FINAL_RECORD, args.device)
            result = predictor.predict(image)
        finally:
            image.close()
        result["image"] = info
        if args.output:
            with args.output.open("x", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("logits", "probabilities")}, ensure_ascii=False, indent=2))
    except (InferenceError, OSError, RuntimeError) as exc:
        parser.exit(1, f"识别失败：{exc}\n")


if __name__ == "__main__":
    main()
