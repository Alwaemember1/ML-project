"""从已完成实验的真实逐轮记录重新画曲线；不训练、不修改已有图片。"""

import argparse
import json
from pathlib import Path

from train_v1 import draw_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "training_summary.json").read_text(encoding="utf-8"))
    filename = "training_curves_reviewed.png"
    if (args.run_dir / filename).exists():
        raise FileExistsError("拒绝覆盖已有审阅曲线")
    draw_curves(summary["history"], args.run_dir, summary["best_epoch"], filename=filename)
    print(args.run_dir / filename)


if __name__ == "__main__":
    main()
