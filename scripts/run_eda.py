from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_vision.data.uav123 import analyze_uav123_annotations
from uav_vision.data.visdrone import analyze_visdrone_det


def write_report(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset EDA for UAV vision datasets.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    visdrone = subparsers.add_parser("visdrone-det", help="Analyze VisDrone detection annotations.")
    visdrone.add_argument("--images", type=Path, required=True)
    visdrone.add_argument("--annotations", type=Path, required=True)
    visdrone.add_argument("--output", type=Path, default=Path("outputs/eda/visdrone_det"))

    uav123 = subparsers.add_parser("uav123", help="Analyze UAV123 tracking annotations.")
    uav123.add_argument("--annotations", type=Path, required=True)
    uav123.add_argument("--output", type=Path, default=Path("outputs/eda/uav123"))

    args = parser.parse_args()

    if args.dataset == "visdrone-det":
        report = analyze_visdrone_det(args.images, args.annotations, args.output)
    elif args.dataset == "uav123":
        report = analyze_uav123_annotations(args.annotations, args.output)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    write_report(report, args.output)


if __name__ == "__main__":
    main()

