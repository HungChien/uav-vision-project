from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


DATASET_HOME = {
    "visdrone": "https://github.com/VisDrone/VisDrone-Dataset",
    "uav123": "https://cemse.kaust.edu.sa/ivul/uav123",
}

VISDRONE_GDRIVE_IDS = {
    "det-train": "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn",
    "det-val": "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59",
}


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output.open("wb") as file:
        shutil.copyfileobj(response, file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset archives when direct URLs are available.")
    parser.add_argument("name", choices=sorted(DATASET_HOME), help="Dataset name.")
    parser.add_argument("--visdrone-split", choices=sorted(VISDRONE_GDRIVE_IDS), help="Known VisDrone Google Drive archive.")
    parser.add_argument("--url", help="Direct archive URL. If omitted, only the official dataset page is printed.")
    parser.add_argument("--output", type=Path, help="Output archive path.")
    args = parser.parse_args()

    print(f"Official {args.name} entry point: {DATASET_HOME[args.name]}")
    if args.name == "visdrone" and args.visdrone_split and not args.url:
        file_id = VISDRONE_GDRIVE_IDS[args.visdrone_split]
        args.url = f"https://drive.google.com/uc?id={file_id}"

    if not args.url:
        print("No direct archive URL provided. Download manually from the official page or pass --url.")
        return

    if args.output is None:
        filename = args.url.rstrip("/").split("/")[-1] or f"{args.name}.zip"
        args.output = Path("data/raw") / filename

    print(f"Downloading {args.url}")
    download(args.url, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
