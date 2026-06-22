from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path


PACKAGES = [
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "yaml",
]


def package_status(name: str) -> dict[str, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"name": name, "status": "missing", "detail": repr(exc)}
    return {"name": name, "status": "ok", "version": getattr(module, "__version__", "unknown")}


def torch_status() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    cuda_available = torch.cuda.is_available()
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def main() -> None:
    report = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": [package_status(name) for name in PACKAGES],
        "torch": torch_status(),
    }

    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase1_env_check.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

