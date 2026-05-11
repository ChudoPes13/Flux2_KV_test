from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.diagnostics import write_json
from flux_trt.env import collect_env_info
from flux_trt.config import load_project_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Windows/CUDA/TensorRT-LLM environment.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when required generation runtime checks fail.",
    )
    args = parser.parse_args()

    diagnostics_dir = ROOT / "data" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    config = load_project_config()
    allow_container = bool(config.runtime.get("allow_docker", False))
    report = collect_env_info(allow_container=allow_container)
    report_path = write_json(diagnostics_dir / "env_report.json", report)

    print(f"env report: {report_path}")
    print(f"status: {report['status']}")
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    if report["errors"]:
        print("errors:")
        for error in report["errors"]:
            print(f"  - {error}")
        print()
        if report.get("allow_container") and report.get("system") == "Windows":
            print("Runtime check should be run inside the NVIDIA TensorRT-LLM Docker container.")
            print("Use: .\\scripts\\docker_enter.ps1")
        else:
            print("Install/check: CUDA-capable PyTorch, tensorrt, tensorrt-llm, NVIDIA driver.")

    return 1 if args.strict and report["status"] != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
