from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import load_project_config
from flux_trt.diagnostics import utc_timestamp, write_json
from flux_trt.runtime_validation import validate_runtime_variant


VARIANTS = ["full", "txtattn_bf16"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VisualGen runtime_dir layout without GPU load.")
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = parser.parse_args()

    config = load_project_config()
    config.ensure_directories()
    selected = VARIANTS if args.variant == "all" else [args.variant]

    results = [validate_runtime_variant(config, variant) for variant in selected]
    report = {
        "stage": "validate_runtime_dir",
        "created_at": utc_timestamp(),
        "status": "ok" if all(item["status"] == "ok" for item in results) else "error",
        "gpu_loaded": False,
        "variants": results,
    }
    report_path = config.output_path("diagnostics") / "validate_runtime_dir.json"
    write_json(report_path, report)

    print(f"runtime validation report: {report_path}")
    for item in results:
        print(f"{item['variant']}: {item['status']}")
        for error in item["errors"]:
            print(f"  - {error}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
