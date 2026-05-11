from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import load_project_config
from flux_trt.diagnostics import write_json
from flux_trt.report import create_run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run both ApacheOne variants.")
    parser.add_argument(
        "--mode",
        default="cached_embeddings_strict",
        choices=["cached_embeddings_strict", "visualgen_prompt_text"],
    )
    args = parser.parse_args()

    config = load_project_config()
    config.ensure_directories()
    compare_dir = create_run_dir(config.output_path("root"), "compare")
    results = []

    for variant in ["full", "txtattn_bf16"]:
        variant_dir = compare_dir / variant
        command = [
            sys.executable,
            str(ROOT / "scripts" / "04_generate_once.py"),
            "--variant",
            variant,
            "--mode",
            args.mode,
            "--output-dir",
            str(variant_dir),
        ]
        completed = subprocess.run(command, cwd=str(ROOT), text=True)
        results.append(
            {
                "variant": variant,
                "mode": args.mode,
                "returncode": completed.returncode,
                "output_dir": str(variant_dir),
                "output_png_exists": (variant_dir / "output.png").exists(),
                "run_report": str(variant_dir / "run_report.json"),
            }
        )

    report = {
        "compare_dir": str(compare_dir),
        "results": results,
        "status": "success" if all(item["returncode"] == 0 for item in results) else "error",
    }
    report_path = write_json(compare_dir / "compare_report.json", report)
    print(f"compare report: {report_path}")
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
