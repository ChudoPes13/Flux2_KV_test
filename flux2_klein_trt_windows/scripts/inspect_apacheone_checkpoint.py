from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.checkpoint_inspection import inspect_safetensors_checkpoint
from flux_trt.config import load_project_config
from flux_trt.diagnostics import utc_timestamp, write_json


VARIANTS = ["full", "txtattn_bf16"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ApacheOne safetensors checkpoints on CPU.")
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = parser.parse_args()

    config = load_project_config()
    config.ensure_directories()
    selected = VARIANTS if args.variant == "all" else [args.variant]

    exit_code = 0
    for variant in selected:
        checkpoint = config.checkpoint_path(variant)
        report = {
            "stage": "inspect_apacheone_checkpoint",
            "created_at": utc_timestamp(),
            "variant": variant,
            "status": "pending",
            "inspection": None,
            "error": None,
            "gpu_loaded": False,
        }
        try:
            inspection = inspect_safetensors_checkpoint(checkpoint)
            report["inspection"] = inspection
            txt_in = inspection["txt_in"]
            report["status"] = "ok" if txt_in["exists"] and txt_in["compatible_with_prompt_embeds"] else "error"
            if report["status"] != "ok":
                report["error"] = "txt_in.weight is missing or incompatible with prompt_embeds width 12288"
        except Exception as exc:  # noqa: BLE001
            report["status"] = "error"
            report["error"] = repr(exc)

        report_path = config.output_path("diagnostics") / f"inspect_apacheone_checkpoint_{variant}.json"
        write_json(report_path, report)
        print(f"{variant}: {report['status']} -> {report_path}")
        if report["inspection"]:
            inspection = report["inspection"]
            print(f"  tensor_count: {inspection['tensor_count']}")
            print(f"  dtypes: {inspection['dtypes']}")
            print(f"  txt_in: {inspection['txt_in']}")
        if report["status"] != "ok":
            exit_code = 1
            if report["error"]:
                print(f"  error: {report['error']}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
