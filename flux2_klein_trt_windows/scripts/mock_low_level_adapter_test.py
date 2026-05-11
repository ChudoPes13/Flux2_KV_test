from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import load_project_config
from flux_trt.diagnostics import utc_timestamp, write_json
from flux_trt.generation_inputs import load_generation_inputs, validate_generation_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble strict GenerationInputs without real generation.")
    parser.add_argument("--text-encoder-variant", default=None)
    args = parser.parse_args()

    config = load_project_config()
    config.ensure_directories()

    report = {
        "stage": "mock_low_level_adapter_test",
        "created_at": utc_timestamp(),
        "text_encoder_variant": args.text_encoder_variant or config.default_text_encoder_variant(),
        "status": "pending",
        "error": None,
        "real_generation_started": False,
    }
    try:
        inputs = load_generation_inputs(config, text_encoder_variant=args.text_encoder_variant)
        validation = validate_generation_inputs(inputs)
        report.update(validation)
    except Exception as exc:  # noqa: BLE001
        report["status"] = "error"
        report["error"] = repr(exc)

    report_path = config.output_path("diagnostics") / "mock_low_level_adapter_test.json"
    write_json(report_path, report)
    print(f"mock low-level adapter report: {report_path}")
    print(f"status: {report['status']}")
    if report.get("errors"):
        for error in report["errors"]:
            print(f"  - {error}")
    if report["error"]:
        print(f"error: {report['error']}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
