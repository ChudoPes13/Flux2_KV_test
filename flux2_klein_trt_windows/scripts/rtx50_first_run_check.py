from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trt.config import load_project_config
from flux_trt.diagnostics import utc_timestamp, write_json
from flux_trt.env import collect_env_info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the first RTX 50 TensorRT-LLM validation sequence."
    )
    parser.add_argument("--smoke-variant", default="full", choices=["full", "txtattn_bf16"])
    parser.add_argument("--strict-variant", default="full", choices=["full", "txtattn_bf16"])
    parser.add_argument(
        "--force-non-target",
        action="store_true",
        help="Run GPU load/generation steps even when the GPU is not Blackwell or newer.",
    )
    args = parser.parse_args()

    config = load_project_config()
    config.ensure_directories()
    diagnostics_dir = config.output_path("diagnostics")
    report_path = diagnostics_dir / "rtx50_first_run_report.json"

    env = collect_env_info(allow_container=bool(config.runtime.get("allow_docker", False)))
    nvfp4_target_gpu = bool(env.get("nvfp4_target_gpu", False))

    report: dict[str, Any] = {
        "stage": "rtx50_first_run_check",
        "created_at": utc_timestamp(),
        "status": "pending",
        "rtx50_required": True,
        "force_non_target": args.force_non_target,
        "cuda_capability": env.get("cuda_capability"),
        "is_blackwell_or_newer": bool(env.get("is_blackwell_or_newer", False)),
        "nvfp4_target_gpu": nvfp4_target_gpu,
        "env": env,
        "steps": [],
        "reports": {},
    }

    python = sys.executable
    _run_step(report, "env_check", [python, "scripts/00_env_check.py", "--strict"])
    _attach_json(report, "env_report", diagnostics_dir / "env_report.json")

    _run_step(report, "validate_runtime_dir", [python, "scripts/validate_runtime_dir.py"])
    _attach_json(report, "validate_runtime_dir", diagnostics_dir / "validate_runtime_dir.json")

    _run_step(
        report,
        "inspect_apacheone_checkpoint",
        [python, "scripts/inspect_apacheone_checkpoint.py"],
    )
    _attach_json(
        report,
        "inspect_apacheone_checkpoint_full",
        diagnostics_dir / "inspect_apacheone_checkpoint_full.json",
    )
    _attach_json(
        report,
        "inspect_apacheone_checkpoint_txtattn_bf16",
        diagnostics_dir / "inspect_apacheone_checkpoint_txtattn_bf16.json",
    )

    _run_step(
        report,
        "mock_low_level_adapter_test",
        [python, "scripts/mock_low_level_adapter_test.py"],
    )
    _attach_json(report, "mock_low_level_adapter_test", diagnostics_dir / "mock_low_level_adapter_test.json")

    if nvfp4_target_gpu or args.force_non_target:
        for variant in ["full", "txtattn_bf16"]:
            _run_step(
                report,
                f"check_visualgen_load_{variant}",
                [python, "scripts/check_visualgen_load.py", "--variant", variant],
            )
            _attach_json(
                report,
                f"check_visualgen_load_{variant}",
                diagnostics_dir / f"check_visualgen_load_{variant}.json",
            )

        smoke_dir = config.output_path("root") / "rtx50_first_run" / f"visualgen_prompt_text_{args.smoke_variant}"
        _run_step(
            report,
            "visualgen_prompt_text_smoke_test",
            [
                python,
                "scripts/04_generate_once.py",
                "--variant",
                args.smoke_variant,
                "--mode",
                "visualgen_prompt_text",
                "--output-dir",
                str(smoke_dir),
            ],
            timeout_sec=1800,
        )
        _attach_json(report, "visualgen_prompt_text_smoke_test", smoke_dir / "run_report.json")

        strict_dir = config.output_path("root") / "rtx50_first_run" / f"cached_embeddings_strict_{args.strict_variant}"
        _run_step(
            report,
            "cached_embeddings_strict_strict_test",
            [
                python,
                "scripts/04_generate_once.py",
                "--variant",
                args.strict_variant,
                "--mode",
                "cached_embeddings_strict",
                "--output-dir",
                str(strict_dir),
            ],
            timeout_sec=1800,
        )
        _attach_json(report, "cached_embeddings_strict_strict_test", strict_dir / "run_report.json")
    else:
        reason = (
            "GPU load/generation steps skipped because this is not a Blackwell-or-newer "
            "NVFP4 target GPU. Re-run on RTX 5060 Ti/5090 or use --force-non-target."
        )
        for name in [
            "check_visualgen_load_full",
            "check_visualgen_load_txtattn_bf16",
            "visualgen_prompt_text_smoke_test",
            "cached_embeddings_strict_strict_test",
        ]:
            report["steps"].append(
                {
                    "name": name,
                    "status": "skipped",
                    "returncode": None,
                    "reason": reason,
                }
            )

    failed = [step for step in report["steps"] if step["status"] == "error"]
    skipped = [step for step in report["steps"] if step["status"] == "skipped"]
    if failed:
        report["status"] = "error"
    elif skipped:
        report["status"] = "skipped_non_target_gpu"
    else:
        report["status"] = "ok"

    write_json(report_path, report)
    print(f"RTX 50 first run report: {report_path}")
    print(f"status: {report['status']}")
    return 0 if report["status"] == "ok" else 1


def _run_step(
    report: dict[str, Any],
    name: str,
    command: list[str],
    *,
    timeout_sec: int = 600,
) -> None:
    started_at = utc_timestamp()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        status = "ok" if completed.returncode == 0 else "error"
        report["steps"].append(
            {
                "name": name,
                "status": status,
                "returncode": completed.returncode,
                "started_at": started_at,
                "finished_at": utc_timestamp(),
                "command": command,
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
            }
        )
    except Exception as exc:  # noqa: BLE001
        report["steps"].append(
            {
                "name": name,
                "status": "error",
                "returncode": None,
                "started_at": started_at,
                "finished_at": utc_timestamp(),
                "command": command,
                "error": repr(exc),
            }
        )


def _attach_json(report: dict[str, Any], key: str, path: Path) -> None:
    item: dict[str, Any] = {"path": str(path), "exists": path.exists(), "content": None}
    if path.exists():
        try:
            item["content"] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            item["error"] = repr(exc)
    report["reports"][key] = item


def _tail(text: str, limit: int = 12000) -> str:
    return text[-limit:]


if __name__ == "__main__":
    raise SystemExit(main())
