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
        "gpu_name": env.get("gpu_name"),
        "cuda_capability": env.get("cuda_capability"),
        "is_blackwell_or_newer": bool(env.get("is_blackwell_or_newer", False)),
        "nvfp4_target_gpu": nvfp4_target_gpu,
        "vram_total_gb": env.get("vram_total_gb"),
        "vram_free_before_load": env.get("vram_free_before_load"),
        "docker_image": env.get("docker_image"),
        "torch_version": env.get("torch_cuda", {}).get("torch_version"),
        "tensorrt_llm_version": _import_version(env, "tensorrt_llm"),
        "model_dir": None,
        "variant": None,
        "mode": "rtx50_first_run_check",
        "prompt_cache_used": None,
        "smoke_test_only": None,
        "detected_oom": False,
        "detected_unsupported_arch": False,
        "detected_missing_model_index": False,
        "detected_invalid_safetensors": False,
        "error_category": None,
        "interpretation": None,
        "env": env,
        "steps": [],
        "reports": {},
    }

    python = sys.executable
    _run_step(report, "container_env_diagnostics", [python, "scripts/00_container_check.py"])
    _attach_json(report, "container_report", diagnostics_dir / "container_report.json")
    if _step_failed(report, "container_env_diagnostics"):
        _skip_steps(
            report,
            [
                "validate_runtime_dir",
                "inspect_apacheone_checkpoint",
                "mock_low_level_adapter_test",
                "check_visualgen_supported_model",
                "check_visualgen_load_full",
                "check_visualgen_load_txtattn_bf16",
                "visualgen_prompt_text_smoke_test",
                "cached_embeddings_strict_strict_test",
            ],
            "Container/env diagnostics failed. Do not continue until container/driver/GPU setup is fixed.",
        )
        _finalize(report, report_path)
        return 1

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

    setup_blocked = any(
        _step_failed(report, name)
        for name in [
            "validate_runtime_dir",
            "inspect_apacheone_checkpoint",
            "mock_low_level_adapter_test",
        ]
    )
    if setup_blocked:
        _skip_steps(
            report,
            [
                "check_visualgen_supported_model",
                "check_visualgen_load_full",
                "check_visualgen_load_txtattn_bf16",
                "visualgen_prompt_text_smoke_test",
                "cached_embeddings_strict_strict_test",
            ],
            "CPU/layout/cache prerequisites failed. Do not continue into GPU runtime checks.",
        )
        _finalize(report, report_path)
        return 1

    if nvfp4_target_gpu or args.force_non_target:
        _run_step(
            report,
            "check_visualgen_supported_model",
            [python, "scripts/check_visualgen_supported_model.py"],
            timeout_sec=1800,
        )
        _attach_json(
            report,
            "check_visualgen_supported_model",
            diagnostics_dir / "check_visualgen_supported_model_black_forest_labs_flux_2_dev.json",
        )
        if _step_failed(report, "check_visualgen_supported_model"):
            _skip_steps(
                report,
                [
                    "check_visualgen_load_full",
                    "check_visualgen_load_txtattn_bf16",
                    "visualgen_prompt_text_smoke_test",
                    "cached_embeddings_strict_strict_test",
                ],
                "Official supported VisualGen model failed. Do not touch ApacheOne/Klein-KV until VisualGen/container setup is fixed.",
            )
            _finalize(report, report_path)
            return 1

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
        _skip_steps(report, [
            "check_visualgen_supported_model",
            "check_visualgen_load_full",
            "check_visualgen_load_txtattn_bf16",
            "visualgen_prompt_text_smoke_test",
            "cached_embeddings_strict_strict_test",
        ], reason)

    _finalize(report, report_path)
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
        stdout_path, stderr_path = _write_step_logs(report, name, completed.stdout, completed.stderr)
        status = "ok" if completed.returncode == 0 else "error"
        report["steps"].append(
            {
                "name": name,
                "status": status,
                "returncode": completed.returncode,
                "started_at": started_at,
                "finished_at": utc_timestamp(),
                "command": command,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
                "error_category": _classify_error(completed.stdout + "\n" + completed.stderr)
                if status == "error"
                else None,
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


def _write_step_logs(report: dict[str, Any], name: str, stdout: str, stderr: str) -> tuple[Path, Path]:
    diagnostics_dir = ROOT / "data" / "diagnostics"
    safe_name = _safe_name(name)
    stdout_path = diagnostics_dir / f"rtx50_first_run_{safe_name}_stdout.log"
    stderr_path = diagnostics_dir / f"rtx50_first_run_{safe_name}_stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
    return stdout_path, stderr_path


def _skip_steps(report: dict[str, Any], names: list[str], reason: str) -> None:
    for name in names:
        report["steps"].append(
            {
                "name": name,
                "status": "skipped",
                "returncode": None,
                "reason": reason,
            }
        )


def _step_failed(report: dict[str, Any], name: str) -> bool:
    return any(step["name"] == name and step["status"] == "error" for step in report["steps"])


def _finalize(report: dict[str, Any], report_path: Path) -> None:
    _aggregate_detection_flags(report)
    failed = [step for step in report["steps"] if step["status"] == "error"]
    skipped = [step for step in report["steps"] if step["status"] == "skipped"]
    report["interpretation"] = _interpret(report)
    report["error_category"] = report["interpretation"].get("category")
    if failed:
        report["status"] = "error"
    elif skipped:
        report["status"] = "skipped_non_target_gpu"
    else:
        report["status"] = "ok"
    write_json(report_path, report)
    print(f"RTX 50 first run report: {report_path}")
    print(f"status: {report['status']}")


def _aggregate_detection_flags(report: dict[str, Any]) -> None:
    text_parts = []
    for step in report.get("steps", []):
        text_parts.extend(
            [
                str(step.get("stdout_tail", "")),
                str(step.get("stderr_tail", "")),
                str(step.get("error", "")),
                str(step.get("reason", "")),
            ]
        )
    for item in report.get("reports", {}).values():
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, dict):
            for key in [
                "detected_oom",
                "detected_unsupported_arch",
                "detected_missing_model_index",
                "detected_invalid_safetensors",
            ]:
                report[key] = bool(report.get(key) or content.get(key))
            text_parts.append(str(content.get("error", "")))
            text_parts.append(str(content.get("traceback", "")))
    combined = "\n".join(text_parts).lower()
    report["detected_oom"] = bool(
        report["detected_oom"] or "out of memory" in combined or "cuda oom" in combined
    )
    report["detected_unsupported_arch"] = bool(
        report["detected_unsupported_arch"]
        or "unsupported gpu architecture" in combined
        or "invalid device function" in combined
    )
    report["detected_missing_model_index"] = bool(
        report["detected_missing_model_index"] or "model_index.json" in combined
    )
    report["detected_invalid_safetensors"] = bool(
        report["detected_invalid_safetensors"]
        or "invalid safetensors" in combined
        or "safetensorerror" in combined
        or "safetensors_rust" in combined
    )


def _interpret(report: dict[str, Any]) -> dict[str, Any]:
    if _step_failed(report, "container_env_diagnostics"):
        return {
            "category": "environment/container/driver",
            "message": "Container/env diagnostics failed. Do not touch ApacheOne or TextEncoder-cache.",
        }
    if _step_failed(report, "check_visualgen_supported_model"):
        return {
            "category": "official VisualGen support",
            "message": "Official VisualGen model failed. Treat this as TensorRT-LLM/VisualGen/container/driver/VRAM/RTX50 setup until fixed.",
        }
    if _step_failed(report, "check_visualgen_load_full") or _step_failed(
        report, "check_visualgen_load_txtattn_bf16"
    ):
        return {
            "category": "ApacheOne/Klein-KV layout",
            "message": "Official VisualGen worked, but ApacheOne/Klein-KV load failed. Next step is ApacheOne loader or TensorRT-compatible model directory.",
        }
    strict_report = report.get("reports", {}).get("cached_embeddings_strict_strict_test", {}).get(
        "content"
    )
    if isinstance(strict_report, dict) and "external prompt embeddings" in str(
        strict_report.get("error", "")
    ).lower():
        return {
            "category": "external embeddings unsupported",
            "message": "cached_embeddings_strict hit the expected public VisualGen API limitation. Next step is lower-level VisualGen/Flux2 adapter.",
        }
    if any(step["status"] == "skipped" for step in report["steps"]):
        return {
            "category": "unsupported GPU architecture",
            "message": "GPU runtime steps were skipped because this is not the RTX 50 / Blackwell target path.",
        }
    return {
        "category": "ok",
        "message": "First-run sequence completed. Inspect attached reports and compare visualgen_prompt_text vs cached_embeddings_strict outputs if both exist.",
    }


def _classify_error(text: str) -> str:
    combined = text.lower()
    if "out of memory" in combined or "cuda oom" in combined:
        return "VRAM/OOM"
    if "unsupported gpu architecture" in combined or "invalid device function" in combined:
        return "unsupported GPU architecture"
    if "external prompt embeddings" in combined:
        return "external embeddings unsupported"
    if "model_index.json" in combined or "safetensors" in combined:
        return "ApacheOne/Klein-KV layout"
    if "tensorrt" in combined or "visualgen" in combined:
        return "official VisualGen support"
    return "environment/container/driver"


def _import_version(env: dict, module_name: str) -> str | None:
    for item in env.get("imports", []):
        if item.get("module") == module_name:
            return item.get("version")
    return None


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def _tail(text: str, limit: int = 12000) -> str:
    return text[-limit:]


if __name__ == "__main__":
    raise SystemExit(main())
